"""Entrena el clasificador de fenotipo a partir del histórico real y guarda
el modelo en ``data/model.pkl``.

Uso:  py scripts/train_model.py

Fuente de datos: ``data/HISTORICO_MF_HACKATON2026 (1).xlsx`` (hoja "Datos").
El proceso completo (EDA, justificación de las categorías y perfilamiento de
clusters) está documentado en los notebooks de ``databricks/``; este script
es la versión ejecutable y reproducible de la etapa de entrenamiento.

Enfoque de modelado (resumen):
  1. Se homologan y limpian las columnas del Excel (decimales con coma,
     valores fisiológicamente imposibles) y se derivan features numéricas y
     binarias (mismas que usa la app en tiempo real, ver
     ``app/domain/ml_features.py``).
  2. Un K-Means conjunto (k=3) sobre las 19 features revela los 3 segmentos
     naturales de la población: uno con síntomas gastrointestinales
     elevados ("Digestivo"), uno con marcadores cardiometabólicos elevados
     ("Cardiometabólico") y uno de bajo riesgo/sin síntomas predominantes.
     Este clustering conjunto es la respuesta al EDA solicitado y queda
     documentado (silueta, tamaños, perfiles) en el notebook 03.
  3. Para la clasificación de cada consulta se usan DOS K-Means auxiliares
     de k=2, cada uno entrenado solo con las features relevantes a un eje
     clínico (cardiometabólico / digestivo). Esto separa mucho mejor cada
     eje que extraer la afinidad del clustering conjunto de 19 dimensiones
     (silueta 0.32 y 0.68 respectivamente, frente a 0.14 del conjunto).
  4. La afinidad de un paciente a cada eje es 1 menos su percentil de
     distancia al centroide "de riesgo" de ese eje (calculado contra la
     distribución de distancias del histórico). Si ambas afinidades caen en
     el cuartil superior (>= percentil 75) de su propio eje, se asigna
     "Mixto"; si no, gana el eje con mayor percentil. Esta lógica reemplaza
     los pesos y umbrales manuales del clasificador de reglas por valores
     aprendidos de los datos, conservando la misma idea (dos puntajes
     continuos + una regla de decisión) para no romper la interfaz
     ``Classifier`` ni la UI existente.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.ml_features import CARDIO_FEATURES, DIGEST_FEATURES, FEATURE_NAMES  # noqa: E402

SRC_PATH = Path(__file__).resolve().parent.parent / "data" / "HISTORICO_MF_HACKATON2026 (1).xlsx"
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"

RAW_COLUMNS = [
    "documento", "habitos_alim", "actividad_tipo", "actividad_duracion",
    "actividad_frecuencia", "estres_alto", "ansioso", "nervioso",
    "tecnica_estres", "medicamentos", "perimetro_abdominal", "peso", "talla",
    "imc", "pa_sistolica", "pa_diastolica", "sintomas_gi",
    "colesterol_total", "colesterol_hdl", "hba1c",
]

MIXTO_PERCENTILE = 0.75  # umbral (cuartil superior) para considerar un eje "elevado"
RANDOM_STATE = 42


def _to_float_co(series: pd.Series) -> pd.Series:
    """Convierte columnas con coma decimal (formato es-CO) a float."""
    return pd.to_numeric(series.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def _has_any(text: str, needles: list[str]) -> bool:
    t = str(text).lower()
    return any(n in t for n in needles)


def load_and_engineer() -> pd.DataFrame:
    df = pd.read_excel(SRC_PATH, sheet_name="Datos")
    df.columns = RAW_COLUMNS

    df["perimetro_abdominal"] = _to_float_co(df["perimetro_abdominal"])
    df["pa_diastolica"] = _to_float_co(df["pa_diastolica"])

    # Descarta registros fisiológicamente imposibles (típico de captura manual).
    mask_valid = (
        df["imc"].between(10, 70)
        & df["perimetro_abdominal"].between(40, 180)
        & df["pa_sistolica"].between(70, 220)
        & df["pa_diastolica"].between(40, 150)
        & df["hba1c"].between(3, 16)
    )
    n_before = len(df)
    df = df[mask_valid].reset_index(drop=True)
    print(f"Registros: {n_before} -> {len(df)} tras filtrar outliers imposibles "
          f"({n_before - len(df)} descartados).")

    df["n_habitos_saludables"] = df["habitos_alim"].apply(
        lambda t: 0 if "ninguno" in str(t).lower() else len([p for p in str(t).split(";") if p.strip()])
    )
    df["hace_actividad"] = (~df["actividad_tipo"].str.lower().str.contains("ninguno", na=False)).astype(int)
    df["actividad_fuerza"] = df["actividad_tipo"].apply(lambda t: int(_has_any(t, ["fuerza"])))
    freq_map = {"ninguno": 0, "1 vez": 1, "2 veces": 2, "3 veces": 3, "4 veces": 4, "5 o más veces": 5}
    df["frecuencia_actividad_num"] = (
        df["actividad_frecuencia"].str.strip().str.lower().str.rstrip(";").map(freq_map).fillna(0)
    )
    df["estres_alto_bin"] = (df["estres_alto"].str.strip().str.lower() == "sí").astype(int)
    df["ansioso_bin"] = (df["ansioso"].str.strip().str.lower() == "sí").astype(int)
    df["nervioso_bin"] = (df["nervioso"].str.strip().str.lower() == "sí").astype(int)
    df["tecnica_estres_bin"] = (df["tecnica_estres"].str.strip().str.lower() == "sí").astype(int)
    df["usa_hipoglicemiante"] = df["medicamentos"].apply(lambda t: int(_has_any(t, ["hipoglicemiante"])))
    df["usa_hipolipemiante"] = df["medicamentos"].apply(lambda t: int(_has_any(t, ["hipolipemiante"])))
    df["usa_antiacido_ibp"] = df["medicamentos"].apply(
        lambda t: int(_has_any(t, ["antiácid", "antiacid", "bomba de protones"]))
    )
    df["n_sintomas_gi"] = df["sintomas_gi"].apply(
        lambda t: 0 if "ninguno" in str(t).lower() else len([p for p in str(t).split(";") if p.strip()])
    )
    return df


def main() -> None:
    df = load_and_engineer()
    X = df[FEATURE_NAMES].copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs_df = pd.DataFrame(Xs, columns=FEATURE_NAMES)

    # --- 1) Clustering conjunto (k=3): EDA / segmentación poblacional -------
    print("\n=== Clustering conjunto k=3 (segmentación de la población) ===")
    km_joint = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    joint_labels = km_joint.fit_predict(Xs)
    sil_joint = silhouette_score(Xs, joint_labels, sample_size=5000, random_state=RANDOM_STATE)
    sizes = pd.Series(joint_labels).value_counts().sort_index()
    print(f"silueta: {sil_joint:.4f} | tamaños de cluster: {sizes.to_dict()}")
    profile = df.assign(cluster=joint_labels).groupby("cluster")[FEATURE_NAMES].mean().round(2)
    print(profile.T)

    # --- 2) Sub-clusters por eje (k=2 cada uno): motor de clasificación -----
    Xc = Xs_df[CARDIO_FEATURES].values
    Xd = Xs_df[DIGEST_FEATURES].values
    km_cardio = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit(Xc)
    km_digest = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit(Xd)
    sil_cardio = silhouette_score(Xc, km_cardio.labels_, sample_size=5000, random_state=RANDOM_STATE)
    sil_digest = silhouette_score(Xd, km_digest.labels_, sample_size=5000, random_state=RANDOM_STATE)
    print(f"\nsilueta cardio (k=2): {sil_cardio:.4f} | silueta digestivo (k=2): {sil_digest:.4f}")

    hi_cardio_cluster = int(df.groupby(km_cardio.labels_)["hba1c"].mean().idxmax())
    hi_digest_cluster = int(df.groupby(km_digest.labels_)["n_sintomas_gi"].mean().idxmax())

    dist_cardio = np.linalg.norm(Xc - km_cardio.cluster_centers_[hi_cardio_cluster], axis=1)
    dist_digest = np.linalg.norm(Xd - km_digest.cluster_centers_[hi_digest_cluster], axis=1)

    pct_cardio = 1 - (rankdata(dist_cardio) / len(dist_cardio))
    pct_digest = 1 - (rankdata(dist_digest) / len(dist_digest))
    # OJO: el umbral es el percentil MIXTO_PERCENTILE en sí mismo (pct_cardio/
    # pct_digest ya son percentiles por construcción). Volver a calcular
    # np.percentile(pct_digest, 75) es incorrecto cuando hay muchos empates
    # (p. ej. la mayoría sin síntomas digestivos): el percentil 75 de una
    # distribución con >75% de valores empatados en el mínimo colapsa al
    # valor empatado en vez de reflejar el cuartil superior real.
    thr_cardio = MIXTO_PERCENTILE
    thr_digest = MIXTO_PERCENTILE

    final_label = np.where(
        (pct_cardio >= thr_cardio) & (pct_digest >= thr_digest), "Mixto",
        np.where(pct_cardio >= pct_digest, "Cardiometabólico", "Digestivo"),
    )
    print("\n=== Distribución final de fenotipos (todo el histórico) ===")
    print(pd.Series(final_label).value_counts())
    print("\nPerfil clínico promedio por fenotipo asignado:")
    print(df.assign(fen=final_label).groupby("fen")[FEATURE_NAMES].mean().round(2).T)

    bundle = {
        "version": 2,
        "feature_names": FEATURE_NAMES,
        "cardio_features": CARDIO_FEATURES,
        "digest_features": DIGEST_FEATURES,
        "medians": medians.to_dict(),
        "scaler": scaler,
        "km_cardio": km_cardio,
        "km_digest": km_digest,
        "hi_cardio_cluster": hi_cardio_cluster,
        "hi_digest_cluster": hi_digest_cluster,
        "dist_cardio_train": dist_cardio,
        "dist_digest_train": dist_digest,
        "mixto_percentile": MIXTO_PERCENTILE,
        "joint_kmeans": km_joint,
        "joint_silhouette": sil_joint,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
