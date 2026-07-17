"""Entrena el clasificador de fenotipo a partir del histórico real y guarda
el modelo en ``data/model.pkl``.

Uso:  py scripts/train_model.py

Fuente de datos: ``data/HISTORICO_MF_HACKATON2026 (1).xlsx`` (hoja "Datos").
El proceso completo (EDA, búsqueda de features/k y perfilamiento de
clusters) está documentado en los notebooks de ``databricks/``; este script
es la versión ejecutable y reproducible de la etapa de entrenamiento.

Enfoque de modelado (resumen; ver notebooks 03 y 04 para el detalle):
  1. Se homologan y limpian las columnas del Excel (decimales con coma,
     valores fisiológicamente imposibles) y se derivan las features
     numéricas y binarias que usa la app en tiempo real (ver
     ``app/domain/ml_features.FEATURE_NAMES``).
  2. Se probaron varios subconjuntos de features (todas, solo estilo de
     vida, solo clínicas, clínicas + síntomas GI) y k de 2 a 9. El mejor
     compromiso silueta/interpretabilidad NO es el clustering conjunto de
     19 dimensiones usado anteriormente (silueta ~0.14-0.16), sino un
     K-Means (k=5) sobre las 11 features clínicas + síntomas digestivos
     (silueta ~0.19), que además separa el antiguo cluster
     "Cardiometabólico" en tres fenotipos clínicamente distintos:
     Obesidad, Dislipidemia y Glicemia — cada uno con una vía de
     intervención distinta. "Bajo riesgo" y "Digestivo" se mantienen como
     clusters propios (ver notebook 03 para la comparación completa).
  3. La afinidad de un paciente a cada fenotipo es 1 menos su percentil de
     distancia al centroide de ese cluster (calculado contra la
     distribución de distancias del histórico); el fenotipo final es el de
     mayor afinidad. Esto reemplaza la combinación "dos ejes + regla Mixto"
     del modelo anterior por una asignación directa al vecino más cercano,
     ahora que hay clusters propios para cada perfil clínico relevante.
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

from app.domain.ml_features import FEATURE_NAMES  # noqa: E402

SRC_PATH = Path(__file__).resolve().parent.parent / "data" / "HISTORICO_MF_HACKATON2026 (1).xlsx"
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"

RAW_COLUMNS = [
    "documento", "habitos_alim", "actividad_tipo", "actividad_duracion",
    "actividad_frecuencia", "estres_alto", "ansioso", "nervioso",
    "tecnica_estres", "medicamentos", "perimetro_abdominal", "peso", "talla",
    "imc", "pa_sistolica", "pa_diastolica", "sintomas_gi",
    "colesterol_total", "colesterol_hdl", "hba1c",
]

N_CLUSTERS = 5
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
    df["peso"] = _to_float_co(df["peso"])
    df["talla"] = _to_float_co(df["talla"])
    df["pa_diastolica"] = _to_float_co(df["pa_diastolica"])

    # Descarta registros fisiológicamente imposibles (típico de captura manual).
    mask_valid = (
        df["imc"].between(10, 70)
        & df["talla"].between(1.0, 2.2)
        & df["peso"].between(25, 250)
        & df["perimetro_abdominal"].between(40, 200)
        & df["pa_sistolica"].between(60, 250)
        & df["pa_diastolica"].between(30, 150)
        & df["colesterol_total"].between(50, 500)
        & df["colesterol_hdl"].between(10, 150)
        & df["hba1c"].between(3.5, 20)
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
    df["estres_alto_bin"] = (df["estres_alto"].str.strip().str.upper() == "SÍ").astype(int)
    df["ansioso_bin"] = (df["ansioso"].str.strip().str.upper() == "SÍ").astype(int)
    df["nervioso_bin"] = (df["nervioso"].str.strip().str.upper() == "SÍ").astype(int)
    df["tecnica_estres_bin"] = (df["tecnica_estres"].str.strip().str.upper() == "SÍ").astype(int)
    df["usa_hipoglicemiante"] = df["medicamentos"].apply(lambda t: int(_has_any(t, ["hipoglicemiante"])))
    df["usa_hipolipemiante"] = df["medicamentos"].apply(lambda t: int(_has_any(t, ["hipolipemiante"])))
    df["usa_antiacido_ibp"] = df["medicamentos"].apply(
        lambda t: int(_has_any(t, ["antiácid", "antiacid", "bomba de protones"]))
    )
    df["n_sintomas_gi"] = df["sintomas_gi"].apply(
        lambda t: 0 if "ninguno" in str(t).lower() else len([p for p in str(t).split(";") if p.strip()])
    )
    return df


def _label_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Asigna un nombre clínico a cada cluster a partir de su perfil
    (medicamento/marcador dominante), en vez de fijar el índice numérico de
    KMeans (arbitrario y dependiente de la semilla)."""
    profile = df.assign(cluster=labels).groupby("cluster")[FEATURE_NAMES].mean()
    remaining = list(profile.index)
    mapping: dict[int, str] = {}

    digestivo = profile.loc[remaining, "n_sintomas_gi"].idxmax()
    mapping[digestivo] = "Digestivo"
    remaining.remove(digestivo)

    glicemia = profile.loc[remaining, "usa_hipoglicemiante"].idxmax()
    mapping[glicemia] = "Glicemia"
    remaining.remove(glicemia)

    dislipidemia = profile.loc[remaining, "usa_hipolipemiante"].idxmax()
    mapping[dislipidemia] = "Dislipidemia"
    remaining.remove(dislipidemia)

    obesidad = profile.loc[remaining, "imc"].idxmax()
    mapping[obesidad] = "Obesidad"
    remaining.remove(obesidad)

    assert len(remaining) == 1, f"Sobraron clusters sin etiquetar: {remaining}"
    mapping[remaining[0]] = "Bajo riesgo"
    return mapping


def main() -> None:
    df = load_and_engineer()
    X = df[FEATURE_NAMES].copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10).fit(Xs)
    sil = silhouette_score(Xs, km.labels_, sample_size=6000, random_state=RANDOM_STATE)
    sizes = pd.Series(km.labels_).value_counts().sort_index()
    print(f"\n=== K-Means k={N_CLUSTERS} sobre features clínicas + GI ===")
    print(f"silueta: {sil:.4f} | tamaños de cluster: {sizes.to_dict()}")

    cluster_to_phenotype = _label_clusters(df, km.labels_)
    print("\nMapeo cluster -> fenotipo:", cluster_to_phenotype)

    phenotype_labels = pd.Series(km.labels_).map(cluster_to_phenotype)
    print("\n=== Distribución final de fenotipos (todo el histórico) ===")
    print(phenotype_labels.value_counts())
    print("\nPerfil clínico promedio por fenotipo asignado:")
    print(df.assign(fen=phenotype_labels.values).groupby("fen")[FEATURE_NAMES].mean().round(2).T)

    dist_train = {}
    for cluster_idx, phenotype in cluster_to_phenotype.items():
        dist_train[phenotype] = np.linalg.norm(Xs - km.cluster_centers_[cluster_idx], axis=1)

    bundle = {
        "version": 3,
        "feature_names": FEATURE_NAMES,
        "medians": medians.to_dict(),
        "scaler": scaler,
        "kmeans": km,
        "cluster_to_phenotype": cluster_to_phenotype,
        "dist_train": dist_train,
        "silhouette": sil,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
