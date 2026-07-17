# Notebooks · Clasificación de Fenotipo Médico (Comfama)

Pipeline de datos y modelo de clasificación (K-Means) entrenado con el
**histórico real** de encuestas y datos clínicos del programa, que reemplaza
al clasificador de reglas de la app (`app/domain/classifier.RuleBasedClassifier`)
por uno entrenado (`app/domain/ml_model.TrainedClassifier`), sin cambiar la
interfaz `Classifier` ni la UI.

## Fuente de datos

`data/HISTORICO_MF_HACKATON2026 (1).xlsx`, hoja `Datos`: **16.204 filas ×
20 columnas**, 4.388 pacientes únicos (~3.7 consultas por paciente en
promedio). Trae en una sola hoja las respuestas de estilo de vida **y** las
mediciones clínicas (IMC, perímetro abdominal, presión arterial, colesterol
total/HDL, HbA1c) de toda la población del programa.

## Orden de ejecución

1. **01_ingesta_exploracion.ipynb** — carga el Excel, corrige tipos (decimales
   con coma), descarta ~29 registros con valores fisiológicamente imposibles,
   explora nulos y distribución de variables. → `data/_processed/01_raw_clean.csv`
2. **02_preprocesamiento.ipynb** — deriva las 11 features clínicas + síntomas
   GI del modelo (mismas que usa `app/domain/ml_features.FEATURE_NAMES` en
   producción), las 8 features de estilo de vida (solo para explicaciones) y
   calcula las medianas de imputación. → `data/_processed/02_features.csv`
3. **03_entrenamiento_clustering.ipynb** — **EDA de segmentación**: compara 4
   subconjuntos de features (todas, solo estilo de vida, solo clínicas, y
   clínicas + síntomas GI) en un sweep de *k* (2 a 8) por silueta/inercia, y
   perfila e interpreta el ganador.
4. **04_perfilamiento_inferencia.ipynb** — **modelo de clasificación**: un
   K-Means (k=5) sobre las features ganadoras, afinidad por fenotipo,
   guardado de `data/model.pkl` y verificación de la integración con la app.

Cada notebook está ejecutado (con sus salidas reales) y documenta propósito,
entradas/salidas y trazabilidad de cada paso.

> ⚠️ Adaptados para correr en **Jupyter local puro** (pandas / scikit-learn,
> sin `dbutils` / Spark / Delta) para poder validar el pipeline de punta a
> punta. La lógica es igual de válida en Databricks: basta con sustituir las
> celdas de lectura/escritura local por `dbutils.widgets` y tablas de Unity
> Catalog si se despliega allí.

## Categorías definidas a partir del EDA

El modelo anterior clusterizaba las 19 features (clínicas + estilo de vida)
juntas y encontraba 3 segmentos (silueta ~0.14-0.16 en k=2-3): Digestivo,
Cardiometabólico y Bajo riesgo. El notebook 03 compara explícitamente esa
configuración contra otras tres (solo estilo de vida, solo clínicas, y
clínicas + síntomas GI) en un sweep de *k* de 2 a 8, y encuentra un máximo
local más alto y más informativo con **clínicas + síntomas GI, k=5**
(silueta ~0.186): separa el antiguo cluster único "Cardiometabólico" en
**tres** perfiles con vías de intervención distintas.

| Fenotipo | ~Tamaño | Rasgo distintivo |
|---|---|---|
| **Bajo riesgo** | ~33% | Los marcadores más bajos en las 11 dimensiones clínicas |
| **Obesidad** | ~29% | IMC y perímetro abdominal más altos, presión arterial elevada |
| **Glicemia** | ~14% | ~99% usa hipoglicemiantes, HbA1c más alta (prediabetes/diabetes) |
| **Dislipidemia** | ~14% | ~100% usa hipolipemiantes, colesterol total más alto |
| **Digestivo** | ~10% | Síntomas gastrointestinales frecuentes, 100% usa antiácidos/IBP |

**Sigue sin existir un cluster natural "Mixto"** (confirmado también en este
sweep más amplio, k=2 a 8 en las 4 configuraciones): un paciente puede tener
afinidad alta a más de un fenotipo (visible en `scores`), pero no hay un
cluster propio para esa combinación — no se fuerza una categoría "Mixto"
artificial.

Las features de **estilo de vida se probaron dentro del clustering y
empeoraban tanto la silueta (~0.16-0.33 según *k*, pero mezclando señal
clínica con conductual) como la interpretación clínica** — se excluyeron de
la definición del fenotipo y se usan únicamente para las explicaciones en
lenguaje sencillo que ve el paciente (`rationale`).

## Modelo de clasificación final

- Un único K-Means (k=5) sobre las 11 features clínicas + síntomas GI —
  silueta 0.186 (frente a 0.144-0.164 del clustering conjunto de 19
  dimensiones que usaba el modelo anterior).
- Cada cluster se etiqueta por su rasgo dominante (medicamento o marcador más
  característico), no por el índice numérico de KMeans — ver `_label_clusters`
  en `scripts/train_model.py`.
- La afinidad de un paciente a un fenotipo es su percentil de cercanía al
  centroide de ese cluster, respecto al histórico de entrenamiento; el
  fenotipo final es el de mayor afinidad.
- Si falta *toda* la información clínica relevante (p. ej. un paciente que
  aún no ha tenido su primera cita médica), el modelo devuelve un puntaje
  neutral en vez de uno distorsionado por imputación.
- **Limitaciones conocidas** (ver conclusiones del notebook 04): no hay
  "Mixto" natural; no hay variable de sexo en el histórico para ajustar el
  perímetro abdominal dentro del clustering no supervisado; al ser no
  supervisado no hay *ground truth* contra la cual medir accuracy — se
  recomienda validación progresiva del equipo médico vía la reclasificación
  manual ya disponible en la consola médica.

## Integración con la app

`app/domain/classifier.get_classifier()` detecta automáticamente
`data/model.pkl` y usa `TrainedClassifier` (`app/domain/ml_model.py`) en vez
de `RuleBasedClassifier`, sin cambios en la UI. Los 5 fenotipos, sus colores
(`app/config.PHENOTYPE_COLORS`) y protocolos de recomendación
(`app/domain/recommendations.PROTOCOLS`) están alineados con las categorías
de este notebook. Para reentrenar el modelo fuera de un notebook:
`py scripts/train_model.py`.

> Nota de compatibilidad: las consultas ya guardadas en `data/fenotipo.db`
> con las categorías anteriores (Cardiometabólico/Mixto) no se migran
> automáticamente — la UI las muestra sin color de fenotipo ni protocolo de
> recomendación (no falla), hasta que el paciente complete una nueva consulta
> o el médico reclasifique.

## Requisitos

```bash
py -m pip install pandas scikit-learn scipy matplotlib openpyxl jupyter nbformat nbconvert
```
