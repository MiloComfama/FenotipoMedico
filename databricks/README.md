# Notebooks Databricks · Clustering de Fenotipo Médico (Comfama)

Pipeline para descubrir **subgrupos/fenotipos emergentes** a partir del bloque de
columnas **L → AB** del formulario histórico (hábitos de alimentación, actividad física,
estrés y medicamentos).

## ⚠️ Supuesto clave (validar con negocio)
El bloque **L:AB está diligenciado casi exclusivamente para el grupo Cardiometabólico**
(consulta de ingreso): **4.170 registros con datos, ~94% Cardiometabólico** (3.928), más
206 Digestivo y 36 Mixto con datos parciales. Por eso el modelo segmenta **dentro** de una
cohorte predominantemente cardiometabólica; no distingue Cardiometabólico / Digestivo / Mixto
(esos tienen sus propios bloques de columnas más adelante en el Excel).

## Orden de ejecución
1. **01_ingesta_exploracion.ipynb** — lee el Excel, selecciona L:AB, filtra la cohorte y explora calidad. → tabla `fenotipo_cardio_ingreso_raw`
2. **02_preprocesamiento.ipynb** — limpia/imputa (`SIN DATO`→`Desconocido`, NaN→0). → tabla `fenotipo_features`
3. **03_entrenamiento_clustering.ipynb** — selección de *k* (silueta/codo/Davies-Bouldin), entrena KMeans y registra en MLflow/Unity Catalog.
4. **04_perfilamiento_inferencia.ipynb** — interpreta clusters, valida y guarda asignaciones. → tabla `fenotipo_clusters`

Cada notebook incluye celdas markdown con **propósito, entradas/salidas y trazabilidad** de cada paso.

## Requisitos
- **Databricks Runtime ML** (trae scikit-learn, MLflow, matplotlib).
- Subir `Preguntas y respuestas medico.xlsx` a un **Volume** de Unity Catalog.
- Permisos para crear catálogo/esquema/tablas y registrar modelos en UC.

## Parámetros (widgets)
| Widget | Por defecto | Nota |
|---|---|---|
| `input_path` | `/Volumes/main/fenotipo/raw/Preguntas y respuestas medico.xlsx` | ruta al Excel |
| `sheet_name` | `DATOS ` | ⚠️ incluye el espacio final |
| `catalog` / `schema` | `main` / `fenotipo` | destino de tablas/modelo |
| `k_min` / `k_max` | `2` / `8` | rango de búsqueda de *k* |
| `k_final` | `0` | `0` = elige el mejor por silueta |
| `model_name` | `main.fenotipo.fenotipo_clustering` | modelo en UC |

## Features usadas
- **Numéricas/binarias (10):** `habitos_alim_cantidad`, `hab_*` (6), `med_hipoglicemiantes`, `med_hipolipemiantes`, `med_ninguno`.
- **Categóricas one-hot (5):** `actividad_tipo`, `actividad_duracion`, `actividad_frecuencia`, `estres_altas_cargas`, `estres_tecnica_manejo`.
- Se descartan los textos crudos `habitos_alim_texto` (L) y `medicamentos_texto` (Y) por ser redundantes.

## Formato e importación
Notebooks **Jupyter `.ipynb`** (compatibles nativamente con Databricks).
Impórtalos con **Workspace → Import → File** (o arrástralos), o versiónalos vía **Repos**.
