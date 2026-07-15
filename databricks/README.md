# Notebooks · Clasificación de Fenotipo Médico (Comfama)

Pipeline de datos y modelo de clasificación (K-Means) entrenado con el
**histórico real** de encuestas y datos clínicos del programa, que reemplaza
al clasificador de reglas de la app (`app/domain/classifier.RuleBasedClassifier`)
por uno entrenado (`app/domain/ml_model.TrainedClassifier`), sin cambiar la
interfaz `Classifier` ni la UI.

## Fuente de datos

`data/HISTORICO_MF_HACKATON2026 (1).xlsx`, hoja `Datos`: **16.204 filas ×
20 columnas**, 4.388 pacientes únicos (~3.7 consultas por paciente en
promedio). A diferencia del histórico anterior (`Preguntas y respuestas
medico.xlsx`, bloque de columnas L:AB, ~4.170 registros casi exclusivamente
del grupo Cardiometabólico), este archivo trae en una sola hoja las
respuestas de estilo de vida **y** las mediciones clínicas (IMC, perímetro
abdominal, presión arterial, colesterol total/HDL, HbA1c) de toda la
población del programa — no requiere aislar un bloque de columnas ni asumir
una cohorte predominante.

## Orden de ejecución

1. **01_ingesta_exploracion.ipynb** — carga el Excel, corrige tipos (decimales
   con coma), descarta ~30 registros con valores fisiológicamente imposibles,
   explora nulos y distribución de variables. → `data/_processed/01_raw_clean.csv`
2. **02_preprocesamiento.ipynb** — deriva las 19 features del modelo (mismas
   que usa `app/domain/ml_features.py` en producción) y calcula las medianas
   de imputación. → `data/_processed/02_features.csv`
3. **03_entrenamiento_clustering.ipynb** — **EDA de segmentación**: K-Means
   conjunto, búsqueda de *k* (2 a 8) por silueta/inercia, perfilamiento de
   los clusters y su interpretación clínica.
4. **04_perfilamiento_inferencia.ipynb** — **modelo de clasificación**: dos
   K-Means (k=2) por eje clínico, puntajes de afinidad, regla de decisión
   para "Mixto", guardado de `data/model.pkl` y verificación de la
   integración con la app.

Cada notebook está ejecutado (con sus salidas reales) y documenta propósito,
entradas/salidas y trazabilidad de cada paso.

> ⚠️ Adaptados para correr en **Jupyter local puro** (pandas / scikit-learn,
> sin `dbutils` / Spark / Delta) para poder validar el pipeline de punta a
> punta. La lógica es igual de válida en Databricks: basta con sustituir las
> celdas de lectura/escritura local por `dbutils.widgets` y tablas de Unity
> Catalog si se despliega allí.

## Categorías definidas a partir del EDA

El clustering conjunto (k=3, silueta 0.144 — óptimo entre 2 y 8) revela 3
segmentos naturales en la población:

| Segmento | ~Tamaño | Rasgos distintivos |
|---|---|---|
| **Digestivo** | ~22% | Síntomas gastrointestinales frecuentes, uso de antiácidos/IBP, alto estrés |
| **Cardiometabólico** | ~15% | IMC, perímetro abdominal, HbA1c y uso de hipoglicemiantes/hipolipemiantes elevados |
| **Bajo riesgo** | ~63% | Sin síntomas digestivos relevantes ni marcadores cardiometabólicos elevados |

**No existe un cuarto cluster natural "Mixto"** (se probó también k=4, 5 y
6): lo usual es que un paciente tenga predominantemente un eje afectado, no
ambos a la vez. Por eso "Mixto" se deriva con una regla explícita (ambos
puntajes de afinidad en el cuartil superior) en el notebook 04, en vez de
salir directamente del clustering — el mismo enfoque conceptual del antiguo
clasificador de reglas (dos puntajes + una regla de decisión), pero con los
puntajes aprendidos de los datos reales en vez de pesos manuales.

## Modelo de clasificación final

- Dos K-Means (k=2) **independientes**, uno con las 9 features
  cardiometabólicas y otro con las 2 digestivas — silueta 0.32 y 0.83
  respectivamente (mucho más limpia que extraerla del clustering conjunto de
  19 dimensiones).
- La afinidad de un paciente a un eje es su percentil de cercanía al
  centroide "de riesgo" de ese eje, respecto al histórico de entrenamiento.
- Si falta *toda* la información clínica de un eje (p. ej. un paciente que
  aún no ha tenido su primera cita médica), el modelo devuelve un puntaje
  neutral en vez de uno distorsionado por imputación.
- **Viabilidad**: coincide con el criterio clínico esperado y con el
  clasificador de reglas anterior sobre perfiles claros; ver la sección de
  conclusiones del notebook 04 para las limitaciones conocidas (no hay
  "Mixto" natural, no hay variable de sexo en el histórico para ajustar el
  perímetro abdominal, y al ser no supervisado no hay *ground truth* contra
  la cual medir accuracy — se recomienda validación progresiva del equipo
  médico vía la reclasificación manual ya disponible en la consola médica).

## Integración con la app

`app/domain/classifier.get_classifier()` detecta automáticamente
`data/model.pkl` y usa `TrainedClassifier` (`app/domain/ml_model.py`) en vez
de `RuleBasedClassifier`, sin cambios en la UI. Para reentrenar el modelo
fuera de un notebook: `py scripts/train_model.py`.

## Requisitos

```bash
py -m pip install pandas scikit-learn scipy matplotlib openpyxl jupyter nbformat nbconvert
```
