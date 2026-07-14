# Fenotipo Médico · Segmentación Inteligente de Pacientes (Comfama)

Prototipo del programa de **Medicina Funcional de Comfama** que segmenta
pacientes por fenotipo clínico (**Cardiometabólico · Digestivo · Mixto**)
integrando hábitos de vida y datos clínicos.

> ⚠️ Prototipo con fines de demostración. Las recomendaciones no reemplazan la
> valoración médica. El manejo de datos clínicos está sujeto a la autorización
> del paciente y al marco de habeas data y políticas internas de Comfama.

## Características

- **Vista del paciente**: ingreso por documento, cuestionario **conversacional**
  (chat con Opus 4.8, con flujo guiado de respaldo), tablero con gráficas,
  clasificación y recomendaciones. Regla de **cita de seguimiento cada 30 días**.
- **Vista del médico**: búsqueda de pacientes, ingreso de datos básicos
  (peso, estatura, **IMC autocalculado**, perímetro abdominal, presión arterial),
  laboratorios (colesterol total, etc.), y **reclasificación por criterio médico**.
- **Historial de consultas** numeradas (#1, #2, #3…) por paciente.

## Arquitectura

```
app/
  main.py            Entrada Streamlit (navegación)
  config.py          Rutas, paleta de marca, reglas
  db/                SQLAlchemy: modelos, sesión, repositorio
  domain/            Cuestionario, clasificador (reglas → swappable a IA), recomendaciones
  services/          chat.py (Opus 4.8 + respaldo), intake.py (clasificar + persistir)
  ui/                branding, charts, patient_view, doctor_view
  assets/            logo.svg, styles.css
data/
  questionnaire.yaml Fuente de verdad de las preguntas
  fenotipo.db        SQLite (se crea solo; ignorado por git)
scripts/seed_demo.py Pacientes de demostración
```

### Base de datos (recomendación para el prototipo)
**SQLite + SQLAlchemy ORM**. Esquema normalizado y listo para migrar a
SQL Server / Microsoft Fabric:
`patients` 1—N `consultations` 1—N `survey_answers` (llave-valor / EAV),
`consultations` 1—1 `measurements`, 1—N `lab_results`.
El modelo EAV permite evolucionar el cuestionario sin cambiar el esquema.

### Modelo de clasificación
Hoy: clasificador **basado en reglas** transparente (`app/domain/classifier.py`),
detrás de la interfaz `Classifier`. El futuro modelo entrenado con los datos
históricos se conecta guardándolo en `data/model.pkl` sin tocar la UI.

## Puesta en marcha

```bash
# 1. (opcional) entorno virtual
py -m venv .venv && .venv\Scripts\activate

# 2. dependencias
py -m pip install -r requirements.txt

# 3. clave de IA para el chat (opcional; sin ella se usa el flujo guiado)
copy .env.example .env   # y completa ANTHROPIC_API_KEY

# 4. datos de demostración (opcional)
py scripts/seed_demo.py

# 5. ejecutar
streamlit run app/main.py
```

Abrir http://localhost:8501

- **Pacientes demo**: `CC 1001` (Digestivo), `CC 1002` (Cardiometabólico).
- **PIN consola médica**: `comfama`.
