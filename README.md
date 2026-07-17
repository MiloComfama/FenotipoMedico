# Fenotipo Médico · Segmentación Inteligente de Pacientes (Comfama)

Prototipo del programa de **Medicina Funcional de Comfama** que segmenta
pacientes por fenotipo clínico (**Obesidad · Dislipidemia · Glicemia ·
Digestivo · Bajo riesgo**) integrando hábitos de vida y datos clínicos.

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
databricks/          Notebooks de clustering (pipeline ML offline, ver su propio README)
scripts/seed_demo.py Pacientes de demostración
```

### Diagrama de arquitectura

Capas de la aplicación y sistemas externos con los que se integra:

```mermaid
graph TB
    subgraph Cliente["Cliente"]
        Browser["Navegador · Streamlit UI"]
    end

    subgraph App["Aplicación Streamlit (app/)"]
        Main["main.py<br/>Router + tema"]
        UIPatient["ui/patient_view.py<br/>Vista paciente"]
        UIDoctor["ui/doctor_view.py<br/>Vista médico"]
        Services["services/<br/>chat.py · intake.py"]
        Domain["domain/<br/>questionnaire · classifier · recommendations"]
        DataAccess["db/<br/>repository · database (SQLAlchemy)"]
    end

    subgraph Datos["Persistencia"]
        SQLite[("SQLite<br/>fenotipo.db")]
        FutureDB[("futuro:<br/>SQL Server / Microsoft Fabric")]
    end

    subgraph IA["IA conversacional"]
        Foundry["Azure AI Foundry<br/>(AnthropicFoundry)"]
        AnthropicAPI["API directa de Anthropic<br/>(alternativa)"]
        Opus["Claude Opus 4.8 · 'Fénix'"]
    end

    subgraph MLOffline["Pipeline ML offline (Databricks)"]
        Notebooks["Notebooks 01→04<br/>clustering KMeans"]
        ModelPkl["model.pkl<br/>(futuro clasificador entrenado)"]
    end

    Browser --> Main
    Main --> UIPatient
    Main --> UIDoctor
    UIPatient --> Services
    UIDoctor --> Services
    Services --> Domain
    Services --> DataAccess
    UIPatient --> DataAccess
    UIDoctor --> DataAccess
    Domain --> DataAccess
    DataAccess --> SQLite
    SQLite -.->|migración futura| FutureDB
    Services --> Foundry
    Services -->|si no hay clave / falla| AnthropicAPI
    Foundry --> Opus
    AnthropicAPI --> Opus
    Notebooks --> ModelPkl
    ModelPkl -.->|reemplaza al clasificador de reglas| Domain
```

### Diagrama de componentes

El cuestionario de ingreso usa un patrón *strategy* intercambiable (`BaseIntake`),
igual que la clasificación (`Classifier`) — así el prototipo basado en reglas y
el futuro modelo entrenado conviven detrás de la misma interfaz:

```mermaid
classDiagram
    class BaseIntake {
        <<abstract>>
        +Questionnaire q
        +bool is_first
        +dict answers
        +start() str
        +handle(texto) ChatTurn
        +progress() tuple
    }
    class ScriptedIntake {
        +start() str
        +handle(texto) ChatTurn
    }
    class LLMIntake {
        -client AnthropicFoundry~Anthropic
        +start() str
        +handle(texto) ChatTurn
        -_run_turn() str
        -_apply_tool(nombre, payload) str
    }
    BaseIntake <|-- ScriptedIntake : sin clave de IA
    BaseIntake <|-- LLMIntake : con Opus 4.8
    LLMIntake --> Questionnaire : usa

    class Questionnaire {
        +int version
        +list~Section~ sections
        +questions_for(is_first) list
    }
    class Classifier {
        <<interface>>
        +predict(features) ClassificationResult
    }
    class RuleBasedClassifier
    class TrainedClassifier {
        <<futuro>>
    }
    RuleBasedClassifier ..|> Classifier
    TrainedClassifier ..|> Classifier : model.pkl
```

### Diagrama de flujo (cuestionario conversacional → clasificación)

```mermaid
sequenceDiagram
    actor Paciente
    participant UI as Streamlit UI
    participant Fenix as Fénix (LLMIntake)
    participant Azure as Azure AI Foundry<br/>(Claude Opus 4.8)
    participant Intake as intake_service
    participant Clf as Classifier
    participant DB as SQLite (SQLAlchemy)

    Paciente->>UI: Ingresa documento
    UI->>Fenix: create_intake() + start()
    Fenix->>Azure: messages.create(system, tools)
    Azure-->>Fenix: pregunta + herramienta guardar_respuesta
    Fenix-->>UI: mensaje con opciones numeradas
    UI-->>Paciente: muestra pregunta

    loop hasta completar el cuestionario
        Paciente->>UI: responde (número o texto)
        UI->>Fenix: handle(texto)
        Fenix->>Azure: nuevo turno + historial
        Azure-->>Fenix: guardar_respuesta(key, valor) o aclaración
        Fenix-->>UI: siguiente pregunta / respuesta a la duda
    end

    Azure-->>Fenix: finalizar(mensaje_cierre)
    Fenix-->>UI: cuestionario completo
    UI->>Intake: save_new_consultation(respuestas)
    Intake->>Clf: predict(features)
    Clf-->>Intake: fenotipo + puntajes + rationale
    Intake->>DB: crear Patient / Consultation / SurveyAnswer
    DB-->>Intake: consulta #N persistida
    Intake-->>UI: resultado (fenotipo, scores)
    UI-->>Paciente: tablero con clasificación y recomendaciones
```

### Base de datos (recomendación para el prototipo)
**SQLite + SQLAlchemy ORM**. Esquema normalizado y listo para migrar a
SQL Server / Microsoft Fabric:
`patients` 1—N `consultations` 1—N `survey_answers` (llave-valor / EAV),
`consultations` 1—1 `measurements`, 1—N `lab_results`.
El modelo EAV permite evolucionar el cuestionario sin cambiar el esquema.

```mermaid
erDiagram
    PATIENT ||--o{ CONSULTATION : tiene
    CONSULTATION ||--o{ SURVEY_ANSWER : responde
    CONSULTATION ||--o| MEASUREMENT : mide
    CONSULTATION ||--o{ LAB_RESULT : analiza

    PATIENT {
        int id PK
        string doc_type
        string doc_number
        string full_name
        string sex
        string birthdate
    }
    CONSULTATION {
        int id PK
        int patient_id FK
        int consultation_number
        string consultation_type
        datetime consultation_date
        string phenotype_model
        string phenotype_final
        string classification_source
    }
    SURVEY_ANSWER {
        int id PK
        int consultation_id FK
        string question_key
        string answer_value
    }
    MEASUREMENT {
        int id PK
        int consultation_id FK
        float weight_kg
        float height_m
        float bmi
        float waist_cm
        int bp_systolic
        int bp_diastolic
    }
    LAB_RESULT {
        int id PK
        int consultation_id FK
        string test_key
        float value
        string unit
    }
```

### Modelo de clasificación
Hoy: clasificador **basado en reglas** transparente (`app/domain/classifier.py`),
detrás de la interfaz `Classifier`. El futuro modelo entrenado con los datos
históricos (pipeline de clustering en `databricks/`, ver su propio README) se
conecta guardándolo en `data/model.pkl`, sin tocar la UI ni la base de datos.

### Asistente conversacional ("Fénix")
El cuestionario de ingreso lo conduce **Fénix**, con dos implementaciones
intercambiables detrás de la misma interfaz `BaseIntake`:

- **`LLMIntake`** — usa Claude Opus 4.8 (vía Azure AI Foundry o la API directa
  de Anthropic, según `.env`). Presenta las opciones numeradas, valida el
  formato de cada respuesta y solo resuelve dudas sobre la pregunta pendiente.
- **`ScriptedIntake`** — flujo guiado determinista de respaldo (sin IA), para
  que el prototipo funcione aunque no haya credenciales configuradas.

## Puesta en marcha

```bash
# 1. (opcional) entorno virtual
py -m venv .venv && .venv\Scripts\activate

# 2. dependencias
py -m pip install -r requirements.txt

# 3. claves/credenciales (opcional; sin ellas se usan los flujos de respaldo)
copy .env.example .env
# - ANTHROPIC_API_KEY (+ ANTHROPIC_FOUNDRY_RESOURCE si usas Azure AI Foundry): chat con IA
# - FABRIC_SQL_USER: consulta de exámenes médicos en Microsoft Fabric desde la consola médica

# 4. datos de demostración (opcional)
py scripts/seed_demo.py

# 5. ejecutar
streamlit run app/main.py
```

Abrir http://localhost:8501

- **Pacientes demo**: `CC 1001` (Digestivo), `CC 1002` (Glicemia).
- **PIN consola médica**: `comfama`.
- **Exámenes médicos (Fabric)**: en la consola médica, tras buscar un paciente,
  el botón "🔬 Consultar exámenes en sistema central (Fabric)" trae el
  resultado más reciente de cada laboratorio desde
  `LH_FabricData.Hackaton2026.ResultadosAyudasDiagnosticas` y prellena los
  campos de laboratorio (revisar y guardar sigue siendo manual). Requiere
  `FABRIC_SQL_USER` en `.env`; la primera consulta pide login interactivo de
  Azure AD (la cuenta del workspace tiene MFA, por eso no usa contraseña).
