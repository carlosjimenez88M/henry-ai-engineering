# Ruta avanzada · Agentes con LangGraph, Deep Agents y GPT-6

Diseño, convenciones y API de apoyo. Para docentes y para quien edite las clases.

## Lugar en el curso

| Ruta | Carpeta | Para quién | Qué enseña |
|---|---|---|---|
| 1 · La Esquina | `clases/agentic_workflows/` | Nunca programaste | Python mínimo y patrones de workflow sin frameworks |
| 2 · Avanzada (esta) | `clases/0*.py` | Terminaste la ruta 1 | Herramientas, RAG, LangGraph, agentes, multiagente, evaluación y Deep Agents |

La ruta avanzada **supone la ruta 1**: listas, diccionarios, funciones, `for`/`if`, f-strings y
`try/except` ya se vieron. La sintaxis nueva se presenta en recuadros **🐍 Python nuevo**.
Los patrones (cadena, routing, paralelo, evaluador–optimizador, orquestador) ya se conocen:
aquí se construyen **con LangGraph** y se suman agentes que deciden.

## Programa

| # | Archivo | Pregunta de la clase | Construcción principal | 🧱 Paso del proyecto |
|---|---|---|---|---|
| 0 | `00_mundo_agentico` | ¿Qué es un agente y cuándo NO usarlo? | Escalera de autonomía, modelos GPT-6 y costo, primer agente observado | Leer el encargo y elegir el escalón |
| 1 | `01_herramientas_y_bucle` | ¿Cómo usa un modelo una herramienta? | Contrato Pydantic de `buscar_archivo` + **bucle de agente escrito a mano** | La herramienta del asistente |
| 2 | `02_rag_con_evidencia` | ¿Cómo responder solo con evidencia? | Prompt, salida estructurada visible, validación de citas, léxico vs semántico | Respuestas con fuentes |
| 3 | `03_workflows_langgraph` | ¿Cómo orquesto pasos conocidos? | Estado, secuencia, routing, paralelo, `Send` + reducer | El flujo fijo del asistente |
| 4 | `04_agentes` | ¿Cómo hago un agente confiable? | `create_agent`, límites, streaming, memoria, salida estructurada, fallas e inyección | El asistente se vuelve agente |
| 5 | `05_multiagente` | ¿Cuándo dividir en varios agentes? | Supervisor con `Command`, especialistas como herramientas, handoff mínimo | Un equipo de especialistas |
| 6 | `06_evaluacion_y_humano` | ¿Cómo sé que funciona y quién aprueba? | Evaluador–optimizador, `interrupt`/`resume`, golden set, juez LLM, costo | Evaluación y aprobación |
| 7 | `07_deep_agents` | ¿Qué cambia en tareas largas? | `create_deep_agent`: plan, archivos, subagentes acotados, aprobación, streaming | Entrega final: el equipo profundo |

**Proyecto acumulativo:** el *Asistente del Archivo* del Centro Cultural. Ver
`proyectos/asistente_archivo/README.md`. Cada clase termina con su paso del proyecto.

## Modos

- **offline (por defecto, estudiantes):** herramientas, grafos, middleware, subagentes y
  aprobaciones reales; el cerebro es `ModeloReglas` (reglas visibles que leen el pedido).
  Se nota en la línea de tiempo: `🤖 reglas-offline`.
- **live (docente):** GPT-6 (`gpt-6-luna`; coordinadores `gpt-6.1-sol`). Las comprobaciones de
  valores exactos van dentro de `if MODE == "offline":`; en live se comprueban invariantes
  (por ejemplo, "toda cita existe en el catálogo").

## Convenciones de escritura (obligatorias)

**Idioma.** Español neutro latinoamericano con **tú** ("ejecuta", "observa", "puedes"), igual
que la ruta 1. Términos en inglés en *cursiva* la primera vez, con su explicación.

**Tono.** Directo y cálido. Sin texto defensivo ("no prometemos…", "no afirmamos…"). Los
límites de lo construido van **una sola vez**, en la sección final "Límites de lo que hicimos".

**Celdas markdown cortas.** Máximo ~8 líneas por celda; una idea por celda. Nada de muros de
texto. Tablas solo si comparan; diagramas en bloques ```text``` pequeños.

**Jerga.** Todo término técnico se define al aparecer por primera vez, en negrita y una frase:
"**reducer** (la regla que dice cómo juntar dos escrituras en el mismo campo)".

**Ritmo por bloque:** 🔮 **Predice** (una pregunta concreta) → ▶️ código → 🔍 **Observa**
(qué mirar en la salida, 1–3 viñetas). Nada de "Dibujá una tabla" sin consigna clara.

**Ejercicios ✏️ Tu turno.**
1. Una celda con la consigna exacta: qué cambiar, dónde y cómo saber si salió bien.
2. Una celda de código con espacios para completar usando `None` o `...` y el comentario
   `# ✏️ completa aquí`. Debe poder ejecutarse sin error aunque esté incompleta.
3. Una celda que revisa con `comprobar(condición, "mensaje si está bien", "pista")`.
4. Una celda con `ver_solucion("NN_nombre")`. **Nunca** la solución escrita en el notebook.
   La solución vive en `soluciones/NN_nombre.py`, es autocontenida (importa todo lo que usa),
   imprime su resultado y termina con `confirmar(...)`. Se ejecuta en la verificación.

**Comprobaciones de demostraciones:** `confirmar(condición, "qué se esperaba")`, no `assert`.

**Sin `input()`**, sin rutas absolutas, sin dependencias de Internet obligatorias.

**Encabezado de cada clase** (y nada más repetido):
```text
# Clase N · Título
Una frase con la pregunta de la clase.
**Vas a construir:** 3 viñetas.
**Necesitas:** clase anterior (y ruta 1). Modo offline por defecto; live si el docente lo activa.
**Recorrido:** 6–9 viñetas, con dos "☕ Pausa".
```
**Cierre de cada clase:** 🧱 Proyecto (paso del día) · 🎟️ Ticket de salida (3 preguntas) ·
📖 Glosario de hoy (términos nuevos) · Límites de lo que hicimos (2–4 viñetas).

**Tamaño objetivo:** una sesión de 2 h para alguien que viene de la ruta 1: ~1.200–1.600
palabras de markdown y ~150–220 líneas de código por clase, dos pausas.

## API de apoyo (`src/henry_agents`)

```python
from henry_agents.config import configure, chat_model, model_name, MODELOS, costo_usd, medir_costo
MODE = configure()                       # "offline" | "live"
chat_model()  / chat_model("agent")      # GPT-6 Luna / Sol (solo live)

from henry_agents.cultural import search_catalog, buscar_archivo, SearchArgs, SearchResult, \
    GroundedAnswer, compose, load_catalog, query_terms
from henry_agents.agentic import (
    ModeloReglas, ModeloGuionado, ModeloCoordinador, cerebro, llamar, interpretar_pedido,
    redactar_con_fuentes, crear_agente, LimiteDeLlamadas, linea_de_tiempo, ver_en_vivo,
    mostrar_grafo, leer_resenas, publicar_anuncio, ANUNCIOS_PUBLICADOS, HumanInTheLoopMiddleware,
    crear_equipo_profundo, SUBAGENTES, PROMPT_COORDINADOR, PROMPT_AGENTE,
    solicitudes_pendientes, ejecutar_con_revision, texto, en_espanol,
)
from henry_agents.semantica import MAPA, EJES, vector_de, similitud, buscar_por_significado
from henry_agents.practica import comprobar, confirmar, ver_solucion
```

- `cerebro(MODE, falla=None)`: GPT-6 en live; `ModeloReglas(falla=...)` en offline.
  Fallas: `no_usa_herramienta`, `inventa_id`, `argumentos_invalidos`, `bucle`, `obedece_inyeccion`.
- `crear_agente(MODE, model=None, tools=None, system_prompt=..., max_llamadas_modelo=4,
  max_llamadas_herramientas=3, checkpointer=None, response_format=None, middleware=())`.
- `ModeloReglas` entiende pedidos como "investigación de Batman", "cooperación de El Chavo",
  "una canción para una actividad de equipo"; recuerda "me llamo X" si hay memoria; completa
  esquemas Pydantic (campos `respuesta`/`texto`, `fuentes`/`source_ids`, booleanos).
- `crear_equipo_profundo(MODE, models=None, subagentes=None, aprobar=("write_file","edit_file"),
  max_llamadas_coordinador=20, max_llamadas_especialista=6)`; pausa antes de escribir.
- `ver_en_vivo(agente, entrada, config)`: imprime cada paso al ocurrir (subagentes con ↳) y
  devuelve el estado (con `__interrupt__` si quedó en pausa).

## Catálogo (para predicciones; verificado con search_catalog, top_k=5)

| Tema | Batman | Fantásticos | Chavo | Canción |
|---|---|---|---|---|
| investigación | BAT-01, BAT-03 | FAN-01 (el texto la menciona) | — | MUS-02 |
| herramientas | BAT-02 | FAN-02 | CHA-03 | — |
| equipo / cooperación | — | FAN-01, FAN-03 | CHA-01 | MUS-01 |
| evidencia | BAT-03 | FAN-02, FAN-03 | CHA-02 | — |
| ciencia | — | FAN-01, FAN-02 | — | MUS-03 |
