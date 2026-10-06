# Henry · AI Engineering desde cero

Empieza por **[Workflows de IA desde cero](clases/agentic_workflows/README.md)**:
ocho clases en español con una tienda de barrio ficticia, Python básico, ejemplos
visibles, talleres, soluciones y un proyecto. Esta ruta introduce los conceptos
estudiados en `cd14525-agentic-workflows-classroom` con material original para un
público latino que comienza con IA. Funciona sin clave ni consumo de API.

## Ruta inicial: empieza aquí

1. Sigue [la instalación](docs/INSTALACION.md) y abre esta carpeta en VS Code.
2. En una terminal dentro de `henry-ai-engineering`:

   ```bash
   uv sync --locked
   uv run python scripts/doctor.py
   ```

3. Abre [00 · Python para empezar](clases/agentic_workflows/00_python_para_empezar.ipynb)
   con el kernel `.venv` y sigue el [orden de las ocho clases](clases/agentic_workflows/README.md).
   Si conoces Python, completa su taller y comienza en la clase 01.

El recorrido trabaja herramientas, modelado, cadena, routing, paralelismo,
evaluador–optimizador y orquestador–workers. Incluye una comparación voluntaria con
un LLM en la clase 03. La versión gratuita usa reglas y guiones explícitos: no son
modelos reales. El [proyecto en dos fases](proyectos/tienda_workflows/README.md)
tiene plantilla, solución separada, rúbrica y diez casos de evaluación.

[Guía docente inicial](docs/GUIA_DOCENTE_WORKFLOWS.md) ·
[Glosario](docs/GLOSARIO_WORKFLOWS.md) ·
[Estudio del origen](docs/ANALISIS_AGENTIC_WORKFLOWS.md)

## Recorrido ampliado: LangGraph y Deep Agents

**Seis notebooks, cada uno una clase completa**, para personas que empiezan en IA aplicada.
Cada notebook trae explicación, código por etapas, predicciones, experimentos, dos pausas,
un taller, soluciones y un ticket de salida. Todo funciona **sin clave de API** (modo
offline) y, opcionalmente, con los modelos **GPT-6** de OpenAI (modo live).

Construimos un archivo cultural con escenarios inventados de Batman, los Cuatro
Fantásticos, El Chavo del Ocho y canciones ficticias. No hace falta conocer esas obras.
No se incluyen letras, audio, páginas de cómics ni episodios originales.

## Empezar el recorrido ampliado

1. Seguí **[docs/INSTALACION.md](docs/INSTALACION.md)**: VS Code + extensiones Python y
   Jupyter + `uv`, paso a paso, con solución de problemas frecuentes.
2. En la terminal de VS Code, dentro de esta carpeta:

   ```bash
   uv sync --locked
   uv run python scripts/doctor.py
   ```

3. Abrí `clases/00_mundo_agentico.ipynb`, elegí el kernel **`.venv`** y ejecutá con Shift + Enter.

## Las clases

| Clase | Lo que construimos y aprendemos | Notebook | Script |
|---|---|---|---|
| 0 · Mundo agéntico | LLM, token, prompt, herramienta, agente, workflow; escalera de autonomía; modelos GPT-6 y costos; primer agente | [Abrir](clases/00_mundo_agentico.ipynb) | [Python](clases/00_mundo_agentico.py) |
| 1 · Una herramienta confiable | Contrato Pydantic, búsqueda, filtros, ranking, límites, IDs y tool calling | [Abrir](clases/01_fundamentos.ipynb) | [Python](clases/01_fundamentos.py) |
| 2 · RAG y primeros grafos | Evidencia, salida estructurada, validación de fuentes, estado, secuencia y routing | [Abrir](clases/02_langchain_rag.ipynb) | [Python](clases/02_langchain_rag.py) |
| 3 · Orquestación | Paralelismo, Send, reducers, agente a mano y con `create_agent` + middleware, supervisor con `Command` | [Abrir](clases/03_multiagente_langgraph.ipynb) | [Python](clases/03_multiagente_langgraph.py) |
| 4 · Revisión y evaluación | Ciclo acotado, referencias inválidas, interrupt/resume, aprobación/rechazo y reporte | [Abrir](clases/04_produccion_llmops.ipynb) | [Python](clases/04_produccion_llmops.py) |
| 5 · Deep Agents | Plan (`write_todos`), archivos virtuales, subagentes en paralelo (`task`), aprobación humana, verificación de citas | [Abrir](clases/05_deep_agents.ipynb) | [Python](clases/05_deep_agents.py) |

**Profundidad con acompañamiento:** no se pide escribir todo desde cero ni memorizar
APIs. Cada bloque exige una predicción, una modificación o una comprobación.

## Qué arquitecturas se trabajan

| Patrón | Nivel de trabajo |
|---|---|
| Secuencia y routing | Construcción guiada y comparación en clase 2 |
| Paralelismo fijo y orquestador–workers | Construcción guiada con pruebas en clase 3 |
| Agente con herramientas | Hecho a mano y con `create_agent` + límites como middleware, clase 3 |
| Supervisor | Construido con `Command` y límite de delegaciones en clase 3; en live decide GPT-6 |
| Evaluador–optimizador | Construcción y falla inyectada en clase 4 |
| Revisión humana y checkpoints | Construida a mano en clase 4; integrada con `interrupt_on` en clase 5 |
| Deep agent (coordinador + subagentes) | Construcción guiada en clase 5 |
| Handoff | Comparación conceptual; no se presenta como implementación completa |

Consultar el [mapa de arquitecturas](docs/ARQUITECTURAS.md). Un worker no es necesariamente
un agente y un grafo no implica que un LLM tome decisiones. El material distingue reglas,
planes explícitos y decisiones del modelo.

## Modos y modelos

**Offline (predeterminado):** herramientas, grafos, middleware, subagentes y aprobaciones
reales; las decisiones del modelo se reemplazan por **guiones** explícitos (`ModeloGuionado`).
No consume API. Las respuestas de los guiones se arman a partir de las observaciones reales
de las herramientas, no de textos inventados.

**Live:** completar `OPENAI_API_KEY` y `COURSE_MODE=live` en `.env`. Consume API y no hace
fallback silencioso. Modelos por defecto (revisar precios antes de cada cohorte):

| Variable | Modelo | Uso | USD por millón de tokens (entrada/salida) |
|---|---|---|---|
| `OPENAI_MODEL` | `gpt-6-luna` | Clases 0–4, especialistas de la clase 5 | 0.10 / 0.50 |
| `OPENAI_MODEL_AGENT` | `gpt-6.1-sol` | Coordinador del deep agent | 2 / 10 |
| — | `gpt-6-astra` | Opcional para demostraciones exigentes | 10 / 50 |

Se usa la Responses API de OpenAI y `OPENAI_REASONING_EFFORT=low` (configurable).
Reiniciar el kernel si cambian las variables de entorno. No publicar el `.env`.

## La herramienta central

`buscar_archivo` permite buscar por tema, colección y tipo, con hasta cinco resultados.
Valida argumentos antes de leer, normaliza tildes, aplica alias explícitos, filtra,
ordena de manera estable y devuelve fragmentos acotados con IDs. Informa no_results
cuando no encuentra evidencia; no accede a Internet ni ejecuta código proporcionado
por el modelo. El score es lexical, no una probabilidad de verdad. La misma herramienta
la usan el agente de la clase 0, los grafos de las clases 2–4 y los subagentes de la clase 5.

## Verificar y mantener

```bash
uv run python scripts/doctor.py           # Diagnóstico del entorno (también revisa VS Code)
uv run python scripts/doctor.py --live    # + una llamada mínima a OpenAI
uv run pytest -q                          # Contratos, grafos, agentes, límites y aprobaciones
uv run ruff check src scripts tests clases proyectos
uv run python scripts/verify.py --mode offline   # 14 scripts + 14 notebooks en kernels nuevos
uv run python scripts/verify.py --mode offline --track workflows  # Solo las 8 clases iniciales
uv run python scripts/verify.py --mode offline --track advanced   # Solo las 6 ampliadas
uv run python scripts/evaluate_workflows.py      # 10 casos del proyecto inicial
uv run python scripts/verify.py --mode live      # Lo mismo con OpenAI; consume API
uv run python scripts/sync_notebooks.py          # .py → notebooks sin outputs
uv run python scripts/sync_notebooks.py --desde-notebooks  # traer ediciones del notebook al .py
```

Con `make` disponible (Mac/Linux): `make setup`, `make doctor`, `make test`, `make verify`, etc.

Los `.py` en formato Jupytext percent son la fuente editable. Los notebooks se guardan sin
outputs. Los ejecutados, logs y reportes quedan en `reports/`, fuera de Git. Cada
verificación registra fecha, versiones, huellas SHA-256 y estado de cada recorrido.
[Informe de validación](VALIDACION.md).

## Organización

- `clases/agentic_workflows/`: ocho pares notebook/script para empezar y su índice.
- `clases/0*.py` y `.ipynb`: seis pares del recorrido ampliado.
- `proyectos/tienda_workflows/`: consigna, plantilla y solución del integrador inicial.
- `src/henry_agents/workflows.py`: herramientas y patrones de la tienda ficticia.
- `src/henry_agents/cultural.py`: herramienta, contratos, agente a mano, equipo y revisión.
- `src/henry_agents/agentic.py`: `create_agent` con límites, Deep Agent, `ModeloGuionado`,
  línea de tiempo de mensajes y dibujo de grafos con o sin Internet.
- `src/henry_agents/config.py`: modos, modelos GPT-6 y parámetros de OpenAI.
- `src/henry_agents/data/`: doce escenarios ficticios y diez casos de evaluación.
- `tests/`: pruebas del caso cultural, de los agentes modernos y de la verificación.
  Otros módulos/tests conservan ejemplos de soporte y ventas de revisiones anteriores.
- `docs/`: instalación, guías docentes por recorrido, glosario, estudio del origen,
  mapa de arquitecturas y fuentes.
- `.vscode/`: configuración compartida (intérprete `.venv`, extensiones recomendadas).

## Proyecto y alcance

Entregar una herramienta con contrato, una arquitectura justificada, fuentes y abstención,
un ciclo acotado, evidencia de aprobación/rechazo y, como integrador, un equipo de deep
agents con verificación de citas. La evaluación admite explicación oral, escrita o
diagramas y no premia rapidez.

Los checkpoints viven en memoria y los archivos del deep agent son virtuales (estado del
grafo): no tocan el disco ni sobreviven reinicios. Las aprobaciones son simuladas y no
producen acciones externas. Validar IDs no demuestra fidelidad semántica completa.
Autenticación, persistencia durable y despliegue quedan fuera del alcance.

[Instalación](docs/INSTALACION.md) · [Guía docente](docs/GUIA_DOCENTE.md) ·
[Arquitecturas](docs/ARQUITECTURAS.md) · [Fuentes y cambios](docs/FUENTES_Y_CAMBIOS.md)
