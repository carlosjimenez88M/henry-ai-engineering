# Henry · AI Engineering desde cero

Empieza por **[Workflows de IA desde cero](clases/agentic_workflows/README.md)**:
diez clases en español con una tienda de barrio ficticia, Python básico, ejemplos
visibles, talleres, soluciones y dos proyectos. Esta ruta introduce los conceptos
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
   con el kernel `.venv` y sigue el [orden de las diez clases](clases/agentic_workflows/README.md).
   Si conoces Python, completa su taller y comienza en la clase 01.

El recorrido trabaja herramientas, modelado, cadena, routing, paralelismo,
evaluador–optimizador y orquestador–workers. Incluye una comparación voluntaria con
un LLM en la clase 03. La versión gratuita usa reglas y guiones explícitos: no son
modelos reales. El [proyecto en dos fases](proyectos/tienda_workflows/README.md)
tiene plantilla, solución separada, rúbrica y diez casos de evaluación.

Las clases [08 · RAG](clases/agentic_workflows/08_rag_desde_cero.ipynb) y
[09 · Agentic RAG](clases/agentic_workflows/09_agentic_rag.ipynb) añaden ingesta,
fragmentación, BM25, embeddings opcionales, cobertura, citas y fidelidad. El agente
puede reformular, reunir fuentes y consultar stock con límites explícitos.
El [proyecto RAG](proyectos/tienda_rag/README.md) compara ambas estrategias con
doce casos. En offline sus decisiones son reglas; en live decide un LLM.

[Guía docente inicial](docs/GUIA_DOCENTE_WORKFLOWS.md) ·
[Glosario](docs/GLOSARIO_WORKFLOWS.md) ·
[Estudio del origen](docs/ANALISIS_AGENTIC_WORKFLOWS.md)

[Guía RAG y Agentic RAG](docs/RAG_Y_AGENTIC_RAG.md) ·
[Modelos actuales por rol](docs/MODELOS.md)

## Recorrido ampliado: LangGraph, agentes y Deep Agents (ruta avanzada)

**Ocho clases** que continúan la ruta inicial: del bucle de agente escrito a mano hasta un
equipo de Deep Agents que planifica, delega y pide permiso. Todo funciona **sin clave de API**
(offline: herramientas, grafos y aprobaciones reales; el modelo se reemplaza por reglas
visibles) y el docente puede demostrarlo con **GPT-6** (live). Usa un archivo de fichas
ficticias (Batman, los Cuatro Fantásticos, El Chavo y canciones inventadas, sin letras).

**Requisito:** la ruta inicial (`clases/agentic_workflows/`) o saber Python básico.
Abre `clases/00_mundo_agentico.ipynb`, elige el kernel **`.venv`** y ejecuta con Shift + Enter.

| Clase | Pregunta y construcción | Notebook | Script |
|---|---|---|---|
| 0 · Mundo agéntico | ¿Qué es un agente y cuándo no usarlo? Escalera de autonomía, GPT-6 y costo | [Abrir](clases/00_mundo_agentico.ipynb) | [Python](clases/00_mundo_agentico.py) |
| 1 · Herramientas y bucle | Contrato de `buscar_archivo` y el bucle del agente escrito a mano | [Abrir](clases/01_herramientas_y_bucle.ipynb) | [Python](clases/01_herramientas_y_bucle.py) |
| 2 · RAG con evidencia | Salida estructurada, validación de citas, palabras vs significado (embeddings) | [Abrir](clases/02_rag_con_evidencia.ipynb) | [Python](clases/02_rag_con_evidencia.py) |
| 3 · Workflows en LangGraph | Estado, secuencia, routing, paralelo, `Send` + reducer | [Abrir](clases/03_workflows_langgraph.ipynb) | [Python](clases/03_workflows_langgraph.py) |
| 4 · Agentes confiables | `create_agent`, límites, streaming, memoria, fallas, inyección de prompts, costo | [Abrir](clases/04_agentes.ipynb) | [Python](clases/04_agentes.py) |
| 5 · Multiagente | Supervisor, agentes como herramientas y handoff | [Abrir](clases/05_multiagente.ipynb) | [Python](clases/05_multiagente.py) |
| 6 · Evaluación y humano | Evaluador–optimizador, interrupt/resume, casos, juez LLM calibrado | [Abrir](clases/06_evaluacion_y_humano.ipynb) | [Python](clases/06_evaluacion_y_humano.py) |
| 7 · Deep Agents | Plan, archivos virtuales, subagentes acotados, aprobación y streaming | [Abrir](clases/07_deep_agents.ipynb) | [Python](clases/07_deep_agents.py) |

Proyecto acumulativo: [Asistente del Archivo](proyectos/asistente_archivo/README.md).
Soluciones de los ejercicios: `soluciones/` (se ven con `ver_solucion(...)` desde el notebook).
Diseño y convenciones: [docs/DISENO_RUTA_AVANZADA.md](docs/DISENO_RUTA_AVANZADA.md) ·
[Guía docente](docs/GUIA_DOCENTE.md) · [Mapa de arquitecturas](docs/ARQUITECTURAS.md).

## Modos y modelos

**Offline (predeterminado):** herramientas, grafos, middleware, subagentes y aprobaciones
reales; las decisiones del modelo se reemplazan por **guiones** explícitos (`ModeloGuionado`).
No consume API. Las respuestas de los guiones se arman a partir de las observaciones reales
de las herramientas, no de textos inventados.

**Live:** completar `OPENAI_API_KEY` y `COURSE_MODE=live` en `.env`. Consume API y no hace
fallback silencioso. Modelos por defecto (revisar precios antes de cada cohorte):

| Variable | Modelo | Uso | USD por millón de tokens (entrada/salida) |
|---|---|---|---|
| `OPENAI_MODEL` | `gpt-6-luna` | Actividades breves y especialistas | 0.10 / 0.50 |
| `OPENAI_MODEL_AGENT` | `gpt-6.1-sol` | Coordinadores y Agentic RAG | 2 / 10 |
| `OPENAI_MODEL_RAG` | `gpt-6-luna` | Respuesta RAG y revisión de fidelidad | 0.10 / 0.50 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Experimento de embeddings | Ver documentación oficial |
| — | `gpt-6-astra` | Opcional para demostraciones exigentes | 10 / 50 |

Se usa la Responses API de OpenAI y `OPENAI_REASONING_EFFORT=low` (configurable).
Reiniciar el kernel si cambian las variables de entorno. No publicar el `.env`.
Verificados el 6 de octubre de 2026; [fuentes y compatibilidad](docs/MODELOS.md).

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
uv run ruff check src scripts tests clases proyectos soluciones
uv run python scripts/verify.py --mode offline   # 18 scripts y 18 notebooks, kernels nuevos
uv run python scripts/verify.py --mode offline --track workflows  # Las 10 clases iniciales
uv run python scripts/verify.py --mode offline --track advanced   # Solo las 8 de la ruta avanzada
uv run python scripts/evaluate_workflows.py      # 10 casos del proyecto inicial
uv run python scripts/evaluate_rag.py --mode offline  # 12 casos: RAG y Agentic RAG
uv run python scripts/verify.py --mode live --track advanced  # OpenAI; consume API
uv run python scripts/evaluate_rag.py --mode live --case RAG-06 # RAG con LLM
uv run python scripts/sync_notebooks.py          # .py → notebooks sin outputs
uv run python scripts/sync_notebooks.py --desde-notebooks  # traer ediciones del notebook al .py
```

Con `make` disponible (Mac/Linux): `make setup`, `make doctor`, `make test`, `make verify`, etc.

Los `.py` en formato Jupytext percent son la fuente editable. Los notebooks se guardan sin
outputs. Los ejecutados, logs y reportes quedan en `reports/`, fuera de Git. Cada
verificación registra fecha, versiones, huellas SHA-256 y estado de cada recorrido.
[Informe de validación](VALIDACION.md).
Los experimentos opcionales de las clases iniciales vienen apagados: la verificación
no activa sus banderas automáticamente.

## Organización

- `clases/agentic_workflows/`: diez pares notebook/script para empezar y su índice.
- `clases/0*.py` y `.ipynb`: ocho pares de la ruta avanzada; `soluciones/`: sus soluciones.
- `proyectos/tienda_workflows/`: consigna, plantilla y solución del integrador inicial.
- `proyectos/tienda_rag/`: comparación entre RAG y Agentic RAG, consigna y solución.
- `src/henry_agents/workflows.py`: herramientas y patrones de la tienda ficticia.
- `src/henry_agents/rag.py`: ingesta, recuperadores, citas, fidelidad y grafo Agentic RAG.
- `src/henry_agents/cultural.py`: herramienta, contratos, agente a mano, equipo y revisión.
- `src/henry_agents/agentic.py`: cerebros offline (`ModeloReglas`, con fallas a propósito),
  `create_agent` con límites, Deep Agent acotado, streaming, línea de tiempo y dibujo de grafos.
- `src/henry_agents/semantica.py` y `practica.py`: búsqueda por significado y autocorrección.
- `src/henry_agents/config.py`: modos, modelos GPT-6 y parámetros de OpenAI.
- `src/henry_agents/data/`: corpus ficticios, catálogo, manual RAG y casos de evaluación.
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
