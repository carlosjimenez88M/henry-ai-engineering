# Henry · Módulo 3 · AI Engineering aplicado con LangGraph

**Cuatro notebooks, cada uno una clase completa de 120 minutos.** Ocho horas en total.
Cada notebook contiene la explicación, el código por etapas, preguntas de predicción,
experimentos, dos pausas de cinco minutos, un taller, soluciones y un ticket de salida.
El script Python equivalente permite reproducir el mismo recorrido.

Construimos un archivo cultural con escenarios inventados de Batman, los Cuatro
Fantásticos, El Chavo del Ocho y canciones ficticias. No hace falta conocer esas
obras. No se incluyen letras, audio, páginas de cómics ni episodios originales.

## Las clases

| Clase | Lo que construimos y aprendemos | Notebook | Script |
|---|---|---|---|
| 1 · Una herramienta confiable | Contrato Pydantic, búsqueda, filtros, ranking, límites, IDs y tool calling | [Abrir](clases/01_fundamentos.ipynb) | [Python](clases/01_fundamentos.py) |
| 2 · RAG y primeros grafos | Evidencia, salida estructurada, validación de fuentes, estado, secuencia y routing | [Abrir](clases/02_langchain_rag.ipynb) | [Python](clases/02_langchain_rag.py) |
| 3 · Arquitecturas | Paralelismo, barrera de unión, Send, reducers, orquestador–workers y demo de agente | [Abrir](clases/03_multiagente_langgraph.ipynb) | [Python](clases/03_multiagente_langgraph.py) |
| 4 · Revisión y evaluación | Ciclo acotado, referencias inválidas, interrupt/resume, aprobación/rechazo y reporte | [Abrir](clases/04_produccion_llmops.ipynb) | [Python](clases/04_produccion_llmops.py) |

**Profundidad con acompañamiento:** no se pide escribir todo desde cero ni memorizar
APIs. Cada bloque exige una predicción, una modificación o una comprobación. El
material incluye lo necesario para explicar cada paso, sin requerir que el docente
improvise contenido externo para llenar las dos horas. Las agendas detalladas están
dentro de cada notebook y suman 120 minutos, incluidas pausas y cierre.

## Qué arquitecturas se trabajan

| Patrón | Nivel de trabajo |
|---|---|
| Secuencia y routing | Construcción guiada y comparación en clase 2 |
| Paralelismo fijo y orquestador–workers | Construcción guiada con pruebas en clase 3 |
| Agente con herramientas | Demo ejecutable con límite de llamadas en clase 3 |
| Evaluador–optimizador | Construcción y falla inyectada en clase 4 |
| Supervisor y handoff | Comparación conceptual; no se presenta como implementación completa |
| Revisión humana y checkpoints | Capacidad transversal construida en clase 4 |

Consultar el [mapa de arquitecturas](docs/ARQUITECTURAS.md). Un worker no es necesariamente
un agente y un grafo no implica que un LLM tome decisiones. El material distingue
reglas, planes explícitos y decisiones del modelo.

## La herramienta central

`buscar_archivo` permite buscar por tema, colección y tipo, con hasta cinco resultados.
Valida argumentos antes de leer, normaliza tildes, aplica alias explícitos, filtra,
ordena de manera estable y devuelve fragmentos acotados con IDs. Informa no_results
cuando no encuentra evidencia; no accede a Internet ni ejecuta código proporcionado
por el modelo. El score es lexical, no una probabilidad de verdad.

La potencia está en un contrato que se puede combinar con distintas arquitecturas.
No se promete un buscador universal o semántico: el corpus tiene doce fichas.

## Preparación previa (docente o sesión asistida)

Python 3.11–3.13 y uv. Desde esta carpeta:

```bash
uv sync --locked
make doctor
make lab
```

Seleccionar el kernel de `.venv`. Si el editor requiere registrarlo:

```bash
uv run python -m ipykernel install --user --name henry-m3 --display-name 'Henry M3'
```

La instalación se prepara antes de las sesiones; no se evalúa como conocimiento
conceptual. Si alguien tiene dificultades, puede trabajar con una pareja y entregar
predicciones/resultados por escrito mientras se resuelve su entorno.

**Offline (predeterminado):** herramientas y grafos reales, con guiones o extractos
en lugar del LLM. No consume API. **Live:** usa OPENAI_API_KEY del `.env` y el modelo
OPENAI_MODEL, por defecto gpt-4.1-mini. Consume API y no hace fallback silencioso.
El docente puede demostrar live; no es un requisito para practicar.

```bash
COURSE_MODE=live uv run python clases/01_fundamentos.py
```

El modo se muestra al comenzar. Reiniciar el kernel si cambian las variables de
entorno. No modificar el `.env` existente ni publicar sus claves. No hay servicio
externo de observabilidad: los reportes se guardan localmente.

## Verificar y mantener

```bash
make test           # Contratos, grafos, límites, rechazo y compatibilidad anterior
make lint
make verify         # 4 scripts + 4 notebooks offline, kernels nuevos
make verify-live    # Los mismos recorridos con OpenAI; consume API
make eval           # 10 casos del catálogo actual y reporte JSON local
uv run python scripts/evaluate.py --mode live  # Evaluación también con generación
make notebooks      # Sincroniza notebooks a partir de sus scripts .py
```

Los `.py` en formato Jupytext percent son la fuente editable. Los notebooks originales
se mantienen sin outputs. Los ejecutados, logs y reportes están en `reports/`, fuera
de Git. Cada verificación registra fecha, versiones, huellas SHA-256 y estado de cada recorrido.
Un fallo sustituye el reporte anterior y conserva los logs para diagnosticarlo.
Las verificaciones fallan si una celda o assertion falla; solo se capturan
los errores que el material marca explícitamente como demostraciones intencionales.
[Informe de validación](VALIDACION.md).

## Organización

- `clases/`: los cuatro pares notebook/script. Es el recorrido de clase.
- `src/henry_agents/cultural.py`: herramienta, contratos, agente, equipo y revisión reutilizables.
- `src/henry_agents/data/cultural_catalog.json`: doce escenarios y metadatos ficticios.
- `src/henry_agents/data/cultural_golden.json`: diez casos etiquetados de evaluación.
- `tests/test_cultural.py`: pruebas del caso actual. Los otros módulos/tests conservan
  ejemplos de soporte y ventas de revisiones anteriores como referencia; no son
  prerrequisitos ni clases adicionales.
- `docs/`: guía docente, mapa de arquitecturas y procedencia del material.
- `scripts/`: diagnóstico, sincronización, verificación y evaluación.

## Proyecto y alcance

Entregar una herramienta con contrato, una arquitectura justificada, fuentes y
abstención, un ciclo acotado y evidencia de aprobación/rechazo. La clase 4 produce
cinco casos; agregar dos más como trabajo posterior. La evaluación admite explicación
oral, escrita o diagramas y no premia rapidez.

Los checkpoints de la demo viven en memoria: no sobreviven reinicios. Las aprobaciones
son simuladas y no producen acciones externas. Validar IDs no demuestra fidelidad
semántica completa. Autenticación, persistencia durable y despliegue quedan fuera
de estas ocho horas.

[Guía docente](docs/GUIA_DOCENTE.md) · [Fuentes y cambios](docs/FUENTES_Y_CAMBIOS.md)
