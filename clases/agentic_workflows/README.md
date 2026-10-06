# Workflows de IA desde cero · La Esquina

Esta es la ruta de entrada para personas de Latinoamérica que recién empiezan
con IA. Los textos están en español, los términos en inglés se explican al aparecer
y los ejercicios comparten una tienda ficticia. No necesitas conocer obras de
ficción, procesos empresariales ni sistemas de agentes para comenzar.

Estudiamos los conceptos de `udacity/cd14525-agentic-workflows-classroom` y
desarrollamos actividades, datos y código nuevos. La [revisión del origen](../../docs/ANALISIS_AGENTIC_WORKFLOWS.md)
explica qué temas cubrimos y qué problemas de reproducción resolvimos.

## Antes de empezar

Sigue [la instalación](../../docs/INSTALACION.md). Necesitas una computadora,
Internet para descargar el entorno la primera vez y saber abrir una carpeta.
La clase 00 enseña el Python mínimo. Después de instalar, la ruta inicial funciona
sin red, sin clave y sin consumo de API, salvo que actives voluntariamente la
comparación live de la clase 03 o los experimentos de RAG de las clases 08/09.

Desde la raíz `henry-ai-engineering`:

```bash
uv sync --locked
uv run python scripts/doctor.py
uv run jupyter lab clases/agentic_workflows
```

También puedes abrir los notebooks en VS Code, elegir el kernel `.venv` y ejecutar
con Shift + Enter. Los scripts contienen las mismas explicaciones y celdas; puedes
ejecutarlos con `uv run python clases/agentic_workflows/00_python_para_empezar.py`.

## Orden de las clases

| Clase | Aprendes y construyes | Notebook | Fuente editable |
|---|---|---|---|
| 00 · Python para empezar | Listas, diccionarios, condiciones, funciones y errores | [Notebook](00_python_para_empezar.ipynb) | [Python](00_python_para_empezar.py) |
| 01 · IA y herramientas | LLM, prompt, token, contrato, cotización y abstención | [Notebook](01_ia_y_herramientas.ipynb) | [Python](01_ia_y_herramientas.py) |
| 02 · Modelado y cadena | Diagrama, entradas/salidas, dependencias y parada | [Notebook](02_modelado_y_cadena.ipynb) | [Python](02_modelado_y_cadena.py) |
| 03 · Routing | Áreas permitidas, tildes, ambigüedad y comparación opcional con LLM | [Notebook](03_routing.ipynb) | [Python](03_routing.py) |
| 04 · Paralelismo | Consultas independientes, futuros y propagación de fallas | [Notebook](04_paralelismo.ipynb) | [Python](04_paralelismo.py) |
| 05 · Evaluador y optimizador | Criterios, feedback, corrección y límite de intentos | [Notebook](05_evaluador_optimizador.ipynb) | [Python](05_evaluador_optimizador.py) |
| 06 · Orquestador y workers | Plan variable, registro de workers y planes inválidos | [Notebook](06_orquestador_workers.ipynb) | [Python](06_orquestador_workers.py) |
| 07 · Proyecto integrador | Propuesta, revisión, aprobación/rechazo y diez casos | [Notebook](07_proyecto_integrador.ipynb) | [Python](07_proyecto_integrador.py) |
| 08 · RAG desde cero | Ingesta, fragmentos, BM25, embeddings opcionales, citas, fidelidad y recall | [Notebook](08_rag_desde_cero.ipynb) | [Python](08_rag_desde_cero.py) |
| 09 · Agentic RAG | Estado, decisiones, reformulación, catálogo, límites y comparación controlada | [Notebook](09_agentic_rag.ipynb) | [Python](09_agentic_rag.py) |

En cada clase: lee la meta, **predice** la salida, ejecuta, explica la diferencia,
haz el taller antes de mirar la solución y completa el ticket de salida. Hay dos
pausas; la persona docente adapta el ritmo según lo que el grupo puede explicar.

El [proyecto](../../proyectos/tienda_workflows/README.md) tiene una plantilla y una
solución separada. Los helpers que reutilizamos están en
[`src/henry_agents/workflows.py`](../../src/henry_agents/workflows.py); los bloques
centrales también se construyen de forma visible en los notebooks.

Las dos últimas clases profundizan en [RAG y Agentic RAG](../../docs/RAG_Y_AGENTIC_RAG.md).
El [segundo proyecto](../../proyectos/tienda_rag/README.md) compara una búsqueda
con recuperación iterativa, usando doce casos y dos preguntas propias.

## Qué es real y qué simulamos

| Componente | Implementación de esta ruta |
|---|---|
| Catálogo, validación, dinero y stock | Datos locales y herramientas de Python |
| Router y plan | Reglas deterministas, no comprensión general del lenguaje |
| Redactor y corrector | Plantillas/guiones visibles, no generación con IA |
| Paralelismo | Hilos reales, sin prometer mejoras de velocidad |
| Revisión automática | Campos estructurados, no verificación semántica exhaustiva |
| Aprobación | Variable didáctica; no autenticación ni checkpoint persistente |
| RAG y recuperación | Fragmentos y BM25 reales; respuesta offline por extractos |
| Decisiones de Agentic RAG | Reglas explícitas offline; LLM coordinador live |
| Modelo de lenguaje | Experimentos voluntarios de clases 03, 08 y 09 |
| Embeddings | OpenAIEmbeddings real solo en el experimento activado de clase 08 |

No se envían mensajes ni se cobran pedidos. No hay integración con WhatsApp.
El stock no se reserva y los precios no cambian durante el ejercicio. Los identificadores
de fuentes ayudan a rastrear evidencia; no garantizan que un texto libre la interprete bien.

## Verificar la ruta

```bash
uv run pytest -q
uv run ruff check src scripts tests clases proyectos
uv run python scripts/verify.py --mode offline --track workflows
uv run python scripts/evaluate_workflows.py
uv run python scripts/evaluate_workflows.py --implementation proyectos/tienda_workflows/solution.py
uv run python scripts/evaluate_rag.py --mode offline
```

La verificación ejecuta **10 scripts y 10 notebooks en kernels nuevos**. Además,
comprueba sincronización y registra huellas de los archivos usados. Los reportes
quedan en `reports/offline/workflows/`; no se versionan. Los notebooks de clase se
guardan limpios, sin resultados de una ejecución anterior.

Una prueba automática comprueba comportamiento, no comprensión del grupo.
Consulta la [guía docente](../../docs/GUIA_DOCENTE_WORKFLOWS.md) y el
[glosario](../../docs/GLOSARIO_WORKFLOWS.md).

Los [modelos actuales](../../docs/MODELOS.md) están centralizados por rol. Los
experimentos live vienen apagados; cambiar el modo de la verificación no cambia
esas banderas. Para probar una comparación real, configura `.env` y ejecuta
`uv run python scripts/evaluate_rag.py --mode live --case RAG-06`.

## Después de estas clases

Sigue el recorrido ampliado existente en `clases/00_mundo_agentico.ipynb` a
`clases/07_deep_agents.ipynb`. Allí se implementan grafos, tool calling, agentes con
límites, checkpoints, interrupt/resume y Deep Agents. La clase 00 ampliada repasa
conceptos; puedes usarla como revisión antes de comenzar herramientas.

Los principiantes no necesitan completar ambos recorridos al mismo tiempo.
