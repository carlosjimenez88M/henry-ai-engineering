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
comparación live de la clase 03.

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

En cada clase: lee la meta, **predice** la salida, ejecuta, explica la diferencia,
haz el taller antes de mirar la solución y completa el ticket de salida. Hay dos
pausas; la persona docente adapta el ritmo según lo que el grupo puede explicar.

El [proyecto](../../proyectos/tienda_workflows/README.md) tiene una plantilla y una
solución separada. Los helpers que reutilizamos están en
[`src/henry_agents/workflows.py`](../../src/henry_agents/workflows.py); los bloques
centrales también se construyen de forma visible en los notebooks.

## Qué es real y qué simulamos

| Componente | Implementación de esta ruta |
|---|---|
| Catálogo, validación, dinero y stock | Datos locales y herramientas de Python |
| Router y plan | Reglas deterministas, no comprensión general del lenguaje |
| Redactor y corrector | Plantillas/guiones visibles, no generación con IA |
| Paralelismo | Hilos reales, sin prometer mejoras de velocidad |
| Revisión automática | Campos estructurados, no verificación semántica exhaustiva |
| Aprobación | Variable didáctica; no autenticación ni checkpoint persistente |
| Modelo de lenguaje | Solo el experimento voluntario de clase 03 hace una llamada real |

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
```

La verificación ejecuta **8 scripts y 8 notebooks en kernels nuevos**. Además,
comprueba sincronización y registra huellas de los archivos usados. Los reportes
quedan en `reports/offline/workflows/`; no se versionan. Los notebooks de clase se
guardan limpios, sin resultados de una ejecución anterior.

Una prueba automática comprueba comportamiento, no comprensión del grupo.
Consulta la [guía docente](../../docs/GUIA_DOCENTE_WORKFLOWS.md) y el
[glosario](../../docs/GLOSARIO_WORKFLOWS.md).

## Después de estas clases

Sigue el recorrido ampliado existente en `clases/00_mundo_agentico.ipynb` a
`clases/05_deep_agents.ipynb`. Allí se implementan grafos, tool calling, agentes con
límites, checkpoints, interrupt/resume y Deep Agents. La clase 00 ampliada repasa
conceptos; puedes usarla como revisión antes de comenzar herramientas.

Los principiantes no necesitan completar ambos recorridos al mismo tiempo.
