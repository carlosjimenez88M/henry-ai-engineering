# Validación · Workflows, RAG y agentes

Última comprobación: **6 de octubre de 2026**, macOS ARM64, Python 3.13.15,
uv 0.12.17, langchain 1.4.3, langgraph 1.2.14, langchain-core 1.6.7,
langchain-openai 1.6.7, deepagents 0.7.22, nbclient 0.11.0.
Las dependencias se instalaron desde `uv.lock`; no se agregó una dependencia para RAG.

## Resultados ejecutados

| Comprobación | Resultado |
|---|---|
| pytest, repo y copia limpia | **234 pruebas aprobadas** en cada entorno |
| Ruff | Sin errores en `src`, `scripts`, `tests`, `clases`, `proyectos` y `soluciones` |
| Scripts offline | **18/18 aprobados**: diez iniciales y ocho avanzados |
| Notebooks offline | **18/18 aprobados**, cada uno en un kernel nuevo |
| Sincronización | Los 18 notebooks coinciden con sus scripts; se guardan sin outputs |
| Proyecto de workflows | Referencia y solución: **10/10** casos cada una |
| Comparación RAG / Agentic RAG | **12/12** comportamientos esperados en cada alternativa, citas válidas y completas |
| Instalación limpia | `uv sync --locked --offline` en copia sin `.venv` ni `.env`: 159 paquetes instalados desde caché local |
| Ejecución en copia limpia | 234 pruebas, 18 scripts, 18 notebooks, proyectos y doctor aprobados |
| Diagnóstico docente | `doctor.py`: todo listo; cuatro roles de modelos reconocidos |
| Enlaces locales de clases y guías | Sin enlaces rotos |

La instalación limpia prueba construcción y resolución con paquetes ya descargados.
No simula la descarga inicial en una red de aula. Las instrucciones explican que
la primera instalación necesita Internet y recomiendan preparar el entorno antes de clase.

## Evidencia y repetición

Los reportes locales, excluidos de Git, conservan fecha, versiones, SHA-256 y estado
por fase. Se comprobó que las huellas del repo y de la copia limpia coinciden con
el código final. Los notebooks ejecutados y logs permanecen en `reports/offline/`.

- [Verificación de las 18 clases](reports/offline/verification.json).
- [Verificación de la copia limpia](reports/reproducibility/verification-clean.json).
- [Comparación RAG](reports/rag-evaluation-offline.json).
- [Comparación RAG en copia limpia](reports/reproducibility/rag-evaluation-clean.json).
- [Proyecto workflows](reports/workflows-evaluation.json) y
  [solución separada](reports/solution-evaluation.json).

Desde la raíz:

```bash
uv sync --locked
uv run python scripts/doctor.py
uv run pytest -q
uv run ruff check src scripts tests clases proyectos soluciones
uv run python scripts/verify.py --mode offline
uv run python scripts/evaluate_workflows.py
uv run python scripts/evaluate_rag.py --mode offline
```

## RAG: resultados y límites de interpretación

Ambas alternativas usan el mismo corpus vigente, BM25 y `k=1`. El RAG clásico
responde 7/12 preguntas con 12 búsquedas; el ciclo Agentic RAG responde 11/12 con
15 consultas a fuentes. Recall promedio por documento en los once casos con
fuente esperada: aproximadamente 0.727 y 1.0 respectivamente. Ambas alternativas
cumplen sus doce estados esperados, incluida la abstención. No se hizo ninguna
llamada a un LLM o a embeddings reales en esta comparación.

Son casos ficticios y decisiones offline por reglas. El resultado no evalúa la
calidad de los modelos ni demuestra superioridad general de Agentic RAG. El
recuperador lexical no comprende sinónimos generales. Las necesidades y cobertura
usan criterios explícitos; las citas se comprueban contra el contexto, y la
fidelidad offline exige extractos exactos. Un juez LLM live también puede fallar.
Ver [guía RAG y Agentic RAG](docs/RAG_Y_AGENTIC_RAG.md).

Las pruebas incluyen documento archivado, consulta sin respuesta, fragmentación,
metadatos, citas inventadas, afirmación contradictoria con ID real, pregunta con
dos necesidades, reformulación tras búsqueda vacía, stock obtenido del catálogo,
repetición bloqueada, presupuestos agotados y fallo de proveedor sin fallback.
Los dobles de modelos comprueban prompts, contratos y transición de acciones;
`HashEmbeddings` comprueba integración y no calidad semántica aprendida.

## Modelos y verificaciones pendientes

La configuración central usa GPT-6 Luna para actividades y respuesta/revisión RAG,
GPT-6.1 Sol para coordinación y `text-embedding-3-large` para el experimento semántico.
GPT-6 Astra queda como alternativa para comparar tareas exigentes. Nombres y guía
oficial se verificaron el 6 de octubre de 2026; ver [MODELOS.md](docs/MODELOS.md).

No había `OPENAI_API_KEY` disponible, por lo que **no se ejecutó el modo live ni
embeddings reales**. Las pruebas inspeccionan solicitudes del SDK con Responses API,
`reasoning.effort=low`, máximo de salida de 8000 tokens y roles configurables.
También comprueban generación, revisión de fidelidad y decisiones con dobles sin red.
La disponibilidad de los modelos en cada cuenta requiere una llamada propia.

Antes de una demostración con API:

```bash
uv run python scripts/doctor.py --live
uv run python scripts/verify.py --mode live --track advanced
uv run python scripts/evaluate_rag.py --mode live --case RAG-06
```

Las banderas opcionales de los notebooks iniciales vienen apagadas; la verificación
live no las activa automáticamente. Las decisiones del LLM pueden variar. Se
cuentan invocaciones lógicas y consultas, pero los reintentos del SDK pueden sumar
solicitudes HTTP: el presupuesto del grafo no es un límite absoluto de facturación.

Windows y CI remota no se ejecutaron en esta revisión. CI está configurada para
pruebas, notebooks, evaluación de workflows y comparación RAG. Checkpoints,
vector stores y archivos del deep agent viven en memoria; las prácticas no producen
compras, cobros ni mensajes externos. La ejecución comprueba funcionamiento, no
comprensión del alumnado: la primera cohorte debe aportar evidencia pedagógica.
