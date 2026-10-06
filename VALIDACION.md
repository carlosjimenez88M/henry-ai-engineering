# Validación · Mundo agéntico, LangGraph y Deep Agents

Última comprobación: **6 de octubre de 2026**, macOS ARM64, Python 3.13.15,
langgraph 1.2.14, langchain 1.4.3, langchain-core 1.6.7, langchain-openai 1.6.7,
deepagents 0.7.22.

## Resultados ejecutados

| Comprobación | Resultado |
|---|---|
| pytest | 100 pruebas aprobadas (incluye 16 nuevas de agentes modernos y configuración GPT-6) |
| Ruff | Sin errores en src, scripts, tests y clases |
| Scripts offline | 6 de 6 aprobados |
| Notebooks offline, kernels nuevos | 6 de 6 aprobados |
| Notebook con imágenes de grafos (con Internet, cwd = `clases/` como en VS Code) | Aprobado; PNG y respaldo en texto |
| Golden dataset offline | 10 de 10; métricas 1.0 |
| Instalación limpia (copia sin `.venv` ni `.env`) | `uv sync --locked`, 100 tests, 12 recorridos y doctor aprobados |
| Ida y vuelta notebook → script → notebook | Scripts idénticos |
| Diagnóstico `doctor.py` en el equipo docente | Todo listo (tras instalar la extensión Jupyter) |

## No verificado en esta revisión

- **Modo live con OpenAI:** no había `OPENAI_API_KEY` disponible, por lo que no se hicieron
  llamadas reales a GPT-6. Se comprobó con pruebas que la configuración genera la solicitud
  esperada (Responses API, `reasoning: {effort: low}`, `max_output_tokens: 4000`, modelos
  `gpt-6-luna` / `gpt-6.1-sol`) y que el deep agent asigna Sol al coordinador y Luna a los
  especialistas. Antes de la primera cohorte ejecutar:

  ```bash
  uv run python scripts/doctor.py --live
  uv run python scripts/verify.py --mode live
  ```

  En live las respuestas varían: las aserciones de las clases 3 y 5 que dependen de
  decisiones exactas solo se exigen en offline.
- Windows: la guía cubre PowerShell y rutas `.venv\Scripts`, pero no se ejecutó en Windows.
- CI remota: configurada (uv 0.12.17), no ejecutada.

## Causa de las fallas reportadas y corrección

| Falla | Corrección |
|---|---|
| VS Code sin extensión **Jupyter**: los `.ipynb` no se pueden ejecutar | `.vscode/extensions.json` la recomienda; `doctor.py` la detecta; guía paso a paso |
| VS Code podía elegir un Python ajeno (el de Homebrew está dañado en este equipo) | `.vscode/settings.json` apunta a `.venv`; la primera celda de la clase 0 lo comprueba |
| Notebooks editados directamente que ya no coincidían con sus `.py`: la verificación se detenía antes de ejecutar | Ediciones incorporadas a los `.py`; `sync_notebooks.py --desde-notebooks` para el futuro |
| Celda con `input()` (rompe la ejecución automática) | Reemplazada por una variable editable `MAX_LLAMADAS` |
| `draw_mermaid_png` requiere Internet | `mostrar_grafo`: imagen si hay red, dibujo ASCII si no |
| `.env` vacío | Se completó desde `.env.example` (modo offline, sin clave) |

## Métricas y límites

Los diez casos miden igualdad de IDs, validez de referencias y abstención; no son una
evaluación semántica exhaustiva. En offline, las decisiones de los agentes son guiones
explícitos (`ModeloGuionado`); herramientas, grafos, middleware, subagentes, archivos
virtuales y aprobaciones son reales, y los textos de los guiones se construyen con las
observaciones reales de las herramientas. Checkpoints y archivos del deep agent viven en
memoria. Los precios de los modelos se consultaron el 6/10/2026 y deben revisarse antes de
cada cohorte. La ejecución automática valida funcionamiento, no comprensión del grupo.
