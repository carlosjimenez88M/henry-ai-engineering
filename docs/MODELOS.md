# Modelos del curso

Verificados en documentación oficial el **6 de octubre de 2026**. La familia actual
incluye GPT-6 Luna, GPT-6.1 Sol y GPT-6 Astra, con distintos costos y capacidades.
El curso mantiene esos roles para que una actividad simple no requiera el modelo
más caro. [Guía oficial de modelos actuales](https://developers.openai.com/api/docs/guides/latest-model).

| Variable de `.env` | Predeterminado | Papel en el curso |
|---|---|---|
| `OPENAI_MODEL` | `gpt-6-luna` | Actividades breves y especialistas |
| `OPENAI_MODEL_AGENT` | `gpt-6.1-sol` | Coordinadores y decisiones de Agentic RAG |
| `OPENAI_MODEL_RAG` | `gpt-6-luna` | Generar respuesta estructurada y revisar fidelidad |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Experimento voluntario de recuperación semántica |
| `OPENAI_REASONING_EFFORT` | `low` | Esfuerzo de razonamiento del chat |

`gpt-6-astra` queda disponible para comparar tareas exigentes. Para probarlo, cambia
solo el rol que quieres comparar y conserva preguntas, corpus y presupuesto.
[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) y
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).

Los embeddings vigentes son de la tercera generación. `text-embedding-3-large`
es el predeterminado del experimento; `text-embedding-3-small` permite comparar
una alternativa de menor costo de la misma generación. La elección de large no
garantiza mejor resultado en nuestro corpus: mide recall con ambos antes de
concluir. Se llaman mediante `OpenAIEmbeddings`, nunca como modelo de chat.
[Documentación de embeddings](https://developers.openai.com/api/docs/guides/embeddings).

La configuración está centralizada en
[`config.py`](../src/henry_agents/config.py); [`rag.py`](../src/henry_agents/rag.py)
consulta los nombres por rol. `.env` puede sobrescribirlos y no se versiona.
`model_name()` solo lee configuración: no llama a la API ni exige clave. El modo
offline tampoco usa modelos remotos. Reinicia el kernel después de editar `.env`.

## Compatibilidad y verificación

El chat usa Responses API y `reasoning_effort="low"`. GPT-6.1 Sol y Astra requieren
Responses para herramientas y no admiten esfuerzo `none`. No enviamos
`temperature` ni `top_p` cuando hay razonamiento. Se permite un máximo de 8000
tokens de salida, incluidos tokens de razonamiento, por invocación de chat.
Eso no fija el consumo de un grafo con varias invocaciones. La configuración tiene
timeout de 120 segundos y hasta dos reintentos por llamada.
[Guía oficial de migración](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md#migration-quickstart).

Las pruebas inspeccionan las solicitudes del SDK y usan dobles para decisiones,
generación, fidelidad y embeddings. La validación documentada no hizo llamadas
reales porque no había una clave disponible. Los nombres se verificaron en la
documentación; la disponibilidad en cada cuenta requiere una prueba propia.

Antes de una demostración con API:

```bash
uv run python scripts/doctor.py --live
uv run python scripts/evaluate_rag.py --mode live --case RAG-06
```

Consulta el reporte, la traza, los tokens y las citas. No sustituyas un error de API
por una simulación silenciosa. El catálogo oficial y los precios pueden cambiar:
vuelve a verificarlos al comenzar una cohorte. Revisa
[VALIDACION.md](../VALIDACION.md) para separar lo ejecutado de lo pendiente.
