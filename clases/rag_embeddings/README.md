# RAG y embeddings desde cero

Una clase diseñada para dos horas, con explicaciones visuales, programación guiada
y un proyecto completo: responder preguntas sobre un centro cultural ficticio usando
sus documentos. No requiere conocer RAG, embeddings ni bases vectoriales.
El material se presenta sin emojis.

Abre [el notebook de la clase](00_rag_y_embeddings.ipynb). Su fuente editable está
en [00_rag_y_embeddings.py](00_rag_y_embeddings.py). El recorrido se organiza por
temas; el [plan docente](GUIA_DOCENTE.md) reúne distribución, preguntas y evaluación.

## Lo que el estudiante construye y comprende

1. Lee documentos y conserva procedencia al dividirlos en párrafos.
2. Explora un mapa manual y una matriz de embeddings aprendidos realmente.
3. Crea puntos, los guarda en Qdrant local y consulta sus vecinos cercanos.
4. Compara búsqueda literal, Small y Large con las mismas preguntas y documentos.
5. Inspecciona contexto, generación, citas y ausencia de evidencia.
6. Modifica `k`, el filtro de vigencia y una pregunta del laboratorio.

Hay cuatro ejercicios con intentos, resultados esperados y soluciones desplegables.
Las comprobaciones iniciales pueden mostrar `False`: son tareas por resolver.
La entrega exige explicar el efecto de una modificación y respaldar una respuesta.
Sin clave, el estudiante redacta y etiqueta su propia respuesta al contexto modificado;
no reutiliza una generación preparada para otro contexto. La generación nueva es voluntaria.

## Preparación

Desde la raíz de `henry-ai-engineering`:

```bash
uv sync --locked
uv run jupyter lab clases/rag_embeddings
```

Selecciona el kernel `.venv`. La instalación se prepara antes de la clase; después
puedes ejecutar el notebook sin clave, sin servidor Qdrant y sin descargar modelos.
Necesitas leer listas, diccionarios y llamadas de funciones; el notebook ofrece
un recordatorio y código preparado. Para practicar Python primero, usa
[la clase de fundamentos](../python_ai/00_python_ai_engineering.ipynb).

Qdrant se ejecuta en modo local con almacenamiento en disco. El notebook cierra y
reabre la base para comprobar persistencia; conserva la carpeta durante la sesión
para repetir ejercicios. La limpieza final requiere activar su bandera.
Para conservar una práctica personal usa una carpeta propia bajo `.local/`.
El modo local usa búsqueda exacta para este pequeño corpus: no prueba el rendimiento
de un servidor, su índice HNSW ni un despliegue distribuido.

## Qué es real y qué se repite

| Parte | Procedencia | Al ejecutar todo por defecto |
|---|---|---|
| Mapa con dos coordenadas | Valores inventados para explicar geometría | Se dibuja localmente |
| Embeddings Small/Large | API de OpenAI, con dimensiones, textos, fecha y tokens | Se leen vectores reales guardados |
| Almacenamiento y búsqueda | Qdrant local | Se ejecutan de nuevo |
| Respuestas preparadas | Generaciones reales con el contexto guardado | Se reproduce la respuesta anterior, sin llamada nueva |
| Laboratorio live | Modelos y clave de `.env` | Desactivado hasta cambiar la bandera |

El corpus contiene siete documentos y catorce fragmentos. La caché de embeddings
incluye 23 textos por modelo: fragmentos, cinco preguntas y cuatro frases para
comparación. Las respuestas guardadas corresponden a cinco consultas con Large,
`k=4` y filtro de vigencia. No sirven automáticamente para otra pregunta o contexto.

Cambiar `k` no cambia el embedding de la pregunta: cambia el contexto recuperado.
La comparación usa cuatro resultados en los tres buscadores y mide cobertura
de fuentes esperadas. Es una evaluación pequeña para aprender, no un benchmark general.
Cambiar el modelo de embeddings exige otra colección compatible. Cambiar el
generador no modifica los vectores de los documentos. Los scores no son porcentajes
de certeza, y una cita existente no garantiza fidelidad de todas las afirmaciones.

## Laboratorio con modelos

El `.env` privado de la raíz configura la clave y el generador por rol. La clase
compara explícitamente `text-embedding-3-small` y `text-embedding-3-large`; las
colecciones identifican modelo y dimensiones. No pegues ni imprimas la clave.

En el notebook, cambia `EJECUTAR_LIVE=True` y escribe una pregunta propia. La ruta
ejecuta una petición para el embedding de la pregunta y otra para generar una
respuesta con fuentes. Los fallos se propagan, sin simulación como reemplazo oculto.
Los tokens de los snapshots pertenecen a su preparación, no a una ejecución nueva.

Para regenerar los materiales preparados (llamadas reales explícitas):

```bash
uv run python scripts/build_rag_taller_cache.py --live
uv run python scripts/build_rag_taller_respuestas.py --live
```

Primero se generan dos batches de embeddings; después, cinco generaciones. Si
cambia el corpus se regeneran ambos materiales. Conserva modelo, fecha, preguntas,
contextos y consumo observado. La validación y límites están en
[EVALUACION_Y_MEJORAS.md](EVALUACION_Y_MEJORAS.md).

## Verificar la clase

```bash
uv run pytest -q tests/test_rag_taller.py tests/test_rag_taller_replay.py
uv run ruff check clases/rag_embeddings src/henry_agents/rag_taller*.py scripts/build_rag_taller*.py
uv run python scripts/verify.py --mode offline --track rag-intro
```

El último comando ejecuta script y notebook en un kernel nuevo, conserva evidencia
en `reports/offline/rag-intro/` y no activa el laboratorio live. El notebook docente
se guarda sin outputs; sus figuras iniciales también están disponibles en
[assets](assets/README.md).

Referencias primarias: [embeddings](https://developers.openai.com/api/docs/guides/embeddings),
[búsqueda semántica](https://developers.openai.com/api/docs/guides/retrieval),
[cliente Qdrant local](https://github.com/qdrant/qdrant-client#local-mode),
[colecciones](https://qdrant.tech/documentation/manage-data/collections/) y
[consulta vectorial](https://qdrant.tech/documentation/search/).
