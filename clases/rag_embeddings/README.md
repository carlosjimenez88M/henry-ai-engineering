# RAG y embeddings desde cero

Una clase con recorrido principal de dos horas, explicaciones visuales, programación guiada
y un proyecto completo: responder preguntas sobre un centro cultural ficticio usando
sus documentos. No requiere conocer RAG, embeddings ni bases vectoriales.
El material se presenta sin emojis.

Camila necesita saber qué presentar para retirar un libro y cuándo devolverlo.
El ejemplo exige recuperar dos fuentes y permite observar cómo una norma archivada
puede contaminar el contexto.

Abre [el notebook de la clase](00_rag_y_embeddings.ipynb). Su fuente editable está
en [00_rag_y_embeddings.py](00_rag_y_embeddings.py). El recorrido se organiza por
temas; esta guía reúne preparación y conducción de los experimentos.

## Lo que el estudiante construye y comprende

1. Lee documentos y conserva procedencia al dividirlos en párrafos.
2. Explora un mapa manual y compara matrices Small/Large con los mismos textos y escala.
3. Crea puntos, los guarda en Qdrant local y consulta sus vecinos cercanos.
4. Compara búsqueda literal, Small y Large con las mismas preguntas y documentos.
5. Inspecciona contexto, generación, citas y ausencia de evidencia.
6. Modifica `k`, el filtro de vigencia y una pregunta del laboratorio.

El explorador interactivo reúne controles, ranking, tarjetas de fuentes, cobertura
y respuesta. Cada cambio ejecuta una búsqueda real en Qdrant con los embeddings
guardados; no realiza llamadas nuevas a modelos.

Las ampliaciones 12–15 añaden otras prácticas:

- Fragmentación por caracteres y solapamiento, con un gráfico de los cortes.
- Búsqueda híbrida con RRF, combinando rankings de palabras y embeddings.
- Precisión y recall al modificar `k`, y comparación controlada de la fusión.
- Actualización de metadatos y retiro de una norma, con comparación antes/después.

Estas ampliaciones continúan el recorrido principal y pueden trabajarse en otra sesión.

Hay cuatro ejercicios con intentos, resultados esperados y soluciones desplegables.
Las comprobaciones iniciales pueden mostrar `False`: son tareas por resolver.
La entrega exige explicar el efecto de una modificación y respaldar una respuesta.
Sin clave, el estudiante redacta y etiqueta su propia respuesta al contexto modificado;
no reutiliza una generación preparada para otro contexto. La generación nueva es voluntaria.

## Preparación

Desde la raíz de `henry-ai-engineering`:

```bash
uv sync --locked
uv run python scripts/start_rag_class.py
```

El comando comprueba dependencias, cachés y Qdrant, registra el kernel
**Henry AI Engineering (.venv)** y abre el notebook en JupyterLab. Selecciona ese
kernel si el editor lo solicita. La instalación se prepara antes de la clase; después
puedes ejecutar el notebook sin clave, sin servidor Qdrant y sin descargar modelos.
Necesitas leer listas, diccionarios y llamadas de funciones; el notebook ofrece
un recordatorio y código preparado. Para practicar Python primero, usa
[la clase de fundamentos](../python_ai/00_python_ai_engineering.ipynb).

Para comprobar todo sin abrir la interfaz:

```bash
uv run python scripts/start_rag_class.py --check
uv run python scripts/check_rag_environment.py --live
```

El primer comando no usa API. El segundo valida `.env` con un embedding nuevo y
una generación con fuentes; guarda el informe en `reports/rag-intro-environment/`.
Los valores y sufijos de las claves nunca aparecen en ese diagnóstico.
No necesitas Docker ni iniciar un servidor Qdrant.

Si ves `VBox(...)` en lugar de controles, reinicia JupyterLab después de
`uv sync --locked` y confirma el kernel del proyecto. Los widgets necesitan un
kernel vivo: la vista estática de GitHub no ejecuta sus controles. Las celdas de
Python permiten repetir los mismos experimentos.

## Conducir los experimentos visuales

1. Con Camila, Large y vigentes, alterna `k=1` y `k=4`: la cobertura pasa de 0/2
   a 2/2. Lee la devolución recuperada primero para explicar qué información falta.
2. Para ver vigencia, elige la pregunta del plazo, Large y `k=4`. Desactiva el filtro:
   aparece CC-07-P1. Con Camila y `k=4`, ambos filtros pueden devolver los mismos IDs.
3. Con Mongolia, Large, `k=4` y vigentes, compara candidatos con abstención.

Pide predecir antes de tocar los controles y explicar el cambio usando una fuente.
Si desaparece la respuesta guardada, el nuevo contexto requiere otra respuesta:
una del estudiante etiquetada o una generación live voluntaria.

En la comparación híbrida mantén Small y cuatro resultados finales. Las seis
candidaturas por lista son la entrada de RRF. En este ejemplo, Small e híbrido
obtienen la misma precisión y recall; combinar búsquedas no garantiza mejorar.
Al retirar una norma, compara cobertura antes/después y observa que siguen
guardados catorce puntos: filtrar no equivale a borrar físicamente.

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
uv run pytest -q tests/test_rag_taller*.py tests/test_rag_environment.py tests/test_verification.py
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
Los controles usan [Jupyter Widgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html).
