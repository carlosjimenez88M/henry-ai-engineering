# Python para AI Engineering

Desde los fundamentos hasta una API que produce y revisa escenas de cómics con
modelos de IA. Cinco notebooks, un proyecto acumulativo y ejercicios con intentos,
comprobaciones y soluciones. El material de este taller se presenta sin emojis.

Las escenas son originales y ficticias: usamos Batman, Spider-Man y los Cuatro
Fantásticos como referencias conocidas, sin reproducir guiones ni afirmar canon.

## Recorrido de la clase

| Bloque | Trabajo práctico | Notebook |
|---|---|---|
| Fundamentos | Texto, tipos, listas, diccionarios, filtros y retorno de funciones | [00](00_python_ai_engineering.ipynb) |
| Datos y contratos | Archivos, JSON, errores, anotaciones y Pydantic | [01](01_datos_y_contratos.ipynb) |
| Embeddings | Texto como vectores, similitud coseno, mapa de calor, mapa 2D y búsqueda | [02](02_embeddings_visual.ipynb) |
| Modelos de IA | Primera llamada, brief, guion, dos ciclos de crítica/reescritura y consumo | [03](03_modelos_y_revision.ipynb) |
| FastAPI | HTTP, endpoints, validación, pruebas y proyecto final | [04](04_fastapi_comics.ipynb) |

Cada notebook empieza desde sus propios datos e imports: no depende del estado de
otro kernel. Los intentos iniciales son programas incompletos pero ejecutables:
algunas comprobaciones muestran `False` hasta que los corriges. Ejecutar todo sin
error comprueba reproducción; resolver las actividades exige observar el cambio
y explicar tu solución.

## Preparación antes de clase

Sigue [la instalación del repo](../../docs/INSTALACION.md). Desde su raíz:

```bash
uv sync --locked
uv run jupyter lab clases/python_ai
```

En VS Code abre `henry-ai-engineering`, selecciona el kernel `.venv` y ejecuta con
Shift + Enter. Instala y prueba el entorno antes de comenzar los notebooks.
Las dependencias ya están en `uv.lock`; no se requiere otro framework.

## Uso del `.env` y modelos reales

El `.env` privado de la raíz configura `OPENAI_API_KEY`, `OPENAI_MODEL` y
`OPENAI_MODEL_AGENT`. El curso carga estos valores mediante
[config.py](../../src/henry_agents/config.py). No pegues ni imprimas la clave en
una celda. Reinicia el kernel si modificas la configuración.

El notebook 02 grafica embeddings reales de `OPENAI_EMBEDDING_MODEL` leídos de
[un cache](../../src/henry_agents/data/embeddings_comics.json): todo el grupo ve
los mismos vectores sin clave ni costo. Su laboratorio final, con bandera, calcula
frases nuevas en vivo. Para regenerar el cache (requiere clave), desde la raíz:
`uv run python -m henry_agents.embeddings_comics`.

El notebook 03 incluye una bandera para una primera llamada simple y otra para el
laboratorio real del flujo completo. Desactivada,
el flujo usa guiones explícitos y muestra `llamadas_modelo=0`; no lo presentamos
como inferencia con un LLM. Activada, utiliza los modelos configurados, propaga
fallos y no vuelve silenciosamente a una simulación.

El flujo completo prepara un brief, redacta y realiza **dos ciclos** de crítica
y reescritura, seguidos de una revisión final: máximo siete invocaciones lógicas
por ejecución completa. Reejecutar consume de nuevo; el presupuesto por petición
no limita el consumo de toda una clase. Se reportan tokens observados, no costos
inventados.

Luna atiende extracción y crítica; Sol redacta y revisa. El notebook explica cómo
comparar un crítico Astra de forma voluntaria. Consulta [modelos](../../docs/MODELOS.md)
y verifica disponibilidad en tu cuenta. Las pruebas locales y las comprobaciones
reales realizadas se documentan por separado en [la evaluación](EVALUACION_Y_FEEDBACK.md).

## API del proyecto

La API usa el mismo contrato y flujo que los notebooks. El cliente HTTP envía tema,
héroes, tono y cantidad de viñetas; el servidor decide el modo. Desde la raíz:

```bash
uv run uvicorn henry_agents.comics_api:app --host 127.0.0.1 --port 8000
```

Usa `COURSE_MODE=offline` para practicar gratis. Para la demostración real, completa
la clave local y establece `COURSE_MODE=live` en `.env`, luego reinicia el servidor.
Abre `http://127.0.0.1:8000/docs` para peticiones interactivas. Detén el servidor
con Ctrl + C. El notebook usa `TestClient` y no deja un servidor corriendo.

| Endpoint | Resultado esperado |
|---|---|
| `GET /health` | Estado del servicio sin llamar al modelo |
| `GET /fuentes` | Escenarios locales con IDs |
| `POST /comics` | Brief, guion, revisiones y consumo observado |

FastAPI valida cuerpo y forma de respuesta. Python comprueba límites e IDs;
el modelo revisa calidad de texto. Son responsabilidades diferentes: schema
válido y revisión favorable no certifican verdad ni resistencia universal a
instrucciones maliciosas. La API es una práctica local, sin publicación automática.

## Evidencia de aprendizaje

Entrega un pedido propio, su JSON, una modificación al filtro, una función con
retorno y tres peticiones HTTP: válida, inválida y de diagnóstico. Explica la traza
de los dos ciclos: qué observó el crítico y qué cambió el redactor. Si trabajas
offline, identifica los guiones; con API, registra tokens reales y errores además
de la calidad del resultado.

| Criterio | Evidencia verificable |
|---|---|
| Fundamentos | Corrige el filtro y devuelve el conteo con dos entradas |
| Datos | Lee/escribe JSON y rechaza un pedido fuera del contrato |
| Modelos | Distingue fases, referencias, crítica y consumo |
| API | Obtiene 200/422 según el cuerpo y separa errores de entrada y proveedor |
| Transferencia | Ejecuta un caso nuevo y explica un resultado que necesita corrección |

Observa qué conceptos requieren apoyo en la primera cohorte. Prioriza corregir
intentos y explicar las decisiones de código; la comparación de modelos puede
usarse como demostración. No evalúes cantidad de llamadas como calidad.
Prepara la ejecución real en una copia del notebook. El notebook 03 explica cómo
mostrar esa evidencia o realizar una demostración en otro kernel mientras el grupo
trabaja con ejercicios offline.

## Edición y verificación

Los `.py` percent son las fuentes editables. A diferencia del resto del curso, los
`.ipynb` de esta carpeta se guardan **ejecutados offline**: se ven los resultados y
los gráficos al abrirlos, sin clave. Los ejercicios quedan sin resolver, con sus
comprobaciones en `False`, y las banderas de modelos reales desactivadas.
Conserva intentos en una copia personal y usa el
[flujo de edición](../../docs/INSTALACION.md#para-docentes-editar-las-clases).

Después de editar un `.py`, regenera y ejecuta solo esta carpeta:

```bash
make notebooks-python   # o: uv run python scripts/sync_notebooks.py --solo python_ai --ejecutar
```

`make notebooks` regenera todas las clases **sin** salidas; si lo usas, repite
después el comando anterior.

```bash
uv run pytest -q tests/test_comics.py tests/test_embeddings_comics.py
uv run ruff check clases/python_ai src/henry_agents/comics.py src/henry_agents/comics_api.py src/henry_agents/embeddings_comics.py tests/test_comics.py tests/test_embeddings_comics.py
uv run python scripts/verify.py --mode offline --track python
```

El último comando ejecuta los cinco scripts y sus notebooks en kernels nuevos;
guarda evidencia en `reports/offline/python/`. Las banderas de modelos quedan
desactivadas. La verificación live del taller se realiza explícitamente desde el
notebook 03; no se presenta una ejecución offline como una prueba del proveedor.

Los hallazgos y cambios de los ciclos de evaluación están en
[EVALUACION_Y_FEEDBACK.md](EVALUACION_Y_FEEDBACK.md).

Referencias primarias: [cuerpos HTTP](https://fastapi.tiangolo.com/tutorial/body/),
[respuestas de FastAPI](https://fastapi.tiangolo.com/tutorial/response-model/),
[TestClient](https://fastapi.tiangolo.com/tutorial/testing/) y
[salidas estructuradas de OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs).
