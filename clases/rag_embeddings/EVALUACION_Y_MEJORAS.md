# Evaluación y mejoras de la clase

Revisión del 8 de octubre de 2026. La clase está preparada para principiantes en
RAG y embeddings, con código guiado, un centro cultural ficticio y un plan de
120 minutos. El notebook organiza sus contenidos por temas.

## Criterios de aceptación

| Requisito | Evidencia en el material |
|---|---|
| Partir sin conocimientos de RAG | Glosario inicial y progresión desde documentos y párrafos |
| Explicar visualmente | Diagrama de indexación/consulta, mapa manual, matriz real y ranking |
| Comparar embeddings y su uso | Small y Large sobre los mismos textos, preguntas, filtro y `k=4` |
| Usar una base vectorial real | Qdrant local en disco: creación, upsert, consulta, cierre y reapertura |
| Mostrar un flujo completo | Documentos → fragmentos → embeddings → Qdrant → contexto → respuesta y citas |
| Practicar decisiones | Cuatro ejercicios sobre `k`, vigencia, cobertura y fidelidad |
| Repetir sin API | Embeddings reales guardados, búsqueda nueva local y reproducción explícita de generaciones anteriores |
| Usar modelos con `.env` | Laboratorio voluntario para una pregunta nueva, sin claves en el notebook |

La proyección PCA es una ampliación opcional. La instalación y la preparación de
la API se realizan antes de la sesión. No se añadió Agentic RAG a esta introducción:
el estudiante necesita distinguir primero recuperación y generación. La continuación
está en [RAG y Agentic RAG](../../docs/RAG_Y_AGENTIC_RAG.md).

## Primer ciclo de feedback: cuatro problemas y sus correcciones

Una revisión independiente leyó la fuente y las guías y reprodujo los casos
problemáticos. Estas correcciones se hicieron antes de aceptar la clase:

| Problema encontrado | Consecuencia | Mejora aplicada |
|---|---|---|
| La limpieza final eliminaba Qdrant tras ejecutar todo | Repetir un ejercicio abría una base vacía | Limpieza con bandera apagada; la base permanece durante la práctica |
| El baseline literal mostraba un resultado y los embeddings cuatro | La comparación anunciada no era equivalente | Cuatro resultados por método, fuentes esperadas y cobertura; tabla de doce filas |
| Cambiar el contexto invalidaba el replay, pero la entrega exigía respuesta | La entrega parecía requerir clave | Ruta sin API: respuesta del estudiante claramente etiquetada y cita del contexto modificado |
| El ejercicio de fidelidad sólo registraba una cadena | Una corrección falsa podía quedar sin discutir | Frase de apoyo copiada y comprobada, corrección propia y revisión humana explícita |

También se añadió una celda para pasar de los IDs a los textos de cualquier caso
y método. El principiante puede inspeccionar evidencia sin memorizar identificadores.
La tabla de comparación se reorganizó para que sus resultados fueran legibles.

## Segundo ciclo de feedback: aceptación y comprobaciones

La revisión independiente volvió a leer la versión corregida y ejecutó el recorrido
offline. Confirmó que la base seguía disponible para otra consulta y que las cuatro
correcciones eran coherentes con los ejercicios. No encontró nuevos problemas
importantes que impidieran realizar la práctica.

La revisión principal comprobó además:

- Script y notebook sincronizados, con notebook docente sin outputs.
- Ejecución completa de ambos en un kernel nuevo, sin activar live.
- Cuatro figuras PNG emitidas por las celdas, sin errores de ejecución; inspección
  visual del flujo, matriz de similitudes y proyección de vectores reales.
- Las cuatro soluciones de referencia ejecutadas después de Run All: todas pasan.
- Tres puntos manuales y catorce fragmentos por colección Small/Large conservados
  al reabrir Qdrant; consultas y ejercicios repetibles durante la sesión.
- Pruebas de rechazo de otro espacio vectorial, caché alterada, contexto distinto
  y citas falsas; la reproducción no activa una llamada oculta al proveedor.
- Ruff aprobado para `src`, `scripts`, `tests`, `clases`, `proyectos` y `soluciones`.

Los tests del nuevo backend, replay y verificador suman **97 pruebas aprobadas**.
La suite completa dio **390 aprobadas y una falla previa**: el notebook
`clases/06_evaluacion_y_humano.ipynb` contiene una edición que difiere de su `.py`
en el apartado de pausa del grafo. Esa edición anterior se conservó.
La verificación aislada `--track rag-intro` pasó con script y notebook.

## Comprobación real de modelos

Se usó el `.env` existente sin imprimir credenciales. Los archivos preparados
conservan su procedencia, fecha, modelo, contexto y uso reportado:

| Operación real | Resultado observado |
|---|---|
| Batch Small, 23 textos | 1536 dimensiones; 642 tokens reportados |
| Batch Large, los mismos 23 textos | 3072 dimensiones; 642 tokens reportados |
| Cinco generaciones RAG con `gpt-6-luna` | Cuatro respuestas con citas y una abstención; 4041 tokens en conjunto |
| Pregunta nueva sobre materiales del taller | Embedding Large: 14 tokens; generación: 789 tokens; cita literal de CC-04-P2 válida |

La pregunta nueva recuperó la fuente de lápices, papel y cuaderno y respondió
con esa evidencia. Se ejecutó la misma celda live del notebook, activando su bandera
sólo en la comprobación. La versión entregada conserva `EJECUTAR_LIVE=False`.

Estos tokens describen ejecuciones observadas; las respuestas guardadas no generan
consumo nuevo. Una generación futura puede redactar distinto. La verificación
offline repite recuperación y controles; reproduce el texto de una generación
anterior, sin afirmar que ejecutó nuevamente el modelo.

## Límites y siguiente evaluación

El plan suma 120 minutos, pero todavía no se midió con una cohorte. Durante una
primera sesión se debe registrar dónde necesitan ayuda y ajustar la conducción.
La rúbrica evalúa representación, recuperación, metadatos, generación, fidelidad
y transferencia; ejecutar todas las celdas no acredita aprendizaje por sí solo.

Los cuatro casos comparativos no prueban superioridad general de ningún modelo.
La cobertura mide presencia de fuentes esperadas: no mide fidelidad, precisión
de todos los resultados ni probabilidad de respuesta correcta. El baseline cuenta
palabras exactas y no pretende representar un buscador léxico completo.

El control de citas comprueba IDs y frases literales, no todas las implicaciones
de la respuesta. Qdrant local y este corpus pequeño no evalúan escala, permisos,
actualización de índices ni rendimiento de un servidor.

## Repetir las comprobaciones

Desde la raíz del repositorio:

```bash
uv sync --locked
uv run pytest -q tests/test_rag_taller.py tests/test_rag_taller_replay.py tests/test_verification.py
uv run ruff check src scripts tests clases proyectos soluciones
uv run python scripts/verify.py --mode offline --track rag-intro
```

Los logs, notebook ejecutado, versiones y huellas quedan en
`reports/offline/rag-intro/`. Las comprobaciones adicionales de soluciones y consulta
nueva de esta revisión quedaron en `reports/rag-intro-feedback/`.
`reports/` es evidencia local ignorada por Git; los datos preparados y las guías sí
forman parte del material de la clase. Para comprobar live otra vez, activa la
bandera del notebook con el `.env` configurado.
