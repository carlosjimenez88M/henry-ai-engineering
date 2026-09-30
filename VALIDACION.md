# Validación · Revisión cultural y arquitecturas de LangGraph

Última comprobación: **30 de septiembre de 2026**. Se ejecutaron los cuatro
notebooks ampliados y sus scripts equivalentes. Cada
notebook contiene un recorrido con dos pausas, explicación, construcción
por etapas, predicciones, un taller, soluciones y un ticket de salida.

Después de estas ejecuciones se realizó una edición de presentación para retirar
los cronogramas y las duraciones visibles, sin cambiar las celdas de código.
Se comprobó que cada celda de código y la estructura ejecutable de los scripts
son idénticas a las de la versión probada. Scripts y notebooks siguen sincronizados
y Ruff pasa. No se repitieron llamadas API por esta edición de Markdown; los reportes
anteriores conservan las huellas de los archivos antes de este cambio de presentación.

## Resultados ejecutados

| Comprobación | Resultado |
|---|---|
| pytest | 84 pruebas aprobadas |
| Ruff | Sin errores en src, scripts, tests y clases |
| Scripts offline | 4 de 4 aprobados |
| Notebooks offline, kernels nuevos | 4 de 4 aprobados |
| Scripts live con OpenAI | 4 de 4 aprobados |
| Notebooks live con OpenAI, kernels nuevos | 4 de 4 aprobados |
| Catálogo, golden dataset offline | 10 de 10 casos correctos |
| Catálogo, golden dataset live | 10 de 10 casos correctos |
| Consistencia de materiales | Scripts/notebooks sincronizados; código idéntico al ejecutado |
| Instalación limpia, sin .env ni entorno anterior | uv sync --locked --offline; 84 tests y 8 recorridos offline aprobados |
| Archivos publicables | 41 archivos revisados; sin claves detectadas; 4 notebooks sin outputs |

Entorno probado: macOS ARM64, Python 3.13.15, langgraph 1.2.12,
langchain-core 1.6.6 y langchain-openai 1.6.7. Modelo real: gpt-4.1-mini.
Las dependencias no cambiaron respecto al lockfile existente.
La instalación limpia usó una carpeta temporal con los archivos publicables y la
caché local de paquetes; no dependió del entorno virtual ni del .env originales.
Esto verifica reconstrucción desde el lockfile, no una descarga nueva desde PyPI.

## Evidencia local

- [Verificación offline de clases](reports/offline/verification.json).
- [Verificación live de clases](reports/live/verification.json).
- [Verificación offline desde instalación limpia](reports/clean-install-verification.json).
- [Golden dataset cultural offline](reports/catalog-evaluation-offline.json).
- [Golden dataset cultural live](reports/catalog-evaluation-live.json).
- [Trabajo de la clase 4 offline](reports/cultural-evaluation-offline.json).
- [Trabajo de la clase 4 live](reports/cultural-evaluation-live.json).

Los notebooks ejecutados y los logs están en reports/offline y reports/live.
Los originales de clases no tienen outputs y siguen siendo publicables.
Los reportes están excluidos de Git y se regeneran mediante los comandos del README.
La CI remota está configurada, pero no se ejecutó ni publicó el repositorio en GitHub.
Cada verification.json registra identificador de corrida, inicio, fin, versiones,
huellas SHA-256 y resultado individual de scripts/notebooks. Antes de verificar se
sustituye el reporte anterior: una ejecución fallida no conserva un éxito antiguo.

## Correcciones realizadas en esta revisión

- La búsqueda usa el nombre de colección como filtro: mencionar Batman no basta
  para responder sobre un tema ausente, como una vacuna marciana.
- El agente exige una observación válida de su herramienta en el turno actual;
  un resultado vacío conduce a abstención.
- Cada solicitud reinicia intentos y decisión. Una aprobación anterior no aprueba
  automáticamente otra consulta; las decisiones inválidas se pueden corregir.
- Los límites de llamadas e intentos rechazan booleanos, decimales y valores no positivos.
- Las clases muestran el contrato y el wrapper de la herramienta; el prompt que
  construye el estudiante participa en la llamada real del modelo.
- Los talleres 3 y 4 separan el punto de partida de la solución y la clase 4
  comprueba qué sucede al agotar el límite de intentos.

## Comportamientos cubiertos

La nueva suite cultural comprueba entradas vacías, extensas y fuera del contrato;
tipos y límites de top_k; filtros de colección/tipo; normalización de tildes;
ranking estable; ausencia de evidencia; rechazo de una cita inventada; ejecución
de herramientas y terminación de un modelo que no deja de pedir acciones.

También comprueba planes vacíos/desconocidos, deduplicación de colecciones,
preservación de resultados de workers, corrección de una falla inyectada,
agotamiento del límite sin entregar el borrador inválido, escalación sin evidencia,
interrupción/reanudación, rechazo, decisiones inválidas y aislamiento de solicitudes.
Los tests anteriores de soporte y ventas permanecen como pruebas de compatibilidad.

La ejecución de los notebooks verifica además los grafos construidos dentro de
sus celdas: secuencia, routing, paralelo con barrera, Send con reducer, revisión
y la etapa humana separada. No se ignoran errores de celda. Solo se capturan las
excepciones anunciadas como demostraciones deliberadas.

## Métricas y límites

Los diez casos miden igualdad exacta de IDs recuperados, validez de referencias y
comportamiento de abstención. Cada métrica fue 1.0 en las corridas offline y live.
Esto no es una evaluación semántica exhaustiva ni una garantía sobre todas las
consultas futuras. Las respuestas del modelo pueden variar.

El corpus es ficticio y lexical; las canciones no incluyen letras ni audio.
El modo offline no usa un LLM. El paralelo y los workers deterministas no se
anuncian como agentes autónomos. Supervisor y handoff se comparan en diseño;
no se afirma haber implementado sistemas conversacionales completos de esos tipos.

Los checkpoints viven en memoria del proceso. La aprobación es una simulación
sin efectos externos. No se probaron persistencia durable, autenticación, carga
ni despliegue de un servicio. No se reintrodujo un servicio externo de trazas.

## Validación pedagógica pendiente de aula

Los notebooks tienen 28, 29, 22 y 29 celdas respectivamente, con explicación y
código intercalados. La ejecución automatizada valida funcionamiento, no demuestra
por sí sola la comprensión del grupo.
El recorrido contempla actividad, discusión y pausas; el docente debe observar
la primera cohorte y ajustar variantes conservando objetivos, pausas y cierre.
