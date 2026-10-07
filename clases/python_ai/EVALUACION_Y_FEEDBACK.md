# Evaluación del taller y dos ciclos de feedback

Revisión del 7 de octubre de 2026. Se evaluó y mejoró primero la clase de fundamentos;
después se incorporó la ampliación solicitada, con FastAPI, más
ejercicios y uso de modelos sobre escenas originales de cómics. Los ciclos de
esta página corresponden al material docente; dentro del proyecto también hay
dos ciclos distintos de crítica y reescritura de cada guion.

## Criterios y puntuación editorial

Cada criterio se puntúa de 1 a 5. La evaluación fue independiente de quien editó
los notebooks. Son juicios sobre el diseño y la evidencia técnica, sin medición
con estudiantes. El alcance cambió entre la versión inicial y la final: la
comparación no demuestra un aumento cuantificado del aprendizaje.

| Criterio | Clase inicial | Taller antes de corregir el ciclo 2 | Taller final |
|---|---:|---:|---:|
| Claridad y secuencia | 4 | 4 | 4 |
| Práctica y feedback | 2 | 4 | 4 |
| Alineación de objetivos y actividades | 3 | 4 | 5 |
| Relevancia para AI Engineering | 4 | 4 | 5 |
| Organización y conducción docente | 3 | 3 | 4 |
| Reproducibilidad | 5 | 5 | 5 |
| **Total** | **21/30** | **24/30** | **27/30** |

Claridad y práctica conservan margen por la cantidad de conceptos nuevos:
clases, decoradores, contratos anidados, `zip` y dobles de prueba necesitan
conducción docente. La comprensión de los temas debe observarse en una cohorte.

## Ciclo 1: crítica, cambios y evaluación

Se revisó la clase original de fundamentos y se ejecutó su script.

| Hallazgo | Cambio aplicado | Evidencia |
|---|---|---|
| El filtro era una demostración; no exigía modificar `if` ni `for` | Intento que acepta texto vacío; el alumno debe corregir la condición | La comprobación cambia de `False` a `True` y conserva DOC-01/DOC-03 |
| Las actividades pedían crear una celda sin ofrecer un intento | Cuatro celdas editables con comprobaciones y soluciones desplegables | Las celdas iniciales ejecutan sin detener el notebook |
| El documento escrito por el alumno no se reutilizaba | Un lote nuevo usa ese documento para preparar contexto y mensajes | La referencia resuelve la transferencia con un dato distinto |
| JSON, imports y serialización sobrecargaban el cierre | JSON pasó a referencia opcional y al bloque 01 | El bloque de fundamentos se concentra en preparar datos y mensajes |
| Se mencionaban errores sin una reparación visible | Diagnóstico de `KeyError` y distinción de `print` frente a `return` | Conteo correcto con un lote y con una lista vacía |

Se ejecutaron script, notebook en un kernel nuevo y las cuatro soluciones después
de los cambios. Se conservaron las versiones inicial e intermedia en
`reports/python-ai-feedback/ciclo-0/` y `ciclo-1/` (evidencia local excluida de Git).

## Ciclo 2: crítica del taller ampliado y correcciones

Se revisaron los cuatro scripts, el recorrido, el contrato y el backend. La segunda
revisión encontró seis problemas; se corrigieron antes de ejecutar la versión final.

| Hallazgo | Cambio aplicado | Evaluación después del cambio |
|---|---|---|
| La respuesta de la API sólo conservaba el guion final | `versiones[0/1/2]` guarda borrador y ambas reescrituras como copias independientes | Se puede leer y comparar cada cambio sin inferirlo de una crítica |
| El ejemplo de integración usaba mensajes sin definición | Celda con instrucciones, JSON de pedido/fuentes y campos de `BriefComic` | La entrada del extractor es visible y no contiene credenciales |
| El conteo de cambios sólo probaba cadenas A/B/C | Se construyen listas mediante bucles sobre los guiones de ambos ciclos | La misma función se aplica a casos pequeños y al proyecto |
| El fallo del proveedor perdía fase y ciclo | Excepciones con atributos seguros `etapa` y `ciclo`; el notebook los muestra | Los dobles de prueba confirman contexto sin revelar detalles internos |
| La ejecución real podía interrumpir la práctica del grupo | Preparación docente previa o demo en otro kernel mientras el grupo practica offline | Se conservan resultados reales y no se oculta un fallo del proveedor |
| El contrato de práctica y el central tenían reglas diferentes | Ejemplos de cantidad predeterminada y rechazo de héroes repetidos | Se explica por qué compartir nombres de campos no iguala todas las reglas |

La revisión posterior confirmó que los seis hallazgos quedaron atendidos y asignó
27/30. El backend tuvo además una revisión independiente de snapshots, contexto de
errores y adaptación de los esquemas a Responses API sin enviar peticiones reales.

## Ciclo 3: embeddings y ajustes para principiantes

Una revisión docente posterior encontró que el salto del contrato Pydantic (01) al
workflow completo (antes 02) era grande para alumnos con poca base, que nunca se
veía una sola llamada a un modelo y que el taller no tenía ningún gráfico.

| Hallazgo | Cambio aplicado | Evidencia |
|---|---|---|
| Faltaba el puente "texto → números" antes de los modelos | Nuevo notebook 02 de embeddings; modelos pasa a 03 y FastAPI a 04 | Tres ejercicios con comprobaciones; con las soluciones aplicadas todas dan `True` |
| Sin gráficos en el taller | Mapa a mano, mapa de calor, proyección PCA a 2D y barras de búsqueda con matplotlib | Cuatro figuras generadas en la ejecución del notebook |
| Embeddings reales exigirían clave a cada alumno | Cache de 17 vectores de `text-embedding-3-large` calculado una vez; laboratorio live con bandera | Seis pruebas: cache completo, vectores unitarios y offline sin llamadas |
| No se veía una llamada mínima a un modelo | Sección 0 en 03: una llamada con los mensajes del notebook 00 | Probada una vez con `gpt-6-luna`: 34 tokens de entrada y 56 de salida |
| Advertencias repetidas en casi cada sección | Un recuadro "Límites" por notebook y frases redundantes retiradas | Mismas celdas de código salvo las nuevas |
| Ejercicio 3 de 00 con la celda antes de su enunciado; soluciones al final en 00, 03 y 04 | Enunciado antes de la celda; cada solución bajo su ejercicio | Contenido de las soluciones sin cambios |
| Frase confusa en el ejercicio 3 de modelos | Reescrita: por qué Pydantic acepta el ID inventado y qué control lo detecta | — |

Los datos reales dieron un resultado útil para la clase: con estas frases, compartir
héroe produce más similitud (0.58) que compartir tema (0.45), pero las consultas sin
nombre de héroe recuperan las cuatro frases de su tema. La proyección 2D conserva el
27 % de la variación, y el notebook lo usa para mostrar que el mapa no sirve para medir.

## Validación técnica final

| Comprobación | Resultado |
|---|---|
| Organización | Cinco bloques progresivos con ejercicios y un proyecto acumulativo |
| Scripts del taller | 5/5 ejecutados offline |
| Notebooks | 5/5 ejecutados completos, cada uno en un kernel nuevo |
| Celdas | 160 celdas; 71 de código |
| Ejercicios y soluciones | 19 intentos; las 3 soluciones nuevas de embeddings se ejecutaron en el ciclo 3 |
| Backend de cómics | 58 pruebas aprobadas, incluidas rutas, contratos, versiones y dobles de modelos |
| Verificador del repo | 7 pruebas aprobadas; incorpora `--track python` |
| Ruff del material y backend nuevos | Sin errores |
| Parejas `.py` / `.ipynb` | Coinciden; los notebooks se guardan ejecutados offline, con ejercicios sin resolver |
| Modelos reales | Cache de embeddings y una llamada simple ejecutados; workflow completo live pendiente |

Las comprobaciones prueban el programa y sus contratos, sin acreditar creatividad,
fidelidad semántica, resistencia general a prompt injection ni calidad de un juez
LLM. Un ID válido y una cita literal correcta verifican procedencia limitada;
el docente y el alumno deben leer el contenido.

La suite general terminó con **294 pruebas aprobadas y un fallo** en
`test_notebook_sources_match_scripts`: detectó una desincronización previa entre
notebooks de la ruta avanzada y sus scripts. Esos notebooks tenían cambios ajenos
a este taller; se conservaron. La validación de los cuatro notebooks Python se
registra por separado y no se presenta como un éxito de toda la suite del repositorio.

## Repetir y conservar evidencia

Desde la raíz del repositorio:

```bash
uv run pytest -q tests/test_comics.py tests/test_verification.py
uv run ruff check clases/python_ai src/henry_agents/comics.py src/henry_agents/comics_api.py tests/test_comics.py scripts/verify.py
uv run python scripts/verify.py --mode offline --track python
```

El verificador registra versiones, SHA-256, resultado por fase y notebooks
ejecutados en `reports/offline/python/verification.json`. El reporte de soluciones
de esta revisión está en `reports/python-ai-feedback/final-solutions.json`;
ambos son evidencia local, no archivos que el alumno deba descargar de Git.
Los bloques `<details>` contienen las soluciones reproducibles.

La práctica real se activa en el notebook 02 con la bandera explícita. Conserva
pedido, modelos usados, fecha, snapshots, revisiones, tokens y fallos observados.
No imprimas ni adjuntes el `.env`. El workflow admite siete invocaciones lógicas
y desactiva reintentos del SDK; repetirlo inicia otra ejecución.

## Próximo feedback con estudiantes

En la primera cohorte registra las dificultades al corregir el filtro, las
actividades que requieren ayuda y la capacidad de explicar un 422 frente a un
503. Revisa si alguien puede completar un pedido nuevo sin copiar la solución.
Usa esos datos para ajustar la densidad: no interpretes cantidad de
llamadas como aprendizaje. La comparación Astra y el servidor real son ampliaciones
opcionales del recorrido.
