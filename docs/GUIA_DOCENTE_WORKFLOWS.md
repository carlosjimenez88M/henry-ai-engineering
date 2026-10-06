# Guía docente · Ruta inicial de workflows

Esta guía acompaña ocho clases completas en `clases/agentic_workflows/`.
El objetivo es que alguien que empieza pueda explicar un flujo, modificarlo y
demostrar una falla. No se evalúa memorizar APIs ni crear la mayor cantidad de agentes.

## Preparación

Desde la raíz ejecuta `uv sync --locked`, `scripts/doctor.py`, las pruebas y
`scripts/verify.py --mode offline --track workflows` con `uv run python`.
Abre el primer notebook con `.venv`. Distribuye el repo antes de la clase para que
las descargas no dependan de la conexión del aula. La ejecución inicial funciona sin red.

Si presentas el experimento live de clase 03, comprueba previamente acceso, modelo
disponible y saldo con la configuración existente. No necesitas que cada estudiante
tenga una cuenta. La clase no pierde sus ejercicios si la API no está disponible.
Nunca muestres ni pegues claves en una celda o en una captura.

## Facilitar cada clase

1. Lee la meta y pregunta por una situación parecida de su trabajo o barrio.
2. Pide una predicción antes de ejecutar; acepta respuesta oral, escrita o en pareja.
3. Ejecuta un bloque y pide localizar entrada, salida y decisión en el código.
4. Usa la falla intencional para distinguir el error de un fallo del entorno.
5. Respeta las dos pausas y el taller antes de mostrar la solución.
6. Cierra con el ticket de salida y anota una confusión para retomar al comenzar.

No asumas conocimientos por país, edad o profesión. Explica tienda/almacén/bodega
sin exigir un término regional. No atribuyas dificultades con Python a falta de
capacidad para IA. Quien termina primero revisa un caso nuevo de un compañero;
no introduce más frameworks al grupo.

## Evidencia de comprensión por clase

| Clase | Evidencia que debes observar | Error para conversar | Punto de reenganche |
|---|---|---|---|
| 00 | Distingue lista y diccionario; filtra arroz y café | `KeyError` por `precio` | Volver a imprimir las claves del primer producto |
| 01 | Calcula 3.60 de subtotal y 5.60 de total; rechaza `"dos"` | Inventar precio o aceptar cantidad negativa | Leer el pedido de tres campos |
| 02 | Dibuja dependencias; bloquea Sur | Redactar sin observar la cotización | Ejecutar de nuevo cotizar y luego redactar |
| 03 | Prueba tildes, palabras completas y ambigüedad | Confundir palabras clave con intención | Comparar un mensaje por área |
| 04 | Identifica tareas independientes y reúne datos | Tragar el error de un worker | Correr las consultas secuenciales |
| 05 | Corrige en intento 2 y se detiene con límite 1 | Considerar agotamiento como aprobación | Leer solo la lista de errores del primer intento |
| 06 | Retiro tiene un worker; entrega tiene dos | Ejecutar nombres inventados o planes incompletos | Escribir el plan como una lista antes de ejecutarlo |
| 07 | Distingue pendiente, aprobado, rechazo y sin stock | Pensar que aprobar crea stock o envía un mensaje | Observar `estado`, `mensaje` y `accion_externa` |

## Profundidad y alcance

Las reglas, plantillas y generadores guionados no son modelos de lenguaje. Repite
esa distinción al comienzo y al mostrar cada patrón. Un worker tampoco implica un
agente. El único experimento de LLM en esta ruta clasifica; no tiene ciclo de herramientas.

Los diagramas ayudan a razonar; las aserciones comprueban las salidas. Después de
la predicción, invita a explicar qué línea cambia una conducta. Para un grupo con
Python previo, amplía los casos de borde y pide escribir la regla sin helper.
Para uno sin Python, usa la clase 00 y conserva ejemplos pequeños.

La aprobación es una variable y cada llamada recalcula la propuesta. No la
presentes como autorización de producción ni como reanudación de un checkpoint.
Si el grupo necesita comprender un agente real, continúa después con los seis
notebooks existentes de LangGraph y Deep Agents, que implementan esos mecanismos.

## Proyecto y evaluación

Usa [la consigna y rúbrica](../proyectos/tienda_workflows/README.md). El reporte
automático comprueba diez contratos y comportamientos; la explicación demuestra
si la persona entiende sus decisiones. Pide además dos casos propios, uno que falle.
Admite diagramas, texto o explicación oral junto al código ejecutado.

La solución está separada para que cada persona pueda comparar después de intentar.
La plantilla ejecuta sin completar el trabajo, y su evaluación falla: no confundas
un script que termina con una solución que satisface la consigna.

No hay calendarios ni duraciones obligatorias. Ajusta el ritmo con los tickets
de salida y vuelve a comprobar el entorno cuando cambie `uv.lock`.
