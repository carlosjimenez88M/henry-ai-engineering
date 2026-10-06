# Arquitecturas de agentes · Mapa para decidir (ruta avanzada)

Los patrones se combinan. Empieza por los requisitos, no por la cantidad de agentes.

| Necesidad | Patrón | Quién decide el siguiente paso | Fallo a comprobar | Clase |
|---|---|---|---|---|
| Pasos conocidos en orden | Secuencia | El código (aristas fijas) | Un paso sin la entrada que necesita | 3 |
| Elegir un camino | Routing | Una regla o un clasificador | Ruta incorrecta o sin salida para "no sé" | 3 |
| Tareas independientes conocidas | Paralelo fijo | El grafo; una barrera une | Perder una rama | 3 |
| Cantidad de tareas variable | Orquestador–workers (`Send` + reducer) | Un plan validado | Duplicados, tareas inválidas, fan-out sin límite | 3 |
| Elegir acciones según lo observado | Agente con herramientas | El modelo, con límites | Bucle, argumentos inválidos, no usar la herramienta | 1 y 4 |
| Delegaciones sucesivas | Supervisor (`Command`) | Un coordinador que vuelve a decidir | Cuello de botella, delegación sin fin | 5 |
| Especialistas con contexto limpio | Agentes como herramientas | El agente coordinador | Especialistas que devuelven demasiado | 5 |
| Transferir la responsabilidad | Handoff (`Command(goto=...)`) | Quien recibe el control | Perder contexto o motivo | 5 |
| Mejorar un borrador con un criterio | Evaluador–optimizador | El evaluador y un límite de intentos | Repetir sin progreso, entregar algo inválido | 6 |
| Acción con efecto visible | Revisión humana (`interrupt`, `interrupt_on`) | Una persona | Reanudar con el hilo equivocado, aprobar sin leer | 4, 6 y 7 |
| Tarea larga con entregables | Deep agent | Coordinador con plan, archivos y subagentes | Delegación inútil, archivo sin aprobación, citas inventadas | 7 |

## Preguntas antes de elegir

1. ¿Conozco el orden de los pasos? Empieza por una secuencia.
2. ¿Solo hay que elegir un camino? Define el contrato del router y su salida "no sé".
3. ¿Las tareas son independientes? Paralelo; si su cantidad varía, `Send` con plan validado.
4. ¿Necesito que el modelo decida qué hacer según lo que observa? Recién ahí, un agente.
5. ¿Hay especialidades con contexto distinto? Supervisor o agentes como herramientas.
6. ¿Tengo un criterio comprobable para repetir y una condición para terminar?
7. ¿Alguna acción debe aprobarla una persona? Pausa antes de la acción, no después.
8. ¿La tarea es larga, con subtareas separables y un entregable? Entonces, un deep agent.

Las respuestas deben aparecer en pruebas y estados observables, no solo en diagramas.
Un worker no es necesariamente un agente, y usar LangGraph no vuelve autónomo a un sistema.
