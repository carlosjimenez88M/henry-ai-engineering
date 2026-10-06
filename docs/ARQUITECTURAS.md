# Arquitecturas de LangGraph · Mapa para decidir

Los patrones se pueden combinar. Empezar por los requisitos y no por el número de
agentes. Este mapa acompaña las explicaciones y ejemplos completos de los notebooks.

| Necesidad | Patrón | Forma de control | Estado que importa | Fallo a comprobar |
|---|---|---|---|---|
| Pasos conocidos en orden | Secuencia | Aristas fijas | Salidas intermedias | Un paso sin su entrada necesaria |
| Elegir un camino | Routing | Regla o modelo clasificador | Resultado de la decisión | Camino incorrecto o sin alternativa |
| Tareas independientes conocidas | Paralelismo fijo | Fan-out y barrera de unión | Contribuciones separadas | Pérdida de una rama |
| Cantidad de tareas variable | Orquestador–workers | Plan, Send y reunión | Plan y lista con reducer | Duplicados, tareas inválidas o fan-out excesivo |
| Elegir acciones según observaciones | Agente con herramientas | Modelo → herramientas → modelo | Mensajes y contador de llamadas | Loop sin respuesta, herramienta inválida |
| Mejorar un borrador contra un criterio | Evaluador–optimizador | Evaluar → corregir o terminar | Evidencia, borrador, intentos | Repetir sin progreso o entregar algo inválido |
| Delegaciones sucesivas coordinadas | Supervisor | Especialista devuelve al coordinador | Tarea, resultado y límite de delegación | Cuello de botella o circularidad |
| Transferir responsabilidad | Handoff | El receptor continúa el control | Motivo, contexto y permisos | Contexto perdido o transferencia circular |
| Tarea larga con entregables y subtareas separables | Deep agent | Coordinador con plan, archivos y subagentes | Todos, archivos virtuales, mensajes | Delegación inútil, archivo sin aprobación, citas inventadas |

## Secuencia frente a routing

La clase 2 usa los mismos nodos de búsqueda y respuesta en dos topologías. La primera
siempre sigue el orden. La segunda tiene una rama explícita para evidencia ausente.
El experimento mantiene las piezas y cambia solo el control: así se identifica qué
valor aporta la arquitectura y no se atribuye la mejora a otro modelo.

## Paralelismo frente a orquestador–workers

La clase 3 construye primero dos ramas fijas. Cada una escribe un campo diferente y
la barrera espera ambas. Después introduce un plan explícito de tamaño variable.
Send crea tareas con estado reducido; un reducer concatena sus contribuciones.
Un reducer de listas no ordena ni deduplica: son responsabilidades explícitas.

Los workers de estos ejemplos son búsquedas deterministas, no agentes autónomos.
Para convertirlos en especialistas con decisiones propias habría que definir
objetivos, herramientas, contexto y límites adicionales, además de evaluar el costo.

## Agente con herramientas

La demo ejecuta un grafo de mensajes, ToolNode y retorno al modelo. En live decide
el modelo; offline representa el mismo protocolo con un guion. El contador limita
llamadas y una salida explícita informa agotamiento. max_calls no mide dólares y
recursion_limit no reemplaza un presupuesto de uso del proveedor.

La versión prearmada usa `create_agent` de LangChain. Los límites se declaran como
middleware (`ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`) y se prueban con un
modelo guionado que nunca deja de pedir herramientas.

## Evaluador–optimizador y revisión humana

La clase 4 revisa si las fuentes citadas existen. Inyecta una cita falsa en el primer
borrador y demuestra corrección acotada. Si no hay evidencia o se agotan intentos,
el sistema escala y no presenta el borrador inválido como respuesta.

Luego interrupt pausa una revisión humana. Se reanuda con Command y el mismo
thread_id. Aprobación y rechazo se prueban en solicitudes distintas. Un checkpointer
es una capacidad de ejecución, no una arquitectura de inteligencia. El InMemorySaver
usado no ofrece persistencia después de reiniciar.

## Supervisor y handoff: alcance de esta entrega

La clase 3 construye un supervisor acotado: un nodo que devuelve `Command(goto, update)`,
especialistas que siempre devuelven el control al centro y un `max_delegaciones` que el
código impone aunque el modelo quiera seguir. Offline decide una regla; en live, GPT-6 con
salida estructurada sobre una lista cerrada de opciones. No es un supervisor
conversacional completo. El handoff se compara en diseño; no hay un handoff autónomo
implementado. No confundir el reparto de workers con un supervisor que replantea
decisiones, ni una aprobación humana con un handoff entre agentes.

## Deep agent

La clase 5 usa `create_deep_agent` (Deep Agents, sobre LangGraph). El coordinador
planifica con `write_todos`, delega con `task` en dos subagentes que pueden trabajar en
paralelo con contexto propio, escribe en un sistema de archivos virtual (estado del grafo)
y pide aprobación antes de `write_file` mediante `interrupt_on`. Es la combinación de
supervisor, revisión humana y agente con herramientas, empaquetada. En live, el
coordinador usa `gpt-6.1-sol` y los especialistas `gpt-6-luna`.

## Preguntas antes de elegir

1. ¿Conozco el orden de los pasos? Empezar por una secuencia.
2. ¿Necesito elegir solo un camino? Definir el contrato del router.
3. ¿Las tareas son independientes y comparten el mismo contexto permitido?
4. ¿Sé cuántas tareas habrá o necesito validar un plan variable?
5. ¿Necesito una decisión del modelo o una regla es suficiente?
6. ¿Tengo un criterio comprobable para repetir y una condición para terminar?
7. ¿Hay una acción que deba revisar una persona y puedo reanudar sin duplicarla?
8. ¿La tarea es larga, con subtareas separables y un entregable? Recién ahí, un deep agent.

Las respuestas deben aparecer en pruebas y estados observables, no solo en diagramas.
