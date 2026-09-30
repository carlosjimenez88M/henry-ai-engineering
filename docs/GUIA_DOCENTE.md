# Guía docente · AI Engineering aplicado, cuatro clases

## Crítica de la versión anterior y cambio de criterio

La versión anterior favorecía el arranque, pero escondía demasiada implementación
tras helpers y reducía varios ejercicios a cambiar una cadena. Faltaban actividades
que exigieran explicar decisiones, comparar alternativas y mostrar cómo falla una
decisión de arquitectura.

Esta revisión profundiza construyendo, contrastando y comprobando. Los notebooks
incluyen las explicaciones necesarias: la guía orienta la facilitación, pero no
contiene una mitad de la clase que el estudiante deba buscar en otro archivo.

No se exige aprender todas las arquitecturas al mismo nivel. Secuencia, routing,
paralelismo, workers y revisión se construyen; el agente se demuestra e inspecciona;
supervisor y handoff se comparan como contratos de diseño. Esa diferencia evita
simular cobertura exhaustiva para un público principiante.

## Acompañamiento y carga cognitiva

Cada sesión incluye dos pausas. Los notebooks presentan un recorrido temático
con actividades concretas. Los bloques de práctica son actividad, no monólogo.
Antes de cada celda: anticipar. Después: observar, explicar y cambiar una sola cosa.

Las dificultades de atención no implican menor capacidad. Estas adaptaciones no
son una intervención clínica. Consultar preferencias, permitir chat/oralidad/texto,
no forzar exposición sorpresiva y no premiar velocidad. Dar espacio para pensar
antes de pedir respuestas. Usar tamaño de letra legible y pocas ventanas abiertas.

El código está preparado: no se copia a la velocidad del docente. En cada bloque
se destacan pocas líneas y su relación con un dibujo o un dato. Hay pistas y puntos
de reenganche antes de las pausas. Las soluciones se revisan después del intento,
comparando una diferencia concreta. No improvisar una página en blanco de 50 líneas.

Preparar el entorno antes, con ayuda. Offline permite participar sin claves ni
cuentas. El docente puede mostrar live explicando claramente cuál usa un modelo.
Si alguien no puede ejecutar, trabaja con una pareja o documenta sus predicciones;
no se califica la instalación como comprensión de arquitectura.

## Clase 1 · Herramienta confiable

**Tres anclas:** contrato, ejecución y evidencia.
**Producto:** consulta filtrada con IDs, límite y rechazo de entrada inválida.

- **Problema y resultado:** plantear el pedido y mostrar el resultado terminado. Preguntar qué haría un
  sistema si no encuentra una ficha; no empezar enumerando librerías.
- **Búsqueda mínima:** construir buscar_minimo. Seguir una fila a mano, cambiar el tema y mostrar
  el fallo por tilde. Pedir una condición de filtro antes de enseñar la solución.
- **Pausa.**
- **Contrato y búsqueda:** usar Pydantic y recorrer validar → normalizar → filtrar → puntuar → limitar.
  Leer el contrato definido en el notebook, comprobar top_k y ausencia de resultados.
  Separar score de probabilidad.
- **Llamada a la herramienta:** observar un tool call. Está forzado en esta demo; explicar que eso no prueba
  autonomía. El decorador y el wrapper se construyen en la celda, usando el contrato
  anterior. Señalar nombre, argumentos, ejecución Python y observación enlazada.
- **Pausa.**
- **Taller del DJ:** buscar una canción de investigación con filtros, mostrar
  MUS-02, probar tres entradas inválidas y justificar si hace falta un LLM.
- **Cierre:** mostrar un contrato válido y otro inválido; explicar qué hace cada actor.

**Error útil:** la búsqueda mínima sin tildes. No arreglar todo de inmediato; pedir
una hipótesis y una prueba. **Profundidad:** la herramienta gana capacidades porque
su contrato se conserva, no porque se añadan acciones arbitrarias.

## Clase 2 · RAG y primeros grafos

**Tres anclas:** evidencia, estado y transición.
**Producto:** secuencia y routing construidos con los mismos nodos.

- **Recuperación y generación:** seguir una ficha de Batman hasta una cita y separar responsabilidades.
- **Prompt y contrato de salida:** examinar prompt y schema. Crear una respuesta con ID inventado: Pydantic
  acepta la forma, pero la validación de referencias la rechaza. La plantilla visible
  se pasa a compose y controla el prompt en live. Offline comprueba el contrato de
  plantilla, pero no genera con un modelo. Discutir fidelidad.
- **Pausa.**
- **Estado y secuencia:** representar el estado y construir buscar → responder. Explicar que TypedDict
  anota tipos, no hace la validación de runtime de Pydantic. Cambiar una colección.
- **Routing:** agregar condición y nodo de abstención. Dibujar las dos rutas.
  Probar “Batman vacuna marciana”: colección conocida no significa tema respaldado.
- **Pausa.**
- **Taller de la vecindad:** consultas de la vecindad, caso vacío y streaming de updates. Diagnosticar
  el filtro equivocado antes de proponer cambiar el modelo.
- **Cierre:** justificar cuándo la secuencia es suficiente y cuándo conviene routing.

**Error útil:** schema válido con contenido incorrecto. **Profundidad:** distinguir
pruebas de recuperación, referencias y fidelidad. No enseñar álgebra de embeddings
además de todo esto: aquí la búsqueda lexical ya ofrece evidencia inspeccionable.

## Clase 3 · Arquitecturas y equipos

**Tres anclas:** dividir trabajo, combinar estado y decidir quién controla.
**Producto:** paralelo fijo y orquestador con Send; demo acotada de tool agent.

- **Elección de arquitectura:** elegir patrón según tres tareas y contrastar el mapa. Los ocho patrones no
  son ocho implementaciones que deban dominar de memoria ese día.
- **Paralelismo fijo:** construir dos workers, campos separados y barrera de unión. Comprobar dos
  fuentes. No prometer velocidad por una demo local de microsegundos.
- **Pausa.**
- **Orquestador y workers:** convertir número fijo en plan variable. Mostrar por qué parts necesita un
  reducer; concatenar no equivale a deduplicar. Validar el plan antes de repartir.
- **Agente con herramientas:** seguir modelo → herramienta → modelo. Nombrar límites, observaciones y modo.
- **Pausa.**
- **Taller del equipo:** agregar música al plan sin copiar nodos, duplicar una colección y rechazar
  otra desconocida. La celda del reto parte de dos colecciones; la ampliación completa
  aparece después, en la solución. Comparar supervisor y handoff con sus responsabilidades.
- **Cierre:** defender una arquitectura y un caso de error.

**Error útil:** creer que dos funciones paralelas son dos agentes. **Profundidad:**
pedir el requisito que justifica cada patrón. Si el grupo se atasca con la sintaxis
Annotated, mostrar la concatenación concreta y continuar con el código preparado.
No quitar pausas ni convertir la demo de agente en una nueva implementación larga.

## Clase 4 · Evaluación y revisión humana

**Tres anclas:** criterio, límite y responsabilidad.
**Producto:** corrección con máximo de intentos, aprobación/rechazo y reporte.

- **Criterio de evaluación:** mostrar una referencia inventada y definir qué puede detectar el evaluador.
- **Ciclo de corrección:** construir los nodos y el ciclo. La primera falla es inyectada de forma
  explícita; no se atribuye falsamente a una equivocación espontánea del modelo.
  Cambiar MAX_INTENTOS de dos a uno permite observar agotamiento y escalación.
- **Pausa.**
- **Revisión humana:** checkpoint, interrupción y reanudación con el mismo thread_id. La aprobación
  de la celda es una simulación; antes de ejecutarla una persona explica su revisión.
- **Decisiones y aislamiento:** enviar una decisión inválida, corregirla con un rechazo y verificar aislamiento.
  Una entrada inválida debe conservar la posibilidad de reanudar. Discutir reinicios.
- **Pausa.**
- **Taller de evaluación:** ejecutar cinco casos y comprobar igualdad de IDs, incluido caso vacío.
  El reto trae cuatro casos; cada pareja agrega el musical antes de ver la solución
  de cinco. Probar filtro incorrecto y falta de evidencia. Guardar el reporte local.
- **Cierre:** defensa del proyecto; agregar dos casos más queda como trabajo posterior.

**Error útil:** seguir corrigiendo sin evidencia o entregar el último borrador porque
se agotó el límite. **Profundidad:** el criterio de IDs no prueba verdad semántica.
Un reporte de pruebas tampoco sustituye autenticación o almacenamiento durable.

## Reenganche y ajuste del ritmo

Si alguien pierde el hilo, preguntar qué fue lo último que funcionó y ubicar el
punto de reenganche. Reejecutar una definición y un ejemplo corto. Usar una solución
es válido si después puede explicar una diferencia. Mostrar el título actual en
pantalla. No pedir releer todo el notebook mientras el resto avanza.

Si varias personas no pueden explicar una salida, reducir una variante del taller
y conservar el bloque de corrección. Dar un rol de revisor a quien terminó antes:
debe proponer un caso de prueba, no introducir más frameworks al resto del grupo.

El recorrido se ajusta con evidencia del aula. Registrar en una primera cohorte
las dudas y los puntos donde se necesita más acompañamiento; ajustar ejemplos
manteniendo objetivos, dos pausas y cierre.

## Evaluación del proyecto

| Dimensión | Puntos | Evidencia |
|---|---:|---|
| Contrato de herramienta | 25 | Entrada inválida rechazada, filtros, límite y IDs |
| Arquitectura justificada | 30 | Grafo ejecutable, decisión comparada y prueba de una falla |
| Evidencia y abstención | 20 | Fuentes recuperadas, referencias válidas y ausencia de evidencia |
| Evaluación y responsabilidad | 25 | Reporte, ciclo acotado, aprobación y rechazo |

Aceptar explicación oral, escrita o diagramas, junto con ejecuciones. Permitir
corregir después de feedback. No premiar cantidad de agentes ni exigir exposición
de razonamiento privado de un modelo. El estudiante debe mostrar comportamiento
observable y reconocer un límite de su propia solución.
