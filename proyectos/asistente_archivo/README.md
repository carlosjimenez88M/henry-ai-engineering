# Proyecto · El Asistente del Archivo

**Ruta avanzada.** Un proyecto que crece clase a clase: al final tienes un equipo de agentes
que prepara actividades para el Centro Cultural, con fuentes verificables y aprobación humana.

## El encargo

El Centro Cultural del barrio guarda un archivo de doce fichas ficticias (Batman, los Cuatro
Fantásticos, El Chavo y canciones instrumentales inventadas). Las y los docentes del centro
piden un asistente que:

1. Encuentre fichas sobre un tema y una colección, **citando sus IDs**.
2. **No invente**: si no hay evidencia, lo dice.
3. Prepare una **actividad de clase** (consigna + fichas + música de ambiente).
4. **No publique ni guarde nada sin que una persona lo apruebe.**
5. Cueste poco y se pueda **comprobar** que funciona.

## Un paso por clase

| Clase | Paso del proyecto | Evidencia que guardas |
|---|---|---|
| 0 | Leer el encargo y ubicar cada requisito en la escalera de autonomía | Tabla requisito → escalón, con una razón |
| 1 | La herramienta `buscar_archivo` y un bucle de agente a mano | Una llamada válida, una rechazada y la línea de tiempo del bucle |
| 2 | Responder solo con evidencia y validar las citas | Una respuesta con fuentes válidas y una abstención |
| 3 | El flujo fijo: buscar → responder o abstenerse, en LangGraph | El grafo dibujado y las dos rutas probadas |
| 4 | El asistente como agente: límites, memoria, salida estructurada y defensa ante inyección | Una conversación de dos turnos y un ataque bloqueado |
| 5 | Un equipo: supervisor con especialistas | La bitácora del supervisor y un límite de delegaciones probado |
| 6 | Evaluación y aprobación humana | Reporte con casos, juez y costo; una aprobación y un rechazo |
| 7 | Entrega final: el equipo profundo | `/actividad.md` aprobado, citas verificadas y la defensa de la arquitectura |

## Entrega final y rúbrica

Presenta (oral, escrita o con diagramas, junto con ejecuciones):

| Dimensión | Puntos | Qué se mira |
|---|---:|---|
| Herramienta y contrato | 15 | Entradas inválidas rechazadas, límites, IDs |
| Evidencia y citas | 20 | Fuentes válidas, abstención, verificación con código |
| Arquitectura justificada | 25 | Por qué este nivel de autonomía y no uno más simple |
| Seguridad y control | 20 | Límites de llamadas, aprobación humana, defensa ante inyección |
| Evaluación y costo | 20 | Casos de prueba, resultado del juez y costo medido o estimado |

No se premia la cantidad de agentes ni la velocidad. Se premia explicar y comprobar.
