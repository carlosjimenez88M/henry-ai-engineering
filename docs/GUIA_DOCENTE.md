# Guía docente · Ruta avanzada (8 clases)

Esta guía corresponde al **recorrido ampliado** de LangGraph y Deep Agents.
Para personas que recién comienzan, usa primero la
[guía de las diez clases iniciales](GUIA_DOCENTE_WORKFLOWS.md) y su
[ruta de workflows](../clases/agentic_workflows/README.md).

Diseño, convenciones y API de apoyo: [DISENO_RUTA_AVANZADA.md](DISENO_RUTA_AVANZADA.md).
Proyecto acumulativo: [Asistente del Archivo](../proyectos/asistente_archivo/README.md).

## Antes de la primera clase

- Instalación con apoyo, siguiendo [INSTALACION.md](INSTALACION.md). Las dos causas más
  comunes de "los notebooks no corren en VS Code" son que falta la extensión **Jupyter** o que
  el kernel no es `.venv`. `uv run python scripts/doctor.py` detecta ambas.
- Quien no completó la ruta 1 necesita, como mínimo, su clase 00 de Python.
- Docente con clave: `uv run python scripts/doctor.py --live` y
  `uv run python scripts/verify.py --mode live --track advanced` antes de cada cohorte.
  Revisar precios de GPT-6 en `src/henry_agents/config.py` (única fuente).

## Cómo se usa cada clase

- **Ritmo:** 🔮 predecir → ▶️ ejecutar → 🔍 observar. Pedir la predicción en voz alta o por
  escrito antes de ejecutar; no saltear ese paso: es donde ocurre el aprendizaje.
- **Ejercicios ✏️:** la celda incompleta corre sin error y la revisión muestra ✅ o una pista.
  La solución se ve con `ver_solucion("...")`. Pedir que la miren **después** de intentar y que
  expliquen una diferencia con su versión.
- **Dos pausas ☕** por clase. Tras cada pausa hay un bloque nuevo, no una continuación a medias.
- **Offline es la experiencia de estudiantes.** El cerebro `reglas-offline` decide con reglas
  visibles; todo lo demás es real. Decirlo con claridad la primera vez y cuando aparezca.
- **Live es una demostración docente.** Mostrar 1–2 momentos por clase con GPT-6 (sugeridos
  abajo) y comparar con offline: qué cambió, qué se mantuvo y por qué las comprobaciones
  de valores exactos solo corren offline.
- Si alguien se pierde: preguntar qué fue lo último que funcionó y reejecutar las celdas desde
  el título de esa sección, en orden. Los cerebros offline no guardan estado, así que
  reejecutar no cambia los resultados.

## Clase 0 · Mundo agéntico

**Anclas:** el modelo propone, el programa ejecuta; workflow antes que agente; el costo importa.
**Momento live:** la primera llamada y el agente con un pedido de la clase.
**Error útil:** creer que el *system prompt* es una barrera de seguridad (se retoma en la 4).

## Clase 1 · Herramientas y bucle

**Anclas:** contrato, error de entrada ≠ falta de evidencia, el bucle cabe en 15 líneas.
**Momento live:** el bucle a mano con GPT-6 Luna; mirar cuántas vueltas da.
**Error útil:** `argumentos_invalidos` muestra cómo el error vuelve al modelo y este se
corrige; `bucle` muestra por qué la condición de parada no es opcional.

## Clase 2 · RAG con evidencia

**Anclas:** recuperar → aumentar → generar; forma válida ≠ contenido correcto; palabras vs significado.
**Momento live:** la salida estructurada real y la búsqueda con embeddings reales
(`OPENAI_EMBEDDING_MODEL`, por defecto `text-embedding-3-large`; medir consumo).
**Error útil:** una cita inventada que Pydantic acepta y la validación con conjuntos rechaza.
La comparación final mantiene corpus y `k=1` y muestra el costo de recuperar otra
fuente. Para profundizar, utiliza las clases iniciales 08/09 y la
[guía de RAG y Agentic RAG](RAG_Y_AGENTIC_RAG.md).

## Clase 3 · Workflows en LangGraph

**Anclas:** estado, nodos que devuelven cambios, aristas; el patrón ya lo conocen de la ruta 1.
**Momento live:** el nodo responder con GPT-6; el grafo y sus rutas no cambian.
**Error útil:** sin reducer, dos workers que escriben el mismo campo chocan.

## Clase 4 · Agentes confiables

**Anclas:** límites como middleware, memoria = mensajes del hilo, salida estructurada,
defensa en profundidad.
**Momento live:** ver un agente real en streaming; probar la reseña envenenada con GPT-6 y
discutir que resistir hoy no garantiza resistir siempre: por eso la aprobación humana.
**Error útil:** el laboratorio de fallas; cada falla con su defensa en código.

## Clase 5 · Multiagente

**Anclas:** dividir solo si hay una razón; supervisor (vuelve al centro), agentes como
herramientas (contexto limpio), handoff (no vuelve).
**Momento live:** el supervisor decidiendo con salida estructurada sobre una lista cerrada.
**Error útil:** bajar `max_delegaciones` y ver un informe incompleto pero honesto.

## Clase 6 · Evaluación y control humano

**Anclas:** criterio verificable, límite de intentos en el estado, pausa y reanudación con el
mismo `thread_id`, juez calibrado contra etiquetas humanas, costo medido.
**Momento live:** el juez GPT-6 sobre los tres casos etiquetados; comparar con el juez de reglas.
**Error útil:** una cita real con una afirmación falsa: pasa la validación de IDs y la detecta el juez.

## Clase 7 · Deep Agents

**Anclas:** plan, archivos virtuales, subagentes con contexto propio y límite propio, permiso
antes de escribir o editar.
**Momento live:** el equipo completo con Sol coordinando y Luna en especialistas, en streaming.
Mostrar el costo al final y discutir cuándo un workflow habría bastado.
**Error útil:** el especialista atascado que su propio límite detiene.

## Evaluación del proyecto

Usar la rúbrica de `proyectos/asistente_archivo/README.md`. Aceptar explicación oral, escrita
o con diagramas, junto con ejecuciones. Permitir corregir después del feedback. No premiar la
cantidad de agentes ni la velocidad: premiar explicar, comprobar y reconocer un límite propio.

## Ajuste con evidencia del aula

Registrar en la primera cohorte dónde se detienen las personas (celda y pregunta), cuánto
tarda cada bloque y qué ejercicios se resuelven sin mirar la solución. Ajustar ejemplos y
pistas manteniendo objetivos, pausas y cierre.
