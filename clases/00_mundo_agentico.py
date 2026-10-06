# %% [markdown]
# # Clase 0 · Bienvenida al mundo agéntico
#
# Esta clase no exige saber programar. Vamos a entender **qué es un agente de IA**,
# en qué se diferencia de un chatbot y de un programa común, y a dejar el entorno
# funcionando en VS Code. Todo lo que hagamos después se apoya en estas ideas.
#
# **Al terminar vas a poder:**
# - Explicar con tus palabras qué es un LLM, una herramienta, un agente y un workflow.
# - Ubicar un problema en la "escalera de autonomía" y elegir el nivel más simple que alcanza.
# - Elegir un modelo de OpenAI según tarea y costo, y estimar cuánto cuesta una llamada.
# - Ver un agente real en acción y leer, paso a paso, qué decidió y qué ejecutó.
#
# **Recorrido de la clase**
#
# - Comprobar que el notebook usa el entorno correcto
# - Cinco palabras clave con analogías: LLM, token, prompt, herramienta, agente
# - Escalera de autonomía: de una pregunta suelta a un equipo de agentes
# - Pausa
# - Los modelos GPT-6 de OpenAI: cuál usar, cuánto cuesta y qué es el "esfuerzo de razonamiento"
# - Primera llamada a un modelo (o a su guion offline)
# - Un agente completo en tres líneas, y su línea de tiempo
# - Pausa
# - Taller: ¿workflow o agente? Clasificar cinco situaciones
# - Glosario y ticket de salida
#
# ### Cómo usar este notebook en VS Code
# 1. Arriba a la derecha, en **Select Kernel / Seleccionar kernel**, elegí el entorno
#    `.venv` de esta carpeta (aparece como `.venv (Python 3.13)`). Si no aparece, mirá
#    `docs/INSTALACION.md`.
# 2. Ejecutá una celda con **Shift + Enter**. El número entre corchetes indica el orden.
# 3. Si algo falla, no te asustes: leé la **última línea** del error. Casi siempre dice qué pasó.
# 4. Si cambiás el archivo `.env`, reiniciá el kernel (botón **Restart**).

# %% [markdown]
# ## Paso 1 · ¿Estamos en el entorno correcto?
# Un **entorno virtual** (`.venv`) es una carpeta con su propio Python y sus propias
# librerías, separada del resto del computador. Así todas las personas del curso usan
# exactamente las mismas versiones. Si el notebook no usa `.venv`, aparecen errores
# como `ModuleNotFoundError: No module named 'henry_agents'`.
#
# **Predicción:** ¿qué ruta esperás ver en "Python que ejecuta este notebook"?

# %%
import importlib.metadata
import sys

print("Versión de Python:", sys.version.split()[0])
print("Python que ejecuta este notebook:", sys.executable)
if ".venv" in sys.executable:
    print("✅ Estás usando el entorno .venv del curso.")
else:
    print("⚠️ Este kernel no es el .venv del curso. Cambialo en 'Select Kernel'.")
for paquete in ["langchain", "langgraph", "langchain-openai", "deepagents"]:
    print(f"   {paquete:18} {importlib.metadata.version(paquete)}")

# %% [markdown]
# Ahora cargamos la configuración del curso. Hay **dos modos**:
#
# | Modo | Qué usa | Costo | Para quién |
# |---|---|---|---|
# | `offline` (por defecto) | Herramientas y grafos reales; el "cerebro" es un **guion** escrito de antemano | Gratis | Todas las personas |
# | `live` | Modelos reales de OpenAI | Consume saldo de la API | Docente o quien tenga clave |
#
# Offline no es "de mentira": la búsqueda, los grafos, los límites y las aprobaciones
# son los mismos. Solo cambia quién toma las decisiones: un guion en lugar del modelo.

# %%
from henry_agents.config import MODELOS_OPENAI, configure, model_name

MODE = configure()
print("Modo:", MODE)
print("Modelo para clases 0–4:", model_name())
print("Modelo para deep agents (clase 5):", model_name("agent"))

# %% [markdown]
# ## Cinco palabras clave, con analogías
#
# **1. LLM (Large Language Model, modelo de lenguaje).** Un programa entrenado con
# muchísimo texto que, dado un texto de entrada, **predice cómo continuarlo**. Es como
# un autocompletado extremadamente bueno. Consecuencias importantes:
# - Escribe con fluidez, pero **puede inventar** datos con total seguridad ("alucinar").
# - No sabe qué hay en *tus* archivos ni en Internet, salvo que se lo des o le des una herramienta.
# - No "recuerda" conversaciones anteriores: cada llamada recibe todo el contexto de nuevo.
#
# **2. Token.** La unidad en que el modelo lee y escribe: pedazos de palabra. En
# español, 100 palabras ≈ 130–150 tokens. **Se paga por token**, de entrada y de salida.
#
# **3. Prompt.** Lo que le enviamos al modelo. Suele tener dos partes:
# - *System*: las reglas del juego ("Respondé solo con la evidencia dada").
# - *Human/User*: el pedido concreto ("¿Qué ficha habla de investigación?").
#
# **4. Herramienta (tool).** Una función de **nuestro programa** que el modelo puede
# *pedir* usar: buscar en un catálogo, consultar una base de datos, escribir un archivo.
# Regla de oro: **el modelo propone, el programa ejecuta.** El modelo nunca toca nada
# directamente; nosotros decidimos qué herramientas existen y qué límites tienen.
#
# **5. Agente.** Un LLM que trabaja **en un bucle**: piensa qué hacer, pide una
# herramienta, mira el resultado y decide si sigue o responde.
#
# ```text
#        ┌──────────────────────────────────────────────┐
#        ▼                                              │
#  pedido → MODELO decide ──¿necesita una herramienta?── sí → PROGRAMA ejecuta → observación
#                 │
#                 └── no → respuesta final
# ```
#
# **Analogía:** un *workflow* es una **receta** (pasos fijos, en orden). Un *agente* es
# un **cocinero** que mira la heladera, decide qué cocinar y prueba antes de servir.
# El cocinero es más flexible, pero también más caro, más lento y más difícil de predecir.
# Por eso, en ingeniería, **usamos la receta cuando alcanza** y el cocinero cuando hace falta.

# %% [markdown]
# ## La escalera de autonomía
# No todo problema necesita un agente. Subimos un escalón solo cuando el anterior no alcanza.
#
# | Escalón | Qué es | Quién decide el próximo paso | Ejemplo | Clase |
# |---|---|---|---|---|
# | 0 · Llamada única | Un prompt, una respuesta | Nadie: hay un solo paso | Resumir un texto | 0 |
# | 1 · Herramienta | Función con contrato que el modelo puede pedir | El programa valida y ejecuta | Buscar fichas con filtros | 1 |
# | 2 · Workflow | Pasos fijos o con reglas (secuencia, routing) | El código | Buscar → responder o abstenerse | 2 |
# | 3 · Orquestación | Varias tareas en paralelo, plan y reparto | El código y un plan | Consultar tres colecciones a la vez | 3 |
# | 4 · Agente | Bucle modelo → herramienta → modelo | **El modelo**, con límites | Investigar hasta tener evidencia | 3 |
# | 5 · Revisión | Ciclos de corrección y aprobación humana | Un evaluador y una persona | Aprobar antes de entregar | 4 |
# | 6 · Deep agent | Agente que planifica, usa archivos y delega en subagentes | El modelo coordinador | Preparar una actividad completa | 5 |
#
# **Orquestación** significa organizar quién hace qué, en qué orden y cómo se juntan
# los resultados. **LangGraph** es la librería que usamos para orquestar: dibujamos el
# trabajo como un **grafo** (nodos = pasos, aristas = flechas entre pasos).
# **Deep Agents** es una librería construida *sobre* LangGraph que trae un agente
# "con oficio": planifica, toma notas en archivos y delega en ayudantes.
#
# **Pregunta para discutir:** ¿por qué no empezar siempre por el escalón 6, si es el
# más potente? Pensá en costo, velocidad, errores difíciles de rastrear y pruebas.
#
# ## Pausa

# %% [markdown]
# ## Los modelos de OpenAI que usamos (GPT-6)
# OpenAI ofrece una familia con tres niveles. Todos entienden herramientas, devuelven
# salidas estructuradas y tienen una ventana de contexto de alrededor de un millón de tokens.
# Los precios son por **millón de tokens** (entrada / salida), consultados en octubre de 2026;
# cambian con el tiempo, así que el docente los revisa antes de cada cohorte.
#
# | Modelo | Para qué | Entrada | Salida |
# |---|---|---|---|
# | `gpt-6-luna` | Tareas acotadas y muchas llamadas: clasificar, extraer, agentes simples | $0.10 | $0.50 |
# | `gpt-6.1-sol` | Agentes con varias herramientas, coordinadores, deep agents | $2 | $10 |
# | `gpt-6-astra` | Problemas largos y difíciles; usar con criterio | $10 | $50 |
#
# **Esfuerzo de razonamiento** (`reasoning_effort`): estos modelos pueden "pensar"
# antes de responder. Más esfuerzo = mejores respuestas en problemas difíciles, pero
# más tokens, más tiempo y más costo. Valores: `none` (solo Luna), `low`, `medium`,
# `high`, `xhigh`, `max`. El curso usa `low` por defecto; se cambia en `.env`.
#
# **Regla práctica de ingeniería:** empezá con el modelo más económico que pase tus
# pruebas. Subí de nivel solo donde haya evidencia de que hace falta. En la clase 5
# combinamos: Sol coordina y Luna hace las tareas de los especialistas.

# %%
for nombre, uso in MODELOS_OPENAI.items():
    print(f"{nombre:12} → {uso}")

PRECIOS = {  # USD por millón de tokens: (entrada, salida)
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
}


def costo_estimado(modelo, tokens_entrada, tokens_salida):
    entrada, salida = PRECIOS[modelo]
    return (tokens_entrada * entrada + tokens_salida * salida) / 1_000_000


# Un agente típico de clase: ~3.000 tokens de entrada y ~500 de salida por llamada.
for modelo in PRECIOS:
    print(f"{modelo:12} 1 llamada ≈ ${costo_estimado(modelo, 3000, 500):.5f}")

# %% [markdown]
# **Ejercicio rápido:** un curso de 30 personas ejecuta un agente que hace 5 llamadas
# por consulta, 20 consultas cada una. ¿Cuánto cuesta con Luna y con Astra?
# Escribí tu estimación antes de ejecutar. La diferencia es la razón por la que
# **elegir el modelo es una decisión de ingeniería**, no de gusto.

# %%
llamadas_totales = 30 * 20 * 5
for modelo in ["gpt-6-luna", "gpt-6-astra"]:
    total = llamadas_totales * costo_estimado(modelo, 3000, 500)
    print(f"{modelo:12} {llamadas_totales} llamadas ≈ ${total:.2f}")

# %% [markdown]
# ## Primera llamada a un modelo
# Enviamos dos mensajes: reglas (*system*) y pedido (*human*). En modo `live` responde
# GPT-6 de verdad. En `offline` responde un **guion**: un modelo falso que devuelve un
# texto preparado. Lo usamos para practicar sin costo, y lo decimos con claridad.
#
# `respuesta.text` es el texto visible. En `live`, `usage_metadata` muestra cuántos
# tokens se usaron: con eso se calcula el costo real.

# %%
from langchain_core.messages import AIMessage

from henry_agents.agentic import ModeloGuionado
from henry_agents.config import chat_model

mensajes = [
    ("system", "Sos un profesor paciente. Respondé en dos oraciones, sin tecnicismos."),
    ("human", "¿Qué es un agente de IA?"),
]
if MODE == "live":
    modelo = chat_model()
else:
    modelo = ModeloGuionado(
        pasos=[
            AIMessage(
                content="(Guion offline) Un agente de IA es un programa que usa un modelo de "
                "lenguaje para decidir qué acciones tomar, como buscar información, y repite "
                "hasta cumplir un objetivo. Siempre trabaja con herramientas y límites que "
                "definen las personas que lo construyen."
            )
        ]
    )
respuesta = modelo.invoke(mensajes)
print(respuesta.text)
print("Tokens usados:", respuesta.usage_metadata or "no aplica en modo offline")

# %% [markdown]
# **Experimento (live):** cambiá el mensaje *system* por "Respondé como un pirata".
# El pedido es el mismo, pero el comportamiento cambia: el *system prompt* es una
# herramienta de diseño poderosa. Ojo: **no es una barrera de seguridad**. Los límites
# reales (qué herramientas existen, cuántas llamadas, qué se aprueba) van en el código.
#
# ## Un agente completo, en tres líneas
# `create_agent` (de LangChain) arma el bucle modelo → herramienta → modelo por nosotros.
# Le damos un modelo, una herramienta (el buscador del catálogo del curso) y límites.
# En la clase 1 vas a construir esa herramienta desde cero; hoy solo la usamos.
#
# **Predicción:** ¿cuántas veces va a pedir el agente la herramienta para "Buscá
# fichas de investigación de Batman"? ¿Qué IDs esperás ver?

# %%
from langchain_core.messages import HumanMessage

from henry_agents.agentic import build_prebuilt_agent, linea_de_tiempo, mostrar_grafo

agente = build_prebuilt_agent(MODE, max_model_calls=4, max_tool_calls=3)
resultado = agente.invoke({"messages": [HumanMessage("Buscá fichas de investigación de Batman")]})
linea_de_tiempo(resultado["messages"])

# %% [markdown]
# Leé la línea de tiempo de arriba hacia abajo:
# - 👤 la persona pide algo;
# - 🤖 el modelo **propone** llamar `buscar_archivo` con ciertos argumentos;
# - 🔧 **nuestro programa** ejecuta la búsqueda y devuelve una observación con IDs;
# - 🤖 el modelo responde usando esa observación.
#
# El agente también **es un grafo**. Dibujémoslo: los nodos con "Middleware" son los
# controles de límites que agregamos. Sin Internet se dibuja en texto; está bien así.

# %%
mostrar_grafo(agente)
assert any(m.type == "tool" for m in resultado["messages"]), "El agente debía usar la herramienta"
print("✅ El agente consultó la herramienta antes de responder.")

# %% [markdown]
# **Punto de reenganche:** podés señalar en la línea de tiempo qué decidió el modelo y
# qué ejecutó el programa. Si eso está claro, ya entendiste lo más importante del curso.
#
# ## Pausa
#
# ## Taller: ¿workflow o agente?
# Para cada situación, elegí `"workflow"` (pasos conocidos de antemano) o `"agente"`
# (el próximo paso depende de lo que se vaya encontrando). Completá el diccionario.
# No hay que programar: solo cambiar el texto entre comillas.
#
# 1. Cada mañana, resumir los correos nuevos y mandarlos a un canal.
# 2. Un asistente que investiga una pregunta abierta y decide qué buscar según lo que encuentra.
# 3. Clasificar tickets de soporte en tres categorías y enviarlos al área correspondiente.
# 4. Ayudar a depurar un error de programación desconocido, probando hipótesis.
# 5. Traducir un documento y revisar que no falten párrafos.

# %%
mis_respuestas = {
    1: "workflow",
    2: "workflow",  # ¿Seguro? Pensá si los pasos se conocen de antemano.
    3: "workflow",
    4: "workflow",
    5: "workflow",
}

# %% [markdown]
# ## Solución comentada
# Compará con tu respuesta. Lo importante es la **razón**, no acertar la palabra.

# %%
solucion = {
    1: ("workflow", "Pasos fijos: leer → resumir → enviar. Un agente agregaría costo sin beneficio."),
    2: ("agente", "No se sabe de antemano qué buscar; cada resultado cambia el siguiente paso."),
    3: ("workflow", "Es routing: una clasificación y una regla. Un LLM puede clasificar, sin bucle."),
    4: ("agente", "Hay que probar hipótesis y reaccionar a lo que se observa."),
    5: ("workflow", "Traducir y luego comprobar: evaluador con criterio claro (clase 4)."),
}
for numero, (tipo, razon) in solucion.items():
    marca = "✅" if mis_respuestas[numero] == tipo else "🔁"
    print(f"{marca} {numero}. {tipo:8} — {razon}")

# %% [markdown]
# ## Glosario de bolsillo
#
# | Término | En una frase |
# |---|---|
# | LLM | Modelo que predice texto; escribe bien, puede inventar |
# | Token | Pedazo de palabra; unidad de lectura y de cobro |
# | Prompt / system prompt | Lo que le enviamos al modelo / las reglas generales |
# | Herramienta (tool) | Función de nuestro programa que el modelo puede pedir |
# | Tool call | El pedido del modelo: nombre de herramienta + argumentos |
# | Observación | Lo que devolvió la herramienta y vuelve al modelo |
# | Agente | LLM en un bucle que decide acciones con herramientas, con límites |
# | Workflow | Pasos definidos por el código, aunque alguno use un LLM |
# | Grafo (LangGraph) | Mapa de pasos (nodos) y transiciones (aristas) con un estado compartido |
# | Estado | Los datos que viajan entre los pasos de un grafo |
# | Orquestación | Organizar quién hace qué, en qué orden y cómo se unen resultados |
# | Subagente | Agente especialista al que otro agente le delega una tarea |
# | Deep agent | Agente que planifica, usa archivos y delega en subagentes |
# | Human-in-the-loop | Una persona aprueba, edita o rechaza antes de una acción |
# | Alucinación | Respuesta segura pero inventada; por eso exigimos fuentes |
#
# ## Ticket de salida
# Respondé en tres líneas (oral, escrito o dibujo):
# - ¿Qué hace el modelo y qué hace el programa en un tool call?
# - Nombrá un problema de tu trabajo que resolverías con un workflow y otro con un agente.
# - ¿Qué modelo GPT-6 elegirías para clasificar 10.000 mensajes, y por qué?
#
# **Próxima clase:** construimos desde cero la herramienta `buscar_archivo` que usó el
# agente de hoy, con un contrato que no se deja engañar.
