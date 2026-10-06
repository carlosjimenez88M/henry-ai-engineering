# %% [markdown]
# # Clase 0 · Bienvenida al mundo agéntico
# ¿Qué es un agente de IA y cuándo **no** conviene usar uno?
#
# **Vas a construir:**
# - Un entorno que funciona en VS Code, comprobado por ti.
# - Tu primera conversación con un modelo y con un agente, leída paso a paso.
# - Un criterio para elegir el nivel de autonomía y el modelo según tarea y costo.
#
# **Necesitas:** la [ruta 1 · La Esquina](agentic_workflows/README.md) (Python y workflows).
# Modo offline por defecto; live si el docente lo activa.

# %% [markdown]
# **Recorrido**
# - Comprobar el entorno
# - Seis ideas clave, con analogías
# - La escalera de autonomía
# - ☕ Pausa
# - Modelos GPT-6 y cuánto cuestan
# - Primera llamada a un modelo
# - Un agente en acción, y tú cambias el pedido
# - ☕ Pausa
# - Taller: ¿workflow o agente?
# - Proyecto, ticket y glosario

# %% [markdown]
# ## Comprobar el entorno
# El **kernel** es el programa de Python que ejecuta las celdas. Debe ser el del curso:
# la carpeta `.venv`. Si no lo es, arriba a la derecha elige **Select Kernel → .venv**.
#
# 🔮 **Predice:** ¿la ruta que vas a ver contiene `.venv`?

# %%
import sys

from henry_agents.practica import comprobar

print("Python:", sys.version.split()[0])
print("Ejecutable:", sys.executable)
comprobar(
    ".venv" in sys.executable,
    "Estás usando el entorno del curso.",
    "Cambia el kernel a .venv (Select Kernel) y vuelve a ejecutar.",
)

# %% [markdown]
# El curso tiene dos **modos**:
# - **offline** (gratis): herramientas y grafos reales; el "cerebro" son **reglas** visibles.
# - **live** (docente): el cerebro es un modelo real de OpenAI (GPT-6) y cuesta dinero.
#
# El modo se lee del archivo `.env`. Si lo cambias, reinicia el kernel (botón **Restart**).

# %%
from henry_agents.config import configure, model_name

MODE = configure()
print("Modo:", MODE)
print("Modelo de las clases:", model_name(), "| Modelo coordinador:", model_name("agent"))

# %% [markdown]
# ## Seis ideas clave
# **LLM** (*Large Language Model*): un programa que, dado un texto, predice cómo seguirlo.
# Es un autocompletado muy bueno: escribe con fluidez, pero **puede inventar** con seguridad.
#
# **Token:** el pedacito de palabra con el que el modelo lee y escribe. Se paga por token.
#
# **Prompt:** lo que le envías al modelo. El *system prompt* fija las reglas; el mensaje
# de la persona trae el pedido.

# %% [markdown]
# **Herramienta** (*tool*): una función de **tu programa** que el modelo puede *pedir*.
# Regla de oro: **el modelo propone, el programa ejecuta.**
#
# **Agente:** un modelo que trabaja en un **bucle**: decide, pide una herramienta, mira el
# resultado y vuelve a decidir, hasta responder.
#
# ```text
# pedido → modelo decide ─¿herramienta?─ sí → programa ejecuta → resultado ─┐
#              ▲                                                            │
#              └────────────────────────────────────────────────────────────┘
#              └─ no → respuesta final
# ```

# %% [markdown]
# **Workflow vs. agente.** En la ruta 1 hiciste *workflows*: una **receta** con pasos fijos.
# Un agente es un **cocinero** que mira la heladera y decide. Es más flexible, pero más caro,
# más lento y más difícil de probar. Usa la receta cuando alcance.
#
# **Orquestación:** organizar quién hace qué, en qué orden y cómo se juntan los resultados.
# **LangGraph** es la librería que usamos para eso. **Deep Agents** es un agente "con oficio"
# construido sobre LangGraph: planifica, toma notas en archivos y delega en ayudantes.

# %% [markdown]
# ## La escalera de autonomía
# Sube un escalón solo si el anterior no alcanza. Cada escalón es una clase de esta ruta.
#
# | Escalón | Quién decide el siguiente paso | Clase |
# |---|---|---|
# | Herramienta con contrato | Tu programa valida y ejecuta | 1 |
# | Respuesta con evidencia (RAG) | Tu programa busca; el modelo redacta | 2 |
# | Workflow en LangGraph | El código (reglas y rutas) | 3 |
# | Agente | **El modelo**, con límites | 4 |
# | Equipo de agentes | Un supervisor | 5 |
# | Evaluación y aprobación humana | Un evaluador y una persona | 6 |
# | Deep agent | Un coordinador que planifica y delega | 7 |
#
# 🔮 **Predice:** ¿por qué no empezar siempre por el escalón más alto?
#
# ## ☕ Pausa

# %% [markdown]
# ## Modelos GPT-6 y cuánto cuestan
# OpenAI ofrece tres niveles. Los precios son por **millón de tokens** (entrada / salida)
# y cambian con el tiempo: el docente los revisa antes de cada cohorte.
#
# **Esfuerzo de razonamiento** (*reasoning effort*): cuánto "piensa" el modelo antes de
# responder. Más esfuerzo = mejores respuestas difíciles, pero más tokens y más costo.
# El curso usa `low`.

# %%
from henry_agents.config import MODELOS, costo_usd

for nombre, datos in MODELOS.items():
    entrada, salida = datos["precio"]
    print(f"{nombre:12} ${entrada:>5} / ${salida:>5}  → {datos['uso']}")

# %% [markdown]
# Un agente típico de clase usa unos 3.000 tokens de entrada y 500 de salida por llamada.
#
# 🔮 **Predice:** ¿cuántas veces más caro es Astra que Luna para la misma llamada?

# %%
for nombre in MODELOS:
    print(f"{nombre:12} una llamada ≈ ${costo_usd(nombre, 3000, 500):.5f}")

# %% [markdown]
# ### ✏️ Tu turno: el costo de un curso
# Un curso de 30 personas hace 20 consultas cada una, y cada consulta hace 5 llamadas.
# Completa `llamadas_totales` (una multiplicación) y ejecuta. La celda siguiente revisa.

# %%
llamadas_totales = None  # ✏️ completa aquí: 30 personas × 20 consultas × 5 llamadas

if llamadas_totales:
    for nombre in ["gpt-6-luna", "gpt-6-astra"]:
        print(f"{nombre:12} ≈ ${llamadas_totales * costo_usd(nombre, 3000, 500):.2f}")

# %%
comprobar(
    llamadas_totales == 3000,
    "Son 3.000 llamadas: con Luna ≈ $1.65 y con Astra ≈ $165. Elegir modelo es ingeniería.",
    "Multiplica las tres cantidades: 30 * 20 * 5.",
)

# %%
from henry_agents.practica import ver_solucion

ver_solucion("00_costo_curso")

# %% [markdown]
# **Regla práctica:** empieza con el modelo más barato que pase tus pruebas. En la clase 7
# combinamos: Sol coordina y Luna hace las tareas acotadas de los especialistas.
#
# ## Primera llamada a un modelo
# Enviamos dos mensajes: reglas (*system*) y pedido (*human*). `cerebro(MODE)` devuelve
# GPT-6 en live y, en offline, `ModeloReglas`: un cerebro de **reglas escritas a mano**.
# No es inteligencia artificial: te lo decimos para que siempre sepas qué estás viendo.

# %%
from henry_agents.agentic import cerebro

modelo = cerebro(MODE)
respuesta = modelo.invoke(
    [
        ("system", "Responde en dos oraciones, sin tecnicismos."),
        ("human", "Hola, me llamo Ana. ¿Qué es un agente de IA?"),
    ]
)
print("Quién respondió:", respuesta.response_metadata.get("model_name"))
print("Respuesta:", respuesta.text)
print("Tokens:", respuesta.usage_metadata or "no aplica en offline")

# %% [markdown]
# 🔍 **Observa:**
# - En offline responde `reglas-offline`: saluda porque una regla detectó "me llamo".
# - En live responde `gpt-6-luna` y `usage_metadata` muestra los tokens reales: con eso se
#   calcula el costo exacto.
# - El *system prompt* cambia el comportamiento, pero **no es una barrera de seguridad**.
#   Los límites reales van en el código, como verás en la clase 4.

# %% [markdown]
# ## Un agente en acción
# `crear_agente` arma el bucle modelo → herramienta → modelo con límites de seguridad.
# La herramienta es `buscar_archivo`: busca en el catálogo ficticio del curso (la construyes
# en la clase 1). `linea_de_tiempo` muestra quién hizo qué.
#
# 🔮 **Predice:** para "Busca fichas de investigación de Batman", ¿cuántas veces usará la
# herramienta? ¿Qué IDs esperas (empiezan con `BAT-`)?

# %%
from langchain_core.messages import HumanMessage

from henry_agents.agentic import crear_agente, linea_de_tiempo

agente = crear_agente(MODE)
resultado = agente.invoke({"messages": [HumanMessage("Busca fichas de investigación de Batman")]})
linea_de_tiempo(resultado["messages"])

# %% [markdown]
# 🔍 **Observa:**
# - 👤 la persona pide; 🤖 el modelo **propone** `buscar_archivo` con argumentos.
# - 🔧 **tu programa** ejecuta la búsqueda y devuelve fichas con IDs.
# - 🤖 el modelo responde citando esos IDs entre corchetes.
#
# Por dentro, el agente es un **grafo**: el nodo `model` decide, el nodo `tools` ejecuta, y
# unos nodos de **middleware** cuentan llamadas para cortar si algo se repite demasiado.
#
# ```text
# START → [límites] → model ──¿herramienta?── sí → tools ──┐
#                       ▲                                 │
#                       └─────────────────────────────────┘
#                       └─ no → END
# ```

# %%
from henry_agents.practica import confirmar

usadas = [m for m in resultado["messages"] if m.type == "tool"]
confirmar(len(usadas) >= 1, "El agente debía consultar la herramienta antes de responder")
if MODE == "offline":
    confirmar("[BAT-01]" in resultado["messages"][-1].text, "La respuesta debía citar BAT-01")
print("✅ El agente buscó antes de responder.")

# %% [markdown]
# ### ✏️ Tu turno: cambia el pedido
# Un agente **sigue tu pedido**. Escribe un pedido sobre **cooperación** en la colección
# **El Chavo** y ejecuta. Si sale bien, la respuesta cita una ficha que empieza con `CHA-`.

# %%
mi_pedido = None  # ✏️ completa aquí, por ejemplo: "Busca fichas de ... de ..."

mi_resultado = None
if mi_pedido:
    mi_resultado = agente.invoke({"messages": [HumanMessage(mi_pedido)]})
    linea_de_tiempo(mi_resultado["messages"])

# %%
comprobar(
    mi_resultado is not None and "CHA-" in mi_resultado["messages"][-1].text,
    "El agente buscó en la colección que pediste y citó una ficha de El Chavo.",
    "Escribe un texto entre comillas que mencione 'cooperación' y 'El Chavo'.",
)

# %%
ver_solucion("00_cambia_el_pedido")

# %% [markdown]
# **Experimento:** prueba un pedido sin tema, como `"hola"`. En offline, la regla no busca
# y te pide un tema. En live, GPT-6 decide por su cuenta. Un buen agente no busca a ciegas.
#
# ## ☕ Pausa

# %% [markdown]
# ## Taller: ¿workflow o agente?
# Para cada situación elige `"workflow"` (los pasos se conocen de antemano) o `"agente"`
# (el siguiente paso depende de lo que se vaya encontrando). Cambia solo el texto.
#
# 1. Cada mañana, resumir los correos nuevos y enviarlos a un canal.
# 2. Investigar una pregunta abierta, decidiendo qué buscar según lo que aparece.
# 3. Clasificar tickets de soporte en tres áreas y enviarlos a cada una.
# 4. Ayudar a depurar un error desconocido, probando hipótesis.
# 5. Traducir un documento y revisar que no falten párrafos.

# %%
mis_respuestas = {
    1: "workflow",
    2: "workflow",  # ✏️ ¿los pasos se conocen de antemano?
    3: "workflow",
    4: "workflow",  # ✏️ revisa esta también
    5: "workflow",
}

# %%
claves = {1: "workflow", 2: "agente", 3: "workflow", 4: "agente", 5: "workflow"}
aciertos = sum(mis_respuestas[n] == claves[n] for n in claves)
comprobar(
    aciertos == 5,
    "Las cinco son correctas. Lo importante es poder decir por qué.",
    f"Llevas {aciertos} de 5. Pregúntate en cada caso: ¿sé qué pasos vendrán?",
)

# %%
ver_solucion("00_workflow_o_agente")

# %% [markdown]
# ## 🧱 Proyecto · Paso 0: leer el encargo
# Abre `proyectos/asistente_archivo/README.md`. El Centro Cultural pide un asistente que
# busque fichas citando IDs, no invente, prepare actividades y no guarde nada sin aprobación.
#
# **Entrega:** una tabla con cada requisito del encargo, el escalón de autonomía que necesita
# y una razón de una línea. Ejemplo: "Citar IDs → herramienta con contrato (clase 1): los IDs
# los devuelve el programa, no el modelo".

# %% [markdown]
# ## 🎟️ Ticket de salida
# - En un *tool call*, ¿qué hace el modelo y qué hace el programa?
# - Nombra un problema de tu trabajo para un workflow y otro para un agente.
# - ¿Qué modelo GPT-6 elegirías para clasificar 10.000 mensajes cortos, y por qué?
#
# ## 📖 Glosario de hoy
# | Término | En una frase |
# |---|---|
# | Kernel | El Python que ejecuta las celdas del notebook |
# | LLM | Modelo que predice texto; escribe bien y puede inventar |
# | Token | Pedacito de palabra; unidad de lectura y de cobro |
# | Prompt / system prompt | Lo que envías al modelo / sus reglas generales |
# | Herramienta (*tool*) | Función de tu programa que el modelo puede pedir |
# | Agente | Modelo en un bucle que decide acciones con herramientas, con límites |
# | Workflow | Pasos definidos por el código |
# | Orquestación | Organizar quién hace qué, en qué orden y cómo se unen los resultados |
# | Middleware | Pieza que corre antes o después del modelo, por ejemplo para contar llamadas |
# | Esfuerzo de razonamiento | Cuánto "piensa" el modelo antes de responder |
#
# ## Límites de lo que hicimos
# - En offline, el cerebro son reglas: entiende pocos tipos de pedido.
# - Los precios son de octubre de 2026; revísalos antes de calcular costos reales.
# - El catálogo tiene doce fichas inventadas: es un laboratorio, no un buscador real.
