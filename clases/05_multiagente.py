# %% [markdown]
# # Clase 5 · Multiagente: cuándo dividir el trabajo
#
# ¿Cuándo conviene que varios agentes colaboren, y cómo se reparten el control?
#
# **Vas a construir:**
# - Un **supervisor** que delega y vuelve a decidir, con un límite de delegaciones.
# - Un coordinador que usa **agentes como herramientas**.
# - Un **handoff**: un agente que entrega el control a otro y no lo recupera.
#
# **Necesitas:** clases 3 y 4 (y la ruta 1). Modo offline por defecto; live si el docente lo activa.
#
# **Recorrido:**
# - ¿Uno o varios agentes?
# - Supervisor con `Command`
# - ☕ Pausa
# - Agentes como herramientas
# - Handoff
# - ☕ Pausa
# - Comparación, proyecto y cierre

# %%
from henry_agents.agentic import mostrar_grafo, ver_en_vivo
from henry_agents.config import configure
from henry_agents.practica import comprobar, confirmar, ver_solucion

MODE = configure()
print("Modo:", MODE)

# %% [markdown]
# ## ¿Uno o varios agentes?
# Dividir en varios agentes **no** es más inteligente por sí solo: suma llamadas, costo y
# lugares donde algo puede fallar. Conviene cuando:
# - las tareas necesitan **instrucciones o herramientas distintas** (un DJ no necesita el archivo);
# - cada parte se beneficia de un **contexto limpio** (no ver el trabajo intermedio de las otras);
# - quieres **probar cada especialista por separado**.
#
# Si una sola búsqueda resuelve el pedido, un agente (clase 4) alcanza.
#
# 🔮 **Predice:** "Fichas de investigación de Batman y una canción de ambiente". ¿Uno o varios?

# %% [markdown]
# ## Arquitectura: supervisor
# Un **supervisor** es un nodo que decide a quién delegar, recibe el resultado y **vuelve a
# decidir**. El control siempre regresa al centro.
#
# ```text
#            ┌──────────── vuelve ────────────┐
#            ▼                                │
# START → supervisor ── delegar ──→ especialista
#            │
#            └── ya alcanza o se acabó el límite ──→ redactar → END
# ```
#
# 🐍 **Python nuevo:** un nodo puede devolver `Command(goto="nodo", update={...})`: **a dónde
# ir** y **qué cambiar** en el estado, en una sola respuesta. La anotación
# `-> Command[Literal["a", "b"]]` le dice a LangGraph a qué nodos puede saltar.

# %%
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from henry_agents.config import chat_model
from henry_agents.cultural import search_catalog

Opcion = Literal["batman", "fantasticos", "chavo", "canciones", "redactar"]


class DecisionSupervisor(BaseModel):
    siguiente: Opcion = Field(description="Colección a consultar o 'redactar' si ya alcanza")
    motivo: str = Field(description="Una frase que justifique la decisión")


class EstadoSupervisor(TypedDict, total=False):
    pedido: str
    tema: str
    necesita: list[str]
    asignada: str
    hallazgos: Annotated[list[dict], operator.add]
    bitacora: Annotated[list[str], operator.add]
    delegaciones: int
    max_delegaciones: int
    informe: str


# %% [markdown]
# ¿Quién decide? En offline, una **regla**: delegar la primera colección pedida que todavía no
# tenga hallazgos. En live, **GPT-6 con salida estructurada**: solo puede elegir una opción de la
# lista cerrada `Opcion`, así que no puede inventar un destino.

# %%
def decidir_con_regla(estado):
    cubiertas = {h["universe"] for h in estado.get("hallazgos", [])}
    faltan = [u for u in estado["necesita"] if u not in cubiertas]
    if faltan:
        return DecisionSupervisor(siguiente=faltan[0], motivo=f"Falta evidencia de {faltan[0]}")
    return DecisionSupervisor(siguiente="redactar", motivo="Todas las colecciones tienen hallazgos")


def decidir_con_modelo(estado):
    instrucciones = (
        "Eres el supervisor de un equipo que busca en un catálogo ficticio. Según el pedido y los "
        "hallazgos, elige UNA colección para consultar o 'redactar' si ya alcanza. No repitas "
        "colecciones ya consultadas."
    )
    contexto = f"Pedido: {estado['pedido']}\nHallazgos: {estado.get('hallazgos', [])}"
    supervisor_llm = chat_model().with_structured_output(DecisionSupervisor)
    return supervisor_llm.invoke([("system", instrucciones), ("human", contexto)])


decidir = decidir_con_modelo if MODE == "live" else decidir_con_regla

# %% [markdown]
# El supervisor **no busca**: solo decide y delega. Lee el orden de sus controles: primero el
# límite (lo impone el código, aunque el modelo quiera seguir), después la decisión, después
# evitar repetir una colección.

# %%
def supervisor(estado) -> Command[Literal["especialista", "redactar"]]:
    hechas = estado.get("delegaciones", 0)
    if hechas >= estado.get("max_delegaciones", 3):
        return Command(goto="redactar", update={"bitacora": ["Límite de delegaciones alcanzado"]})
    decision = decidir(estado)
    consultadas = {h["universe"] for h in estado.get("hallazgos", [])}
    if decision.siguiente == "redactar" or decision.siguiente in consultadas:
        return Command(goto="redactar", update={"bitacora": [f"Redactar: {decision.motivo}"]})
    return Command(
        goto="especialista",
        update={
            "asignada": decision.siguiente,
            "delegaciones": hechas + 1,
            "bitacora": [f"Delegar en {decision.siguiente}: {decision.motivo}"],
        },
    )


def especialista(estado):
    resultado = search_catalog(estado["tema"], universe=estado["asignada"], top_k=1)
    return {"hallazgos": [{"universe": estado["asignada"], "ids": [h.id for h in resultado.hits]}]}


def redactar(estado):
    lineas = [f"{h['universe']}: {h['ids'] or 'sin evidencia'}" for h in estado.get("hallazgos", [])]
    return {"informe": "\n".join(lineas) or "Sin hallazgos."}


grafo = StateGraph(EstadoSupervisor)
grafo.add_node("supervisor", supervisor)
grafo.add_node("especialista", especialista)
grafo.add_node("redactar", redactar)
grafo.add_edge(START, "supervisor")
grafo.add_edge("especialista", "supervisor")  # El control vuelve al centro.
grafo.add_edge("redactar", END)
app_supervisor = grafo.compile()
mostrar_grafo(app_supervisor)

# %% [markdown]
# 🔮 **Predice:** para "investigación" en batman y canciones, ¿cuántas veces decide el
# supervisor? ¿Cuántas delegaciones hay?

# %%
pedido_supervisor = {
    "pedido": "Necesito evidencia de investigación de Batman y una canción de ambiente.",
    "tema": "investigación",
    "necesita": ["batman", "canciones"],
    "max_delegaciones": 3,
}
supervisado = app_supervisor.invoke(pedido_supervisor)
for linea in supervisado["bitacora"]:
    print("🧭", linea)
print(supervisado["informe"])
if MODE == "offline":
    confirmar(supervisado["delegaciones"] == 2, "Debían ser dos delegaciones: batman y canciones")

# %% [markdown]
# 🔍 **Observa:** tres decisiones (batman, canciones, redactar) y dos delegaciones. La
# **bitácora** (registro de cada decisión y su motivo) te permite auditar al supervisor.
#
# ### ✏️ Tu turno 1 · Un límite más corto
# Con `max_delegaciones=1`, ¿cuántas delegaciones habrá? Escribe tu predicción antes de ejecutar.

# %%
mi_prediccion = None  # ✏️ completa aquí con un número

corto = app_supervisor.invoke({**pedido_supervisor, "max_delegaciones": 1})
print("Bitácora:", corto["bitacora"])
print("Informe:", corto["informe"])

# %%
comprobar(
    mi_prediccion == corto["delegaciones"],
    "Bien: el código cortó en el límite y el informe dice qué falta.",
    "Mira la bitácora: ¿cuántas veces dice 'Delegar'? El límite manda sobre la decisión.",
)

# %%
ver_solucion("05_limite_delegaciones")

# %% [markdown]
# ### ✏️ Tu turno 2 · Tres colecciones
# Pide "herramientas" en **batman**, **fantasticos** y **chavo**. Completa la lista y predice
# cuántas delegaciones habrá con `max_delegaciones=3`.

# %%
mis_colecciones = None  # ✏️ completa aquí: lista con tres colecciones

tres = {}
if mis_colecciones is not None:
    tres = app_supervisor.invoke(
        {
            "pedido": f"Herramientas en {', '.join(mis_colecciones)}.",
            "tema": "herramientas",
            "necesita": mis_colecciones,
            "max_delegaciones": 3,
        }
    )
    print(tres["informe"])

# %%
comprobar(
    tres.get("delegaciones") == 3 and {h["universe"] for h in tres.get("hallazgos", [])}
    == {"batman", "fantasticos", "chavo"},
    "Tres delegaciones, una por colección.",
    'Usa ["batman", "fantasticos", "chavo"] y vuelve a ejecutar la celda anterior.',
)

# %%
ver_solucion("05_tres_colecciones")

# %% [markdown]
# ## ☕ Pausa
#
# ## Agentes como herramientas
# Otra forma de equipo: cada especialista es un **agente completo** (clase 4) y lo envolvemos
# en una **herramienta**. El coordinador es otro agente que decide a quién llamar. Desde su
# punto de vista, "pedirle al DJ" es igual que usar `buscar_archivo`.

# %%
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from henry_agents.agentic import ModeloGuionado, crear_agente

investigador = crear_agente(
    MODE,
    system_prompt="Eres el investigador. Busca fichas con buscar_archivo y cita sus IDs.",
)
dj = crear_agente(
    MODE,
    system_prompt="Eres el DJ. Busca UNA canción con universe='canciones' y cita su ID.",
)


@tool
def consultar_investigador(pedido: str) -> str:
    """Pide al investigador fichas del catálogo sobre un tema y una colección."""
    return investigador.invoke({"messages": [HumanMessage(pedido)]})["messages"][-1].text


@tool
def consultar_dj(pedido: str) -> str:
    """Pide al DJ una canción del catálogo para una actividad."""
    return dj.invoke({"messages": [HumanMessage(pedido)]})["messages"][-1].text


# %% [markdown]
# El coordinador en live es GPT-6 Sol (más capaz: decide y combina). En offline es un **guion**:
# primero llama a los dos especialistas a la vez; después arma la propuesta con lo que
# **realmente** devolvieron.

# %%
def componer_propuesta(mensajes):
    respuestas = [m.text for m in mensajes if isinstance(m, ToolMessage)]
    return AIMessage(content="Propuesta de actividad:\n- " + "\n- ".join(respuestas))


if MODE == "live":
    cerebro_coordinador = chat_model("agent")
else:
    cerebro_coordinador = ModeloGuionado(
        pasos=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "consultar_investigador",
                        "args": {"pedido": "Fichas de investigación de Batman"},
                        "id": "c-1",
                    },
                    {
                        "name": "consultar_dj",
                        "args": {"pedido": "Una canción para una actividad de investigación"},
                        "id": "c-2",
                    },
                ],
            ),
            componer_propuesta,
        ]
    )

coordinador = crear_agente(
    MODE,
    model=cerebro_coordinador,
    tools=[consultar_investigador, consultar_dj],
    system_prompt="Coordinas al investigador y al DJ. Llama a ambos y combina sus respuestas.",
)

# %% [markdown]
# 🔮 **Predice:** ¿el coordinador verá las búsquedas que hizo el DJ por dentro?

# %%
final = ver_en_vivo(coordinador, {"messages": [HumanMessage("Prepara material de investigación")]})
print()
print(final["messages"][-1].text)

# %% [markdown]
# 🔍 **Observa:** las líneas con ↳ son el trabajo **interno** de cada especialista. El coordinador
# solo recibe sus respuestas finales (🔧 `consultar_...`): ese es el **contexto limpio**. En la
# clase 7, la herramienta `task` de Deep Agents es esta misma idea, empaquetada.
#
# ### ✏️ Tu turno 3 · Un tercer especialista
# Completa `consultar_cientifico` para que invoque al agente `cientifico` y devuelva el texto
# de su última respuesta (copia el patrón de `consultar_dj`).

# %%
cientifico = crear_agente(
    MODE,
    system_prompt="Eres el especialista en ciencia. Busca fichas de los Cuatro Fantásticos.",
)


@tool
def consultar_cientifico(pedido: str) -> str:
    """Pide al especialista en ciencia fichas de los Cuatro Fantásticos."""
    return None  # ✏️ completa aquí: invoca a `cientifico` y devuelve el texto final


resultado_cientifico = consultar_cientifico.invoke({"pedido": "ciencia de los Fantásticos"})
print(resultado_cientifico)

# %%
comprobar(
    isinstance(resultado_cientifico, str) and "[FAN-" in resultado_cientifico,
    "El especialista respondió citando fichas de los Fantásticos.",
    'Usa: cientifico.invoke({"messages": [HumanMessage(pedido)]})["messages"][-1].text',
)

# %%
ver_solucion("05_tercer_especialista")

# %% [markdown]
# ## Handoff: entregar el control
# En un **handoff** (traspaso) un agente le pasa el control a otro **y no lo recupera**: quien
# recibe termina el trabajo. Como en una recepción: te derivan a la ventanilla correcta y ahí
# te atienden hasta el final. Lo importante es pasar el **motivo** del traspaso.
#
# ```text
# START → recepción ──(música)──→ música ──→ END
#              └──(fichas)───→ biblioteca ──→ END
# ```

# %%
from henry_agents.agentic import interpretar_pedido, redactar_con_fuentes


class EstadoRecepcion(TypedDict, total=False):
    pedido: str
    responsable: str
    motivo: str
    respuesta: str


def recepcion(estado) -> Command[Literal["musica", "biblioteca"]]:
    if interpretar_pedido(estado["pedido"])["universe"] == "canciones":
        return Command(goto="musica", update={"responsable": "musica", "motivo": "Piden música"})
    return Command(goto="biblioteca", update={"responsable": "biblioteca", "motivo": "Piden fichas"})


def musica(estado):
    tema = interpretar_pedido(estado["pedido"])["query"] or "equipo"
    resultado = search_catalog(tema, universe="canciones", kind="cancion", top_k=1)
    return {"respuesta": redactar_con_fuentes(resultado, tema)}


def biblioteca(estado):
    argumentos = interpretar_pedido(estado["pedido"])
    tema = argumentos["query"] or "equipo"
    resultado = search_catalog(tema, universe=argumentos["universe"], kind="ficha", top_k=2)
    return {"respuesta": redactar_con_fuentes(resultado, tema)}


traspaso = StateGraph(EstadoRecepcion)
traspaso.add_node("recepcion", recepcion)
traspaso.add_node("musica", musica)
traspaso.add_node("biblioteca", biblioteca)
traspaso.add_edge(START, "recepcion")
traspaso.add_edge("musica", END)
traspaso.add_edge("biblioteca", END)
app_traspaso = traspaso.compile()

for pedido in ["Una canción para trabajar en equipo", "Fichas de herramientas de El Chavo"]:
    salida = app_traspaso.invoke({"pedido": pedido})
    print(f"👤 {pedido}\n   → {salida['responsable']} ({salida['motivo']}): {salida['respuesta']}\n")

# %% [markdown]
# 🔍 **Observa:** después de `recepcion` nadie vuelve a decidir. Si el traspaso fue incorrecto,
# el error llega hasta el final: por eso el **motivo** queda registrado.
#
# ## ☕ Pausa
#
# ## Comparación
# | | Supervisor | Agentes como herramientas | Handoff |
# |---|---|---|---|
# | ¿Quién decide después? | El supervisor, siempre | El coordinador | Quien recibe el control |
# | ¿El control vuelve? | Sí | Sí (la herramienta devuelve) | No |
# | Riesgo a probar | Delegar sin fin | Coordinador que no llama a nadie | Traspaso equivocado o circular |
# | Defensa | `max_delegaciones` | Límite de herramientas | Motivo registrado y rutas cerradas |
#
# ## 🧱 Proyecto · Paso 5: un equipo de especialistas
# En `proyectos/asistente_archivo/README.md`, guarda:
# 1. La **bitácora del supervisor** para un pedido de dos colecciones.
# 2. La prueba de un **límite de delegaciones** más corto y qué faltó en el informe.
# 3. Una frase: para el Asistente del Archivo, ¿supervisor, agentes como herramientas o
#    handoff? Justifica con el riesgo que más te preocupa.
#
# ## 🎟️ Ticket de salida
# - ¿Qué diferencia al supervisor del handoff en una frase?
# - ¿Qué es el "contexto limpio" y qué lo produce?
# - Nombra un pedido donde un solo agente alcanza y dividir sería exagerado.
#
# ## 📖 Glosario de hoy
# | Término | En una frase |
# |---|---|
# | Supervisor | Nodo que delega y vuelve a decidir tras cada resultado |
# | `Command` | Respuesta de un nodo que dice a dónde ir y qué cambiar |
# | Bitácora | Registro de cada decisión y su motivo |
# | Agente como herramienta | Un agente completo envuelto en una herramienta |
# | Contexto limpio | El especialista no ve el trabajo de otros; el coordinador solo ve resultados |
# | Handoff | Traspaso del control a otro agente, sin retorno |
#
# ## Límites de lo que hicimos
# - En offline las decisiones del supervisor y del coordinador son reglas o guiones.
# - Los especialistas del supervisor son búsquedas, no agentes completos.
# - El handoff es de un solo paso; uno conversacional necesita memoria y más rutas.
