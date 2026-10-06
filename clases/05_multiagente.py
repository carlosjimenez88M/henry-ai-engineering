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
# - Supervisor, en cuatro pasos cortos
# - ☕ Pausa
# - Agentes como herramientas
# - Handoff
# - ☕ Pausa
# - Comparación, proyecto y cierre

# %%
import re

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
# 🔮 **Predice:** "Fichas de investigación de Batman y una canción de ambiente". ¿Uno o varios?

# %% [markdown]
# ## Supervisor · Paso 1: la idea
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
# 🔍 Fíjate en la flecha "vuelve": es lo que lo distingue de los workflows de la clase 3.

# %% [markdown]
# ## Supervisor · Paso 2: el estado y la decisión
# El estado guarda la **bitácora** (registro de cada decisión y su motivo) y los hallazgos.
# Ambas listas usan el reducer `operator.add` de la clase 3: cada paso **agrega**, no reemplaza.
#
# ¿Quién decide? En offline, una **regla**: delegar la primera colección pedida que todavía no
# tenga hallazgos. En live, **GPT-6 con salida estructurada**: solo puede elegir una opción de la
# lista cerrada `Opcion`, así que no puede inventar un destino.

# %%
import operator
from typing import Annotated, Literal, TypedDict

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
print(decidir_con_regla({"necesita": ["batman", "canciones"], "hallazgos": []}))

# %% [markdown]
# 🔍 Con la regla, sin hallazgos todavía, la primera decisión es delegar en `batman`.

# %% [markdown]
# ## Supervisor · Paso 3: el nodo que decide a dónde ir
# Hasta ahora, cada nodo devolvía **qué cambiar** y las aristas decían **a dónde ir**. El
# supervisor necesita decidir las dos cosas a la vez.
#
# 🐍 **Python nuevo:** un nodo puede devolver `Command(goto="nodo", update={...})`: **a dónde
# ir** y **qué cambiar**, en una sola respuesta. La anotación `-> Command[Literal["a", "b"]]`
# después de los paréntesis le avisa a LangGraph a qué nodos puede saltar; así puede dibujarlo.
#
# Lee el orden de los controles: primero el **límite** (lo impone el código, aunque el modelo
# quiera seguir), después la **decisión**, después **no repetir** una colección.

# %%
from langgraph.types import Command


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


# %% [markdown]
# ## Supervisor · Paso 4: especialista, redacción y grafo
# El especialista aquí es una búsqueda simple; lo importante es la forma del control.
# 🔮 **Predice:** ¿cuántas aristas necesita el supervisor, si él mismo decide a dónde ir?

# %%
from langgraph.graph import END, START, StateGraph


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
# 🔍 Ninguna arista **sale** del supervisor: sus salidas las decide `Command` en cada vuelta.
# Las líneas punteadas del dibujo son esos saltos posibles.
#
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
    confirmar(supervisado.get("delegaciones", 0) == 2, "Debían ser dos delegaciones: batman y canciones")

# %% [markdown]
# 🔍 **Observa:** tres decisiones (batman, canciones, redactar) y dos delegaciones. La bitácora
# te deja auditar cada decisión y su motivo.

# %% [markdown]
# ### ✏️ Tu turno 1 · Un límite más corto
# Con `max_delegaciones=1`, ¿cuántas delegaciones habrá? Escribe tu número **antes** de ejecutar.
#
# 🐍 **Python nuevo:** `{**pedido_supervisor, "max_delegaciones": 1}` crea un diccionario
# **nuevo**: copia todo `pedido_supervisor` y reemplaza solo `max_delegaciones`.

# %%
mi_prediccion = None  # ✏️ completa aquí con un número

corto = app_supervisor.invoke({**pedido_supervisor, "max_delegaciones": 1})
print("Bitácora:", corto["bitacora"])
print("Informe:", corto["informe"])

# %%
comprobar(
    mi_prediccion == corto.get("delegaciones", 0),
    "Bien: el código cortó en el límite y el informe muestra qué faltó.",
    "¿Cuántas líneas de la bitácora empiezan con 'Delegar'? ¿Quién manda: la regla o el límite?",
)

# %%
ver_solucion("05_limite_delegaciones")

# %% [markdown]
# ### ✏️ Tu turno 2 · Tres colecciones
# Pide "herramientas" en **batman**, **fantasticos** y **chavo**:
# 1. Completa `mis_colecciones` con esas tres colecciones (en una lista).
# 2. Escribe en `mi_prediccion_tres` cuántas delegaciones esperas con `max_delegaciones=3`.

# %%
mis_colecciones = None  # ✏️ completa aquí: lista con tres colecciones
mi_prediccion_tres = None  # ✏️ completa aquí con un número

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
consultadas = {h["universe"] for h in tres.get("hallazgos", [])}
comprobar(
    set(mis_colecciones or []) == {"batman", "fantasticos", "chavo"} and consultadas <= set(mis_colecciones),
    "Pediste las tres colecciones y el supervisor solo consultó colecciones pedidas.",
    "¿Escribiste los nombres tal como aparecen en `Opcion`, en minúsculas y sin tildes?",
)
comprobar(
    mi_prediccion_tres == tres.get("delegaciones", 0),
    f"Tu predicción coincide: {tres.get('delegaciones', 0)} delegaciones.",
    "Cuenta: ¿cuántas colecciones faltan al principio y cuánto permite el límite?",
)

# %%
ver_solucion("05_tres_colecciones")

# %% [markdown]
# ## ☕ Pausa

# %% [markdown]
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
    name="investigador",  # el nombre aparece en cada línea ↳ del streaming
    system_prompt="Eres el investigador. Busca fichas con buscar_archivo y cita sus IDs entre corchetes.",
)
dj = crear_agente(
    MODE,
    name="dj",
    system_prompt="Eres el DJ. Busca UNA canción con universe='canciones' y cita su ID entre corchetes.",
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
# ¿Quién coordina? En live, **GPT-6 Sol** (más capaz: decide a quién llamar y combina).
# En offline, seamos honestos: el coordinador es un **guion fijo** (`ModeloGuionado`). Siempre
# llama a los dos especialistas y no decide nada. Lo que sí es real: los especialistas buscan
# de verdad y la propuesta final se arma con lo que **ellos** devolvieron.

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
# 🔍 **Observa:** las líneas `↳ [investigador]` y `↳ [dj]` son el trabajo **interno** de cada
# especialista (el nombre sale del parámetro `name`). Trabajan a la vez, por eso se intercalan.
# El coordinador solo recibe sus respuestas finales (🔧 `consultar_...`): ese es el **contexto
# limpio**. En la clase 7, la herramienta `task` de Deep Agents es esta misma idea, empaquetada.

# %% [markdown]
# ### ✏️ Tu turno 3 · Un tercer especialista
# Completa `consultar_cientifico` para que invoque al agente `cientifico` y devuelva el texto
# de su última respuesta. Mira cómo lo hace `consultar_dj` dos celdas más arriba.

# %%
cientifico = crear_agente(
    MODE,
    name="cientifico",
    system_prompt=(
        "Eres el especialista en ciencia. Busca fichas de los Cuatro Fantásticos y cita cada ID "
        "entre corchetes, por ejemplo [FAN-01]."
    ),
)


@tool
def consultar_cientifico(pedido: str) -> str:
    """Pide al especialista en ciencia fichas de los Cuatro Fantásticos."""
    return None  # ✏️ completa aquí: invoca a `cientifico` y devuelve el texto final


resultado_cientifico = consultar_cientifico.invoke({"pedido": "ciencia de los Fantásticos"})
print(resultado_cientifico)

# %%
comprobar(
    isinstance(resultado_cientifico, str) and bool(re.search(r"FAN-\d{2}", resultado_cientifico)),
    "El especialista respondió citando fichas de los Fantásticos.",
    "¿Qué devuelve `consultar_dj`? Tu función debe invocar a `cientifico` del mismo modo.",
)

# %%
ver_solucion("05_tercer_especialista")

# %% [markdown]
# ## Handoff: entregar el control
# En un **handoff** (traspaso) un agente le pasa el control a otro **y no lo recupera**: quien
# recibe termina el trabajo. Como en una recepción: te derivan a la ventanilla correcta y ahí
# te atienden hasta el final. Lo importante es dejar registrado el **motivo** del traspaso.
#
# ```text
# START → recepción ──(música)──→ música ──→ END
#              └──(fichas)───→ biblioteca ──→ END
# ```
#
# 🔮 **Predice:** ¿quién atenderá "Una canción para trabajar en equipo"?

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
# el error llega hasta el final: por eso el **motivo** queda registrado en el estado.

# %% [markdown]
# ## ☕ Pausa

# %% [markdown]
# ## Comparación
# | | Supervisor | Agentes como herramientas | Handoff |
# |---|---|---|---|
# | ¿Quién decide después? | El supervisor, siempre | El coordinador | Quien recibe el control |
# | ¿El control vuelve? | Sí | Sí (la herramienta devuelve) | No |
# | Riesgo a probar | Delegar sin fin | Coordinador que no llama a nadie | Traspaso equivocado o circular |
# | Defensa | `max_delegaciones` | Límite de herramientas | Motivo registrado y rutas cerradas |

# %% [markdown]
# ## 🧱 Proyecto · Paso 5: un equipo de especialistas
# Guarda en tu copia de `proyectos/asistente_archivo/mi_entrega.md` (sección Paso 5):
# 1. La **bitácora del supervisor** para un pedido de dos colecciones.
# 2. La prueba de un **límite de delegaciones** más corto y qué faltó en el informe.
# 3. Una frase: para el Asistente del Archivo, ¿supervisor, agentes como herramientas o
#    handoff? Justifica con el riesgo que más te preocupa.

# %% [markdown]
# ## 🎟️ Ticket de salida
# - ¿Qué diferencia al supervisor del handoff, en una frase?
# - ¿Qué es el "contexto limpio" y qué lo produce?
# - Nombra un pedido donde un solo agente alcanza y dividir sería exagerado.

# %% [markdown]
# ## 📖 Glosario de hoy
# | Término | En una frase |
# |---|---|
# | Supervisor | Nodo que delega y vuelve a decidir tras cada resultado |
# | `Command` | Respuesta de un nodo que dice a dónde ir y qué cambiar |
# | Bitácora | Registro de cada decisión y su motivo |
# | Agente como herramienta | Un agente completo envuelto en una herramienta |
# | Contexto limpio | El especialista no ve el trabajo de otros; el coordinador solo ve resultados |
# | Handoff | Traspaso del control a otro agente, sin retorno |

# %% [markdown]
# ## Límites de lo que hicimos
# - En offline las decisiones del supervisor y del coordinador son reglas o guiones.
# - Los especialistas del supervisor son búsquedas, no agentes completos.
# - El handoff es de un solo paso; uno conversacional necesita memoria y más rutas.
