# %% [markdown]
# # Clase 3 · Los Cuatro Fantásticos: elegir y construir una arquitectura
#
# No vamos a medir inteligencia contando agentes. Vamos a decidir qué estructura
# resuelve una necesidad y a comprobar que no pierde resultados.
#
# **Producto:** un grafo paralelo, un orquestador de workers con plan variable, un
# agente con herramientas (hecho a mano y prearmado) y un supervisor con límite.
# Construimos el paralelo, los workers y el supervisor por partes. El agente hecho a
# mano es una demo guiada; su implementación queda disponible para releer.
#
# **Recorrido de la clase**
#
# - Clasificar tres necesidades por su forma de trabajo
# - Comparar el mapa de arquitecturas y justificar una elección
# - Construir estado y workers de un informe paralelo
# - Unir ramas sin perder datos y probar el resultado
# - Pausa
# - Distinguir paralelismo fijo de plan variable
# - Construir un orquestador con Send y un reducer
# - Observar el bucle de un agente y su límite de llamadas
# - El mismo agente, prearmado con create_agent y middleware de límites
# - Pausa
# - Taller: agregar la colección musical al plan
# - Construir un supervisor que vuelve a decidir tras cada especialista
# - Distinguir supervisor de handoff y anticipar los deep agents
# - Defender una arquitectura con sus límites
#
# **Cómo trabajar:** anticipá una salida, ejecutá una celda, observá y explicá.
# La persona que ejecuta y la que revisa intercambian roles durante el reto.
# No hay carrera: el criterio es explicar el resultado. Si te perdés, usá el punto
# de reenganche más cercano. Las soluciones están después de los retos para poder
# volver a ejecutar todo sin dejar celdas rotas.
#
# **Contexto:** las fichas de Batman, los Cuatro Fantásticos y El Chavo son escenarios
# inventados para aprender ingeniería, no resúmenes de cómics o episodios reales.
# Las canciones son inventadas y solo tienen metadatos: no contienen letras ni audio.
# No necesitás conocer estos personajes para resolver las actividades.
#
# **Dos modos:** offline usa búsqueda real y grafos reales, pero sustituye al LLM por
# reglas/extractos explícitos. Live usa OpenAI (GPT-6) y consume API. El estudiante puede
# hacer toda la práctica offline; el docente demuestra live. No compartas el .env.
#
# **En VS Code:** elegí el kernel `.venv` (arriba a la derecha), ejecutá con Shift + Enter
# y, si cambiás el `.env`, reiniciá el kernel. Conceptos base y glosario: clase 0.

# %% [markdown]
# ## La forma del problema importa
# Tres pedidos: A) buscar y resumir una ficha; B) consultar dos colecciones independientes;
# C) elegir cuántas colecciones consultar según un plan. Dibujá una línea para A,
# dos ramas para B y una lista de tareas para C. Todavía no elijas un framework.
#
# Una arquitectura organiza el control y el estado. Puede tener funciones, llamadas
# a modelos o agentes. **Dos funciones paralelas no son automáticamente dos agentes.**

# %%
from henry_agents.config import configure
from henry_agents.cultural import search_catalog

MODE = configure()
print("Modo de las demos con modelo:", MODE)

# %% [markdown]
# ## Mapa de arquitecturas: cuándo usar cada una
# Estos son patrones que podemos expresar con LangGraph, no botones mágicos ni
# categorías excluyentes. Un sistema puede combinar varios.
#
# | Patrón | Quién decide el siguiente paso | Buen uso | Riesgo o costo | Trabajo en este módulo |
# |---|---|---|---|---|
# | Secuencia / chaining | Orden fijo | Buscar y redactar | Ejecutar pasos innecesarios | Construida en clase 2 |
# | Routing | Regla o clasificador | Elegir especialista o abstenerse | Una ruta incorrecta pierde contexto | Construido en clase 2 |
# | Paralelismo fijo | Grafo conocido | Dos revisiones independientes | Conflictos al escribir estado | Construimos hoy |
# | Orquestador–workers | Plan y reparto de tareas | Número variable de tareas | Duplicados o fan-out sin límite | Construimos hoy |
# | Agente con herramientas | Modelo dentro de un bucle | Elegir acciones según observaciones | Loops y gasto | Demo guiada hoy |
# | Evaluador–optimizador | Evaluación y criterio de salida | Corregir un borrador verificable | Mejoras aparentes sin progreso | Construimos en clase 4 |
# | Supervisor | Coordinador que vuelve a decidir | Delegaciones sucesivas | Cuello de botella central | Construimos hoy (acotado) |
# | Handoff | Agente que transfiere control | Cambiar responsable de conversación | Perder contexto o permisos | Comparación de diseño hoy |
# | Deep agent | Coordinador con plan, archivos y subagentes | Tareas largas con entregables | Costo y depuración más difíciles | Clase 5 |
#
# Checkpointing y revisión humana son capacidades que podemos combinar con estos
# patrones. No convierten por sí solos un workflow en un agente.
#
# **Actividad:** una persona elige patrón para A/B/C y la otra pregunta:
# “¿Qué evidencia necesitarías para elegir algo más complejo?”. Cambien roles.

# %% [markdown]
# ## Arquitectura 3: paralelo fijo
# Inspiración: el equipo de los Cuatro Fantásticos reúne lecturas independientes.
# Nuestro programa comparará evidencia de Batman y Fantásticos sobre herramientas.
# No intentamos simular poderes ni razonamientos de los personajes.
#
# ```text
#           ┌─ worker_batman ─────┐
# START ────┤                    ├─ unir ─→ END
#           └─ worker_fantasticos ┘
# ```
#
# Ambos workers leen la misma consulta. Escriben campos distintos: batman y fantasticos.
# Así evitamos que uno sobrescriba al otro. El nodo unir debe esperar a ambos.

# %%
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class EstadoParalelo(TypedDict, total=False):
    query: str
    batman: list[str]
    fantasticos: list[str]
    sources: list[str]


def worker_batman(estado):
    resultado = search_catalog(estado["query"], universe="batman", top_k=1)
    return {"batman": [h.id for h in resultado.hits]}


def worker_fantasticos(estado):
    resultado = search_catalog(estado["query"], universe="fantasticos", top_k=1)
    return {"fantasticos": [h.id for h in resultado.hits]}


# %% [markdown]
# Antes de conectar, probá cada worker por separado. Si un worker busca en la
# colección incorrecta, paralelizarlo solo hace que el error suceda más rápido.
# ¿Qué esperan obtener al buscar herramientas? BAT-02 y FAN-02.

# %%
print(worker_batman({"query": "herramientas"}))
print(worker_fantasticos({"query": "herramientas"}))


def unir(estado):
    return {"sources": sorted(set(estado["batman"] + estado["fantasticos"]))}


# %% [markdown]
# ## La unión es parte del diseño
# La arista con una **lista de nodos de origen** es una barrera: unir espera ambos.
# Si cada worker tarda un tiempo independiente t1 y t2, una ejecución ideal paralela
# se aproxima a max(t1,t2), más coordinación. No prometemos esa mejora en esta demo
# local diminuta; hay overhead y no todo trabajo admite paralelismo.

# %%
paralelo = StateGraph(EstadoParalelo)
paralelo.add_node("batman", worker_batman)
paralelo.add_node("fantasticos", worker_fantasticos)
paralelo.add_node("unir", unir)
paralelo.add_edge(START, "batman")
paralelo.add_edge(START, "fantasticos")
paralelo.add_edge(["batman", "fantasticos"], "unir")
paralelo.add_edge("unir", END)
app_paralela = paralelo.compile()
informe = app_paralela.invoke({"query": "herramientas"})
assert set(informe["sources"]) == {"BAT-02", "FAN-02"}
print("Informe conjunto:", informe["sources"])

# %% [markdown]
# Dibujamos el grafo: dos ramas que salen de START y se juntan en unir.
# Si el dibujo no muestra la barrera como esperabas, compará con el diagrama de texto.

# %%
from henry_agents.agentic import mostrar_grafo

mostrar_grafo(app_paralela)

# %% [markdown]
# **Microexperimento:** cambiá query por investigación. Batman tiene evidencia y
# Fantásticos puede no tenerla. La unión no debe inventar una fuente para llenar la
# rama vacía. Decí qué afirmarías y qué no afirmarías en un informe final.
#
# **Punto de reenganche 1:** podés identificar dos ramas, los campos que escriben
# y la barrera de unión. No dependemos del orden en que terminan.
#
# ## Pausa
#
# ## Arquitectura 4: orquestador–workers
# En el paralelo fijo conocíamos las dos tareas. Ahora la entrada trae una lista de
# colecciones: puede tener una, dos, tres o cuatro. El plan determina cuántos workers
# se crean. `Send` lleva un estado pequeño a cada tarea.
#
# ```text
# START → validar plan → Send(worker, tarea) × N → reunir → END
# ```
#
# En esta clase el plan es explícito y validado. En otro sistema podría proponerlo
# un LLM con salida estructurada. **No decimos que una lista fija sea planificación
# inteligente**: primero aprendemos a ejecutar y verificar el contrato de un plan.
#
# Todos los workers escribirán la misma clave parts. Necesitamos explicar cómo
# combinar esas escrituras: un reducer recibe lo anterior y lo nuevo y los combina.

# %%
import operator
from typing import Annotated

from langgraph.types import Send

print("Reducer de listas:", operator.add(["BAT-02"], ["FAN-02"]))


class EstadoEquipo(TypedDict, total=False):
    query: str
    universes: list[str]
    parts: Annotated[list[dict], operator.add]
    summary: str


# %% [markdown]
# operator.add concatena listas; no elimina duplicados ni las ordena. Por eso
# normalizamos el plan antes de enviarlo. El orden de llegada de workers no es un
# contrato para la presentación final: ordenamos explícitamente al reunir.
#
# ## Construimos el plan, el reparto y la reunión
# Cada función tiene una responsabilidad pequeña. El worker recibe query y universe,
# no toda la conversación ni credenciales. Reducir el contexto facilita probarlo.


# %%
def planificar(estado):
    permitidas = {"batman", "fantasticos", "chavo", "canciones"}
    pedidas = estado["universes"]
    if not pedidas or not set(pedidas) <= permitidas:
        raise ValueError("Plan vacío o colección desconocida")
    return {"universes": sorted(set(pedidas))}


def repartir(estado):
    return [Send("worker", {"query": estado["query"], "universe": u}) for u in estado["universes"]]


def worker(estado):
    resultado = search_catalog(estado["query"], universe=estado["universe"], top_k=1)
    return {"parts": [{"universe": estado["universe"], "ids": [h.id for h in resultado.hits]}]}


# %%
def reunir(estado):
    partes = sorted(estado["parts"], key=lambda parte: parte["universe"])
    return {"summary": "\n".join(f"{p['universe']}: {p['ids']}" for p in partes)}


equipo = StateGraph(EstadoEquipo)
equipo.add_node("planificar", planificar)
equipo.add_node("worker", worker)
equipo.add_node("reunir", reunir)
equipo.add_edge(START, "planificar")
# ----- Acá está la diferencia con el paralelo fijo -----
# repartir devuelve una lista de Send: una tarea por colección del plan.
equipo.add_conditional_edges("planificar", repartir, ["worker"])
# -------------------------------------------------------
equipo.add_edge("worker", "reunir")
equipo.add_edge("reunir", END)
app_equipo = equipo.compile()
plan = {"query": "equipo", "universes": ["fantasticos", "chavo"], "parts": []}
resultado = app_equipo.invoke(plan)
assert len(resultado["parts"]) == 2
print(resultado["summary"])
mostrar_grafo(app_equipo)

# %% [markdown]
# **Comprobación de comprensión:** ¿qué pasa si duplicamos chavo en el plan?
# Debe ejecutarse una sola tarea para esa colección, porque planificar deduplica.
# ¿Qué pasaría sin reducer si varios workers escribieran parts? LangGraph detectaría
# actualizaciones incompatibles en la misma clave; no debemos ocultarlo con un try genérico.
#
# ## Arquitectura 5: agente con herramientas
# Ahora cambia quién decide: el modelo propone acciones, observa resultados y decide
# si ya puede responder. El grafo controla el bucle y sus límites.
#
# ```text
# START → modelo ── respuesta final ─→ END
#            │
#            └─ tool call → herramienta ─→ modelo
# ```
#
# El módulo implementa AgentState con add_messages, ToolNode y un contador. Al llegar
# al máximo de llamadas termina con un mensaje de límite. recursion_limit es una
# segunda defensa del runtime, no un presupuesto de dinero.
# La demo offline es un guion; live sí consulta el modelo. Registramos acciones y
# observaciones, no pedimos cadenas privadas de pensamiento.

# %%
from langchain_core.messages import HumanMessage, ToolMessage

from henry_agents.cultural import build_tool_agent

agente = build_tool_agent(MODE, max_calls=3)
estado_agente = agente.invoke(
    {"messages": [HumanMessage(content="Busca investigación de Batman")], "calls": 0},
    config={"recursion_limit": 12},
)
assert any(isinstance(m, ToolMessage) for m in estado_agente["messages"])
print("Llamadas al modelo o guion:", estado_agente["calls"])
print("Respuesta final:", estado_agente["messages"][-1].text)

# %% [markdown]
# **Experimento de límite:** cambiá `MAX_LLAMADAS` a 1 y anticipá qué pasa. El agente
# alcanza a pedir la herramienta, pero no le quedan llamadas para redactar con la
# observación: termina con el mensaje de límite. Probá también con 2 y con 5.

# %%
MAX_LLAMADAS = 1  # Cambiá este número y volvé a ejecutar la celda.
agente_limitado = build_tool_agent(MODE, max_calls=MAX_LLAMADAS)
estado_limitado = agente_limitado.invoke(
    {"messages": [HumanMessage(content="Busca investigación de Batman")], "calls": 0},
    config={"recursion_limit": 12},
)
print("Llamadas al modelo o guion:", estado_limitado["calls"])
print("Respuesta final:", estado_limitado["messages"][-1].text)

# %% [markdown]
# ## El mismo agente, prearmado: create_agent y middleware
# Escribir el bucle a mano sirve para entenderlo. En proyectos reales usamos
# `create_agent` de LangChain, que construye ese mismo grafo modelo ↔ herramientas.
# Los límites se agregan como **middleware**: piezas que se ejecutan antes o después
# del modelo, como un control de seguridad en la puerta.
#
# - `ModelCallLimitMiddleware(run_limit=N)`: corta tras N llamadas al modelo.
# - `ToolCallLimitMiddleware(run_limit=N)`: limita cuántas herramientas se ejecutan.
#
# **Prueba de estrés:** usamos un guion que **nunca deja de pedir la herramienta**, como
# un modelo atascado. Sin límite, el bucle seguiría hasta el `recursion_limit`.
# **Predicción:** con `max_model_calls=3`, ¿cuántas búsquedas se ejecutan?

# %%
from henry_agents.agentic import ModeloGuionado, build_prebuilt_agent, linea_de_tiempo, llamar

# Este guion se usa también en live: no queremos pagar para ver un bucle atascado.
atascado = ModeloGuionado(
    pasos=[llamar("buscar_archivo", {"query": "investigación"}, f"repite-{i}") for i in range(50)]
)
agente_prearmado = build_prebuilt_agent(MODE, model=atascado, max_model_calls=3, max_tool_calls=10)
salida = agente_prearmado.invoke({"messages": [HumanMessage("Busca investigación")]})
linea_de_tiempo(salida["messages"], ancho=90)
busquedas = [m for m in salida["messages"] if isinstance(m, ToolMessage)]
assert len(busquedas) == 3
print("Búsquedas ejecutadas antes del corte:", len(busquedas))

# %% [markdown]
# El middleware cortó el bucle y dejó un mensaje claro. Ahora el mismo agente con el
# modelo del curso (GPT-6 Luna en live; guion que busca una vez y responde en offline).
# Fijate en el dibujo: los nodos de middleware rodean al modelo.

# %%
agente_normal = build_prebuilt_agent(MODE, max_model_calls=4, max_tool_calls=3)
normal = agente_normal.invoke({"messages": [HumanMessage("Busca fichas de investigación de Batman")]})
linea_de_tiempo(normal["messages"], ancho=90)
mostrar_grafo(agente_normal)

# %% [markdown]
# **Punto de reenganche 2:** compará una arista fija con la decisión de llamar una
# herramienta. Si todos los pasos fueran conocidos, ¿qué costo extra agrega un agente?
# ¿Qué ventaja tiene declarar los límites como middleware en lugar de escribirlos a mano?
#
# ## Pausa
#
# ## Taller: el equipo prepara una actividad y su música
# Agregá canciones al plan de cooperación sin crear un nodo nuevo.
#
# 1. Anticipá cuántos workers se crearán para fantástico + chavo + canciones.
# 2. Ejecutá y comprobá que parts conserve una contribución por colección.
# 3. Duplicá una colección y demostrá que no aparece un worker extra.
# 4. Probá una colección desconocida: debe rechazarse, no aceptarse silenciosamente.
#
# **Pista 1:** se modifica universes; la topología permanece igual.
# **Pista 2:** compará conjuntos de universe y cantidad de parts.
# **Entrega:** un plan, su resumen y una justificación de usar Send en lugar de copiar nodos.
# La celda siguiente es el punto de partida con dos colecciones. Modificá su lista
# y comprobá tu predicción antes de avanzar a la solución de tres colecciones.

# %%
mi_plan = {
    "query": "cooperación",
    "universes": ["fantasticos", "chavo"],
    "parts": [],
}
mi_equipo = app_equipo.invoke(mi_plan)
print(mi_equipo["summary"])

# %% [markdown]
# ## Solución y crítica de arquitectura
# El conjunto de dominios y el conteo permiten detectar pérdida o duplicación. La
# síntesis no debe transformar una lista vacía en una afirmación inventada.
# Comparamos una solución completa con tu variante. Las cuatro entradas de la lista
# crean tres tareas: el plan normalizado elimina la repetición de chavo.

# %%
plan_solucion = {
    "query": "cooperación",
    "universes": ["fantasticos", "chavo", "canciones", "chavo"],
    "parts": [],
}
equipo_solucion = app_equipo.invoke(plan_solucion)
esperadas = {"fantasticos", "chavo", "canciones"}
assert {p["universe"] for p in equipo_solucion["parts"]} == esperadas
assert len(equipo_solucion["parts"]) == 3
print(equipo_solucion["summary"])
try:
    app_equipo.invoke({"query": "equipo", "universes": ["internet"], "parts": []})
    raise AssertionError("El plan debía rechazarse")
except ValueError:
    print("Plan desconocido rechazado.")

# %% [markdown]
# ## Arquitectura 6: supervisor
# En orquestador–workers el plan se decide **una vez** al inicio. Un **supervisor**
# vuelve a decidir **después de cada especialista**: mira qué se encontró y elige el
# siguiente paso. El control siempre regresa al centro.
#
# ```text
#            ┌──────────────── vuelve ────────────────┐
#            ▼                                        │
# START → supervisor ── delegar(colección) ──→ especialista
#            │
#            └── ya alcanza o se agotó el límite ──→ redactar → END
# ```
#
# Usamos `Command(goto=..., update=...)`: el nodo devuelve **a dónde ir** y **qué
# cambiar** en el estado, en una sola respuesta. En offline decide una regla ("delegá la
# primera colección pedida que todavía no tenga hallazgos"). En live decide GPT-6 con
# **salida estructurada**: solo puede elegir entre opciones de una lista cerrada.
# El límite de delegaciones lo impone el código, **aunque el modelo quiera seguir**.

# %%
from typing import Literal

from langgraph.types import Command
from pydantic import BaseModel, Field

from henry_agents.config import chat_model

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
        "Sos el supervisor de un equipo de búsqueda en un catálogo ficticio. Según el pedido "
        "y los hallazgos, elegí UNA colección para consultar o 'redactar' si ya alcanza. "
        "No repitas colecciones ya consultadas."
    )
    contexto = f"Pedido: {estado['pedido']}\nHallazgos: {estado.get('hallazgos', [])}"
    supervisor_llm = chat_model().with_structured_output(DecisionSupervisor)
    return supervisor_llm.invoke([("system", instrucciones), ("human", contexto)])


decidir = decidir_con_modelo if MODE == "live" else decidir_con_regla


# %% [markdown]
# El nodo supervisor no busca nada: **solo decide y delega**. Leé el orden de sus
# controles: primero el límite, después la decisión, después evitar repeticiones.


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


grafo_supervisor = StateGraph(EstadoSupervisor)
grafo_supervisor.add_node("supervisor", supervisor)
grafo_supervisor.add_node("especialista", especialista)
grafo_supervisor.add_node("redactar", redactar)
grafo_supervisor.add_edge(START, "supervisor")
grafo_supervisor.add_edge("especialista", "supervisor")  # El control vuelve al centro.
grafo_supervisor.add_edge("redactar", END)
app_supervisor = grafo_supervisor.compile()
mostrar_grafo(app_supervisor)

# %% [markdown]
# **Predicción:** para "investigación" en batman y canciones, ¿cuántas veces pasa el
# control por el supervisor? Contá: decide batman, vuelve, decide canciones, vuelve,
# decide redactar. Tres decisiones, dos delegaciones.

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
assert supervisado["delegaciones"] <= 3
if MODE == "offline":
    assert supervisado["delegaciones"] == 2
    assert {h["universe"] for h in supervisado["hallazgos"]} == {"batman", "canciones"}

# %% [markdown]
# **Experimento:** bajá `max_delegaciones` a 1. El informe queda incompleto y la bitácora
# dice por qué. Un sistema honesto informa lo que falta; no lo inventa.

# %%
corto = app_supervisor.invoke({**pedido_supervisor, "max_delegaciones": 1})
print(corto["bitacora"])
assert corto["delegaciones"] == 1
assert "Límite de delegaciones alcanzado" in corto["bitacora"]

# %% [markdown]
# ### Supervisor y handoff: no son lo mismo
# **Supervisor:** lo acabamos de construir. El coordinador recibe el resultado de un
# especialista y vuelve a decidir. El control regresa al centro; hay que limitar
# delegaciones (lo hicimos con `max_delegaciones`).
#
# **Handoff:** el responsable actual transfiere el control y contexto al siguiente.
# En una analogía de la vecindad, quien recibe el recado incompleto lo deriva a quien
# puede confirmar la hora; no vuelve a decidir todo desde el principio. En LangGraph,
# Command(update=..., goto=...) puede expresar actualización más transferencia.
#
# | Pregunta | Supervisor | Handoff |
# |---|---|---|
# | ¿Quién decide luego del especialista? | El coordinador | El nuevo responsable |
# | ¿Qué contrato necesita? | Tarea y resultado | Contexto, motivo y responsabilidad |
# | ¿Qué error debemos probar? | Delegación infinita | Transferencia circular o pérdida de contexto |
#
# Hoy construimos un supervisor acotado y comparamos el handoff en diseño; no
# afirmamos haber implementado un handoff conversacional completo. En la clase 5,
# Deep Agents usa la herramienta `task` para delegar: es un supervisor donde el
# coordinador es un modelo y los especialistas son subagentes con su propio bucle.
#
# **Discusión:** si solo queremos dos IDs, no necesitamos un supervisor.
# ¿Qué requisito nuevo justificaría agregarlo? Escribí el requisito antes del patrón.
#
# ## Ticket de salida
# Elegí una arquitectura para “comparar evidencia de tres colecciones y recomendar
# una canción”. Dibujá dónde viaja el estado, qué se ejecuta simultáneamente y quién
# une resultados. Nombrá un caso de error y cómo lo probarías.
#
# **Criterio de logro:** justificás el patrón por la tarea, no por parecer más avanzado.
# En la clase 4 agregamos revisión, ciclos acotados y una decisión humana; en la
# clase 5 juntamos todo en un deep agent.
