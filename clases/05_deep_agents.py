# %% [markdown]
# # Clase 5 · Deep Agents: un equipo que planifica, delega y pide permiso
#
# En las clases anteriores construimos cada pieza a mano: herramienta (1), workflow
# (2), orquestación y agente (3), revisión humana (4). Hoy usamos **Deep Agents**, una
# librería sobre LangGraph que junta esas piezas en un agente "con oficio".
#
# **Producto:** un equipo con un coordinador y dos especialistas que prepara una
# ficha de actividad para una clase: busca evidencia, elige música, escribe un archivo
# y **espera tu aprobación** antes de guardarlo.
#
# **Recorrido de la clase**
#
# - ¿Qué le falta a un agente simple para tareas largas?
# - Los cuatro ingredientes de un deep agent: plan, archivos, subagentes y prompt
# - Mirar el equipo antes de ejecutarlo: prompts, especialistas y límites
# - Ejecutar hasta la pausa de aprobación e inspeccionar plan y estado
# - Pausa
# - Aprobar, leer el archivo y seguir la línea de tiempo con subagentes en paralelo
# - Rechazar con un motivo y comprobar que no se escribió nada
# - Verificar las citas con código, no con confianza
# - Pausa
# - Taller: cambiar el pedido, escribir tu propia herramienta y sumar un especialista
# - Costos, modelos por rol y cuándo NO usar un deep agent
# - Ticket de salida y proyecto
#
# **Cómo trabajar:** anticipá, ejecutá, observá y explicá. Offline, el coordinador y
# los especialistas siguen **guiones** (decisiones escritas de antemano), pero el
# harness de Deep Agents, los subagentes, las herramientas, el sistema de archivos y la
# aprobación humana son reales. En live decide GPT-6: Sol coordina y Luna investiga.
#
# **Contexto:** el catálogo es ficticio (Batman, Cuatro Fantásticos, El Chavo y canciones
# inventadas, sin letras). No necesitás conocer las obras.

# %% [markdown]
# ## ¿Qué le falta a un agente simple?
# El agente de la clase 3 funciona bien para "buscá y respondé". Pero pensá en un pedido
# como "preparame una actividad completa con evidencia y música". Un agente simple:
# - **Pierde el hilo**: no tiene dónde anotar qué hizo y qué falta.
# - **Satura su contexto**: todos los resultados intermedios se acumulan en la conversación.
# - **Hace todo solo**: no puede pasar una subtarea a alguien con un contexto limpio.
# - **Actúa sin preguntar**: si escribe o envía algo, lo hace sin revisión.
#
# **Deep Agents** agrega cuatro ingredientes, cada uno con una herramienta concreta:
#
# | Ingrediente | Herramienta | Analogía | Lo vimos antes en… |
# |---|---|---|---|
# | Plan | `write_todos` | Una lista de pendientes visible | Plan validado (clase 3) |
# | Archivos | `write_file`, `read_file`, `ls`, `edit_file` | Un cuaderno de trabajo | Estado del grafo (clase 2) |
# | Subagentes | `task` | Delegar en especialistas | Orquestador–workers y supervisor (clase 3) |
# | Prompt detallado | `system_prompt` | El manual de procedimientos | Prompt y contrato (clase 2) |
#
# Y sumamos **aprobación humana** (`interrupt_on`), que es el `interrupt` de la clase 4
# ya integrado. Por dentro, todo es un grafo de LangGraph: nada de magia.
#
# **Predicción:** si el coordinador delega en dos especialistas, ¿tienen que trabajar
# uno después del otro? ¿Qué ventaja tiene que cada uno empiece con un contexto limpio?

# %%
from henry_agents.agentic import PROMPT_COORDINADOR, SUBAGENTES
from henry_agents.config import configure, model_name

MODE = configure()
print("Modo:", MODE)
if MODE == "live":
    print("Coordinador:", model_name("agent"), "| Especialistas:", model_name())
print("\n--- Manual del coordinador (system prompt) ---\n")
print(PROMPT_COORDINADOR)

# %% [markdown]
# Leé el prompt como si fueras el coordinador: ¿qué pasos tiene que seguir? ¿Qué le
# prohibimos? Fijate que **el formato del archivo está especificado**: título, consigna,
# evidencia con IDs, canción. Un buen prompt de agente describe el trabajo como se lo
# describirías a una persona nueva en el equipo.
#
# Ahora miremos a los especialistas. Cada uno es un diccionario con:
# - `name`: cómo lo llama el coordinador;
# - `description`: **cuándo** conviene delegarle (el coordinador lee esto para decidir);
# - `system_prompt`: su propio manual;
# - `tools`: las herramientas que puede usar (les damos solo `buscar_archivo`).

# %%
for especialista in SUBAGENTES:
    print(f"🧑‍💼 {especialista['name']}: {especialista['description']}")
    print(f"   Manual: {especialista['system_prompt']}\n")

# %% [markdown]
# ## Construimos el equipo
# `build_deep_researcher` llama a `create_deep_agent` (ver `src/henry_agents/agentic.py`).
# Lo esencial cabe en pocas líneas:
#
# ```python
# create_deep_agent(
#     model=coordinador,                    # GPT-6.1 Sol en live
#     tools=[buscar_archivo],               # la herramienta de la clase 1
#     system_prompt=PROMPT_COORDINADOR,     # el manual
#     subagents=[investigador, dj],         # especialistas con contexto propio
#     middleware=[TodoListMiddleware(),     # habilita write_todos (el plan)
#                 ModelCallLimitMiddleware(run_limit=20)],  # freno de seguridad
#     interrupt_on={"write_file": True},    # pedir permiso antes de escribir
#     checkpointer=InMemorySaver(),         # recordar el estado para poder pausar
# )
# ```
#
# Los **archivos son virtuales**: viven en el estado del grafo, no en tu disco. El agente
# no puede borrar ni leer archivos reales de tu computador. Esa es una decisión de seguridad.

# %%
from uuid import uuid4

from langchain_core.messages import HumanMessage

from henry_agents.agentic import build_deep_researcher, mostrar_grafo, solicitudes_pendientes

PEDIDO = (
    "Prepará una actividad de clase sobre investigación usando fichas de Batman "
    "y una canción de ambiente del catálogo."
)
equipo = build_deep_researcher(MODE, tema="investigación", universe="batman")
mostrar_grafo(equipo)

# %% [markdown]
# El dibujo muestra el bucle de siempre (`model` ↔ `tools`) rodeado de middleware:
# el que corrige llamadas incompletas, el que cuenta llamadas, el del plan y el de
# aprobación humana. **Un deep agent es un agente de la clase 3 con más oficio**, no otra cosa.
#
# ## Ejecutamos hasta la pausa
# Cada conversación necesita un `thread_id` (como en la clase 4): es la "carpeta" donde
# el checkpointer guarda el estado para poder pausar y continuar.
#
# **Predicción:** ¿en qué paso se va a detener el agente? ¿Va a existir ya el archivo?

# %%
config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 60}
estado = equipo.invoke({"messages": [HumanMessage(PEDIDO)]}, config)

pendientes = solicitudes_pendientes(estado)
print("Acciones esperando tu aprobación:", [accion["name"] for accion in pendientes])
print("Archivos guardados hasta ahora:", list(estado.get("files", {})))
print("\nPlan del coordinador:")
for paso in estado.get("todos", []):
    print(f"  [{paso['status']:11}] {paso['content']}")

# %% [markdown]
# El agente **planificó**, **delegó** y **redactó**, pero se detuvo antes de guardar.
# La pausa no es un error: es el diseño. Leamos qué quiere escribir antes de decidir.

# %%
if pendientes:
    propuesta = pendientes[0]["args"]
    print("Archivo:", propuesta.get("file_path"))
    print("-" * 60)
    print(propuesta.get("content", "")[:1500])

# %% [markdown]
# **Antes de aprobar (en pareja):** una persona lee la propuesta y responde:
# ¿cita IDs? ¿la canción tiene sentido para la actividad? ¿hay algo inventado?
# La otra persona decide. Luego cambien roles en el rechazo.
#
# ## Pausa
#
# ## Aprobar y leer el resultado
# Reanudamos con `Command(resume=...)` y una **decisión por cada acción pendiente**.
# Las opciones que permite este middleware son: `approve` (aprobar), `edit` (corregir los
# argumentos), `reject` (rechazar con un motivo) y `respond` (responder sin ejecutar).

# %%
from langgraph.types import Command

decisiones = [{"type": "approve"} for _ in pendientes]
estado = equipo.invoke(Command(resume={"decisions": decisiones}), config)
for _ in range(3):  # En live el agente podría pedir otra aprobación; ponemos un tope.
    if not solicitudes_pendientes(estado):
        break
    decisiones = [{"type": "approve"} for _ in solicitudes_pendientes(estado)]
    estado = equipo.invoke(Command(resume={"decisions": decisiones}), config)

print("Archivos:", list(estado["files"]))
print("Respuesta final:", estado["messages"][-1].text)

# %% [markdown]
# Ahora seguimos la **línea de tiempo** completa. Prestá atención a dos llamadas `task`
# en el mismo paso: el coordinador delegó **en paralelo**. Cada especialista trabajó en
# su propio bucle (con su propia búsqueda) y devolvió solo su conclusión. Eso mantiene
# limpio el contexto del coordinador: no ve las búsquedas intermedias de nadie.

# %%
from henry_agents.agentic import linea_de_tiempo

linea_de_tiempo(estado["messages"], ancho=110)

# %%
actividad = next(iter(estado["files"].values()))["content"]
print(actividad)

# %% [markdown]
# **Punto de reenganche 1:** podés señalar en la línea de tiempo el plan (`write_todos`),
# la delegación (`task`), la escritura (`write_file`) y la respuesta final.
#
# ## Rechazar también es parte del diseño
# Un nuevo pedido en **otro `thread_id`**. Esta vez rechazamos con un motivo. El agente
# recibe el motivo como observación y debe informarlo; **no debe aparecer el archivo**.

# %%
config_rechazo = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 60}
equipo_rechazo = build_deep_researcher(MODE, tema="investigación", universe="batman")
pausado = equipo_rechazo.invoke({"messages": [HumanMessage(PEDIDO)]}, config_rechazo)
rechazos = [
    {"type": "reject", "message": "Antes de guardar, quiero revisar la consigna con el equipo."}
    for _ in solicitudes_pendientes(pausado)
]
rechazado = equipo_rechazo.invoke(Command(resume={"decisions": rechazos}), config_rechazo)
for _ in range(3):  # Si insiste en escribir, seguimos rechazando, con un tope.
    if not solicitudes_pendientes(rechazado):
        break
    rechazos = [
        {"type": "reject", "message": "Sigue sin aprobarse."}
        for _ in solicitudes_pendientes(rechazado)
    ]
    rechazado = equipo_rechazo.invoke(Command(resume={"decisions": rechazos}), config_rechazo)

assert not rechazado.get("files"), "Un rechazo no debe dejar archivos escritos"
print("Archivos tras el rechazo:", list(rechazado.get("files", {})))
print("Respuesta final:", rechazado["messages"][-1].text)

# %% [markdown]
# ## Verificar con código, no con confianza
# El archivo *parece* bien escrito. Pero, como en las clases 2 y 4, una respuesta
# fluida no prueba que las citas existan. Extraemos los IDs con una **expresión regular**
# (un patrón de texto: tres letras mayúsculas, guion, dos dígitos) y los comparamos con
# el catálogo real.

# %%
import re

from henry_agents.cultural import load_catalog

ids_catalogo = {ficha["id"] for ficha in load_catalog()}
citados = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", actividad))
inventados = citados - ids_catalogo
print("IDs citados:", sorted(citados))
print("IDs que no existen en el catálogo:", sorted(inventados) or "ninguno")
assert citados, "La actividad debe citar al menos una fuente"
assert not inventados, "Hay citas inventadas: no se debería haber aprobado"
assert any(i.startswith("MUS-") for i in citados), "Falta la canción citada"

# %% [markdown]
# **Discusión:** si esta comprobación hubiera corrido *antes* de pedir la aprobación,
# la persona revisora tendría menos trabajo. ¿Dónde la pondrías? (Pista: clase 4,
# evaluador–optimizador; o un especialista "verificador", como en el taller.)
#
# **Punto de reenganche 2:** podés explicar por qué aprobar y verificar son pasos distintos.
#
# ## Pausa
#
# ## Taller: el equipo trabaja para la vecindad
# **Parte A · Cambiar el pedido.** Pedí una actividad de **cooperación** con fichas de
# **El Chavo**. Antes de ejecutar, anticipá qué ficha y qué canción deberían aparecer.
# Pista: mirá las etiquetas en `src/henry_agents/data/cultural_catalog.json`.
#
# **Parte B · Tu propia herramienta.** Escribí `validar_ids(texto)` que devuelva un
# diccionario con los IDs citados, los inventados y si todo es válido. Es la comprobación
# de arriba convertida en herramienta (clase 1: contrato claro, salida estructurada).
#
# **Parte C (extra) · Un especialista verificador.** Dale tu herramienta a un nuevo
# subagente. En la solución lo probamos solo, con su propio guion offline.

# %%
MI_TEMA = "cooperación"
MI_COLECCION = "chavo"
mi_pedido = f"Prepará una actividad sobre {MI_TEMA} con fichas de la colección {MI_COLECCION}."
mi_equipo = build_deep_researcher(MODE, tema=MI_TEMA, universe=MI_COLECCION)
mi_config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 60}
mi_estado = mi_equipo.invoke({"messages": [HumanMessage(mi_pedido)]}, mi_config)
for accion in solicitudes_pendientes(mi_estado):
    print("Quiere escribir:", accion["args"].get("file_path"))
    print(accion["args"].get("content", "")[:600])

# %% [markdown]
# ## Solución A: aprobar y comprobar la predicción

# %%
from henry_agents.agentic import ejecutar_con_revision

aprobar_todo = [{"type": "approve"} for _ in solicitudes_pendientes(mi_estado)]
mi_estado, registro = ejecutar_con_revision(
    mi_equipo,
    Command(resume={"decisions": aprobar_todo}),
    mi_config,
    decidir=lambda accion: {"type": "approve"},
)
print("Decisiones tomadas después de la primera:", registro)
mi_actividad = next(iter(mi_estado["files"].values()))["content"]
mis_ids = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", mi_actividad))
print("IDs citados:", sorted(mis_ids))
if MODE == "offline":
    assert "CHA-01" in mis_ids and "MUS-01" in mis_ids

# %% [markdown]
# ## Solución B: una herramienta que verifica citas
# El decorador `@tool` (clase 1) convierte la función en algo que un agente puede pedir.
# El docstring le explica al modelo cuándo usarla. La función no "opina": compara conjuntos.

# %%
from langchain_core.tools import tool


@tool
def validar_ids(texto: str) -> dict:
    """Revisa que todos los IDs citados (formato ABC-01) existan en el catálogo del curso.

    Usar antes de entregar un texto que cite fuentes. Devuelve citados, inventados y valido.
    """
    citados = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", texto))
    inventados = citados - ids_catalogo
    return {
        "citados": sorted(citados),
        "inventados": sorted(inventados),
        "valido": bool(citados) and not inventados,
    }


print(validar_ids.invoke({"texto": mi_actividad}))
print(validar_ids.invoke({"texto": "Según [BAT-99], todo está bien."}))
assert validar_ids.invoke({"texto": mi_actividad})["valido"]
assert not validar_ids.invoke({"texto": "Según [BAT-99]"})["valido"]

# %% [markdown]
# ## Solución C: un especialista verificador
# Un subagente es, por dentro, un agente con su propio prompt y herramientas. Lo creamos
# con `create_agent` y lo probamos **solo**, antes de sumarlo al equipo: si un especialista
# falla aislado, ponerlo en un equipo solo hace más difícil encontrar el error.
#
# Para sumarlo al equipo agregarías a la lista de subagentes:
#
# ```python
# {"name": "verificador",
#  "description": "Revisa que las citas de un texto existan en el catálogo.",
#  "system_prompt": "Usa validar_ids y responde VALIDO o INVALIDO con los IDs inventados.",
#  "tools": [validar_ids]}
# ```
#
# y en el prompt del coordinador: "Antes de write_file, pedí al verificador que revise el texto".

# %%
import json

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from henry_agents.agentic import ModeloGuionado, llamar
from henry_agents.config import chat_model


def veredicto(mensajes):
    """Paso de guion offline: decide usando la observación REAL de validar_ids."""
    observacion = json.loads(mensajes[-1].text)
    if observacion["valido"]:
        return AIMessage(content="VALIDO")
    return AIMessage(content=f"INVALIDO: {', '.join(observacion['inventados'])}")


TEXTO_A_REVISAR = "Según [BAT-01] y [ZZZ-01], el reloj estaba adelantado."
PROMPT_VERIFICADOR = "Usa validar_ids y responde VALIDO o INVALIDO con los IDs inventados."
if MODE == "live":
    cerebro_verificador = chat_model()
else:
    cerebro_verificador = ModeloGuionado(
        pasos=[llamar("validar_ids", {"texto": TEXTO_A_REVISAR}, "v-1"), veredicto]
    )

# %% [markdown]
# El segundo paso del guion arma el veredicto a partir de la **observación real** de la
# herramienta, no de un texto fijo. Así, offline también refleja lo que devolvió validar_ids.
# **Predicción:** ¿qué ID debería marcar como inventado?

# %%
verificador = create_agent(cerebro_verificador, tools=[validar_ids], system_prompt=PROMPT_VERIFICADOR)
revision = verificador.invoke(
    {"messages": [HumanMessage(f"Revisá este texto: {TEXTO_A_REVISAR}")]}
)
linea_de_tiempo(revision["messages"])
assert "INVALIDO" in revision["messages"][-1].text and "ZZZ-01" in revision["messages"][-1].text

# %% [markdown]
# ## Costos, modelos por rol y cuándo NO usar un deep agent
# En live, este equipo hace del orden de 8 a 12 llamadas al modelo por pedido. Por eso:
# - **Modelos por rol:** Sol (más capaz) coordina; Luna (más barata) hace búsquedas acotadas.
# - **Frenos:** `ModelCallLimitMiddleware` corta tras N llamadas; `recursion_limit` corta pasos
#   del grafo. Ninguno es un presupuesto en dólares: eso se controla en la cuenta de OpenAI.
# - **Esfuerzo de razonamiento:** `low` alcanza aquí. Subirlo multiplica tokens.
#
# | Necesidad | Herramienta adecuada | Por qué |
# |---|---|---|
# | Pasos conocidos, auditables | Workflow de LangGraph (clases 2–4) | Más barato, predecible y fácil de probar |
# | Pocas acciones según lo observado | `create_agent` con límites (clase 3) | Un bucle simple alcanza |
# | Tarea larga, varios pasos, subtareas separables, entregables | Deep agent (hoy) | Plan, archivos y delegación evitan perder el hilo |
# | Acción con efecto externo (enviar, pagar, publicar) | Cualquiera + aprobación humana | La responsabilidad final es de una persona |
#
# **Errores típicos que vas a ver en live** (y cómo se diagnostican con la línea de tiempo):
# el coordinador no delega y busca solo; el especialista devuelve demasiado texto; el
# agente intenta escribir dos veces. La respuesta casi nunca es "un modelo más grande":
# primero se ajusta el prompt, las descripciones de los especialistas y los límites.
#
# ## Ticket de salida
# - ¿Qué ingrediente de Deep Agents resuelve "pierde el hilo"? ¿Y "satura su contexto"?
# - ¿Por qué el archivo virtual es una decisión de seguridad?
# - Nombrá una tarea de tu trabajo donde un deep agent sería exagerado y otra donde ayudaría.
#
# ## Proyecto integrador
# Armá un equipo para otra necesidad del catálogo (por ejemplo, "material para una clase
# de ciencia con los Fantásticos"). Entregá: el prompt del coordinador, la lista de
# especialistas con sus descripciones, una ejecución aprobada, una rechazada, y la
# verificación de citas. Justificá por qué un workflow no alcanzaba (o reconocé que sí).
#
# Referencias: [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview),
# [create_agent](https://docs.langchain.com/oss/python/langchain/agents),
# [modelos de OpenAI](https://developers.openai.com/api/docs/models).
