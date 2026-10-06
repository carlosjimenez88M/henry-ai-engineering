# %% [markdown]
# # Clase 7 · Deep Agents: un equipo que planifica, delega y pide permiso
#
# ¿Qué cambia cuando la tarea es larga y tiene un entregable?
#
# **Vas a construir:**
# - Un equipo con un coordinador y dos especialistas que trabajan en paralelo.
# - Un plan visible, un archivo de trabajo y una pausa antes de guardar, editar o borrar.
# - La entrega final de tu proyecto, con las citas verificadas por código.
#
# **Necesitas:** clases 1–6 (y la ruta 1). Modo offline por defecto; live si el docente lo activa.
#
# **Recorrido:**
# - Qué le falta a un agente simple para tareas largas
# - Construir el equipo a la vista: coordinador, especialistas, límites y permisos
# - Verlo trabajar en vivo hasta la pausa
# - ☕ Pausa
# - Aprobar, leer el archivo, rechazar en otra conversación
# - Verificar las citas con código y frenar a un especialista atascado
# - ☕ Pausa
# - ✏️ Tu turno: otro pedido y tu propia herramienta de verificación
# - Costos, cuándo NO usar un deep agent y entrega del proyecto

# %% [markdown]
# ## Qué le falta a un agente simple
# El agente de la clase 4 resuelve bien "busca y responde". Pide ahora "prepara una actividad
# completa con fichas y música" y aparecen cuatro problemas:
#
# | Problema | Ingrediente de Deep Agents | Ya lo viste en… |
# |---|---|---|
# | Pierde el hilo | **Plan** con `write_todos` | Plan validado (clase 3) |
# | Acumula demasiado texto | **Subagentes** con contexto limpio (`task`) | Supervisor (clase 5) |
# | No tiene dónde redactar | **Archivos** virtuales (`write_file`, `read_file`) | Estado del grafo (clase 3) |
# | Actúa sin preguntar | **Permiso** antes de guardar, editar o borrar | Pausa humana (clase 6) |
#
# Deep Agents es un **harness** (arnés): todo lo que rodea al modelo para que trabaje ordenado.
# Por dentro es un grafo de LangGraph, como los que ya construiste.

# %% [markdown]
# ## El manual del coordinador
# Un buen prompt de agente describe el trabajo como se lo explicarías a una persona nueva:
# pasos, a quién delegar, qué entregar y qué está prohibido.
#
# 🔮 **Predice:** según el manual, ¿el coordinador busca fichas por su cuenta?

# %%
from henry_agents.agentic import AYUDANTE_GENERAL, PROMPT_COORDINADOR, SUBAGENTES
from henry_agents.config import configure, model_name
from henry_agents.practica import comprobar, confirmar, ver_solucion

MODE = configure()
print("Modo:", MODE)
print(PROMPT_COORDINADOR)
for especialista in SUBAGENTES:
    print(f"\n🧑‍💼 {especialista['name']}: {especialista['description']}")

# %% [markdown]
# 🔍 **Observa:** el coordinador **no busca**: delega. Cada especialista tiene un `name`, una
# `description` (el coordinador la lee para decidir a quién delegar) y su propio manual.

# %% [markdown]
# ## Construimos el equipo a la vista
# Cada pieza ya la conoces. Lee los comentarios de arriba hacia abajo:
#
# - En **live**, Sol (más capaz) coordina y Luna (más barata) hace las búsquedas.
# - En **offline**, `ModeloCoordinador` y `ModeloReglas` deciden con reglas visibles.
# - Deep Agents agrega por su cuenta un ayudante **general-purpose**. Lo declaramos nosotros
#   (`AYUDANTE_GENERAL`) para darle también un límite.

# %%
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from henry_agents.agentic import LimiteDeLlamadas, ModeloCoordinador, ModeloReglas
from henry_agents.config import chat_model
from henry_agents.cultural import buscar_archivo

if MODE == "live":
    coordinador, especialista = chat_model("agent"), chat_model()
    print("Coordinador:", model_name("agent"), "| Especialistas:", model_name())
else:
    coordinador, especialista = ModeloCoordinador(), ModeloReglas()

especialistas = [
    {**perfil,
     "tools": [buscar_archivo],              # busca; sus herramientas de archivos también piden permiso
     "model": especialista,
     "middleware": [LimiteDeLlamadas(run_limit=6, exit_behavior="end")]}  # freno propio
    for perfil in [*SUBAGENTES, AYUDANTE_GENERAL]
]
equipo = create_deep_agent(
    model=coordinador,
    tools=[],                                       # el coordinador no busca: delega
    system_prompt=PROMPT_COORDINADOR,
    subagents=especialistas,
    middleware=[TodoListMiddleware(),               # habilita el plan (write_todos)
                LimiteDeLlamadas(run_limit=20, exit_behavior="end")],
    interrupt_on={"write_file": True, "edit_file": True, "delete": True},  # permiso antes de actuar
    checkpointer=InMemorySaver(),                   # recordar el estado para poder pausar
)

# %% [markdown]
# Deep Agents le da a **cada** agente (coordinador y especialistas) herramientas de archivos:
# `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob` y `grep`. Leer es libre;
# **guardar, editar y borrar piden permiso** (el diccionario `interrupt_on`).
#
# Los archivos son **virtuales**: viven en el estado del grafo, no en tu disco. El agente
# no puede leer ni borrar archivos reales de tu computador.

# %% [markdown]
# ## Verlo trabajar en vivo
# `ver_en_vivo` muestra cada paso apenas ocurre. Las líneas con ↳ son de un **subagente**, y
# entre corchetes dice cuál.
#
# 🔮 **Predice:** ¿los dos especialistas trabajan uno después del otro o a la vez?
# ¿Va a existir el archivo cuando el equipo se detenga?

# %%
from uuid import uuid4

from langchain_core.messages import HumanMessage

from henry_agents.agentic import solicitudes_pendientes, ver_en_vivo

PEDIDO = "Prepara una actividad de investigación con fichas de Batman y una canción de ambiente."
conversacion = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 80}
estado = ver_en_vivo(equipo, {"messages": [HumanMessage(PEDIDO)]}, conversacion, ancho=110)

# %% [markdown]
# 🔍 **Observa:**
# - Primero un plan (`write_todos`), después **dos** `task` en el mismo paso: delegó en paralelo.
# - Las líneas `↳ [investigador]` y `↳ [dj]` se intercalan: trabajan a la vez.
# - El coordinador no ve las búsquedas internas; solo recibe la conclusión de cada uno.
# - Terminó en ⏸️: pidió permiso para `write_file`.

# %%
pendientes = solicitudes_pendientes(estado)
print("Esperando aprobación:", [accion["name"] for accion in pendientes])
print("Archivos guardados:", list(estado.get("files", {})))
for paso in estado.get("todos", []):
    print(f"  [{paso['status']:11}] {paso['content']}")
if pendientes:
    print("\n--- Propuesta ---\n" + pendientes[0]["args"].get("content", ""))

# %% [markdown]
# Antes de aprobar (solo o en pareja): ¿cita IDs? ¿la canción tiene sentido? ¿hay algo inventado?

# %% [markdown]
# ## ☕ Pausa

# %% [markdown]
# ## Aprobar y leer el archivo
# Se responde con **una decisión por cada acción pendiente**: `approve` (aprobar), `edit`
# (corregir los argumentos), `reject` (rechazar con motivo) o `respond` (contestar sin ejecutar).

# %%
from langgraph.types import Command

for _ in range(3):  # En live el coordinador podría pedir más de un permiso; ponemos un tope.
    pendientes = solicitudes_pendientes(estado)
    if not pendientes:
        break
    decisiones = [{"type": "approve"} for _ in pendientes]
    estado = ver_en_vivo(equipo, Command(resume={"decisions": decisiones}), conversacion, ancho=110)
actividad = estado.get("files", {}).get("/actividad.md", {}).get("content", "")
print("\n" + (actividad or "(no hay /actividad.md)"))
if MODE == "offline":
    confirmar(bool(actividad), "Después de aprobar debía existir /actividad.md")
else:
    comprobar(bool(actividad), "El equipo guardó /actividad.md.",
              "En live el modelo no propuso guardar: revisa la línea de tiempo de arriba.")

# %% [markdown]
# 🔍 **Observa:** al continuar, la propuesta aprobada aparece otra vez (el middleware la
# reenvía) y solo entonces se ejecuta `write_file`. Leemos el archivo **por su nombre**,
# `"/actividad.md"`, porque el agente podría tener otros archivos de notas.

# %% [markdown]
# ## Rechazar en otra conversación
# Otro `thread_id` es otra solicitud. `crear_equipo_profundo` arma el mismo equipo que
# construiste arriba. Rechazamos con un motivo claro: **no debe quedar ningún archivo**.
#
# Si el agente insiste en pedir permiso una y otra vez, `ejecutar_con_revision` se detiene
# con un error (`RuntimeError`): lo atrapamos con `try/except`, como en la ruta 1.

# %%
from henry_agents.agentic import crear_equipo_profundo, ejecutar_con_revision

MOTIVO = "No guardes ningún archivo: termina y explica qué faltó."
rechazos = []


def rechazar(accion):
    rechazos.append(accion["name"])
    return {"type": "reject", "message": MOTIVO}


otro_equipo = crear_equipo_profundo(MODE)
config_rechazo = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 80}
try:
    rechazado, _ = ejecutar_con_revision(
        otro_equipo, {"messages": [HumanMessage(PEDIDO)]}, config_rechazo, decidir=rechazar
    )
except RuntimeError as error:
    print("⚠️", error)
    rechazado = otro_equipo.get_state(config_rechazo).values
print("Acciones rechazadas:", rechazos)
print("Archivos:", list(rechazado.get("files", {})))
print("Respuesta final:", rechazado["messages"][-1].text)
confirmar(not rechazado.get("files"), "Un rechazo no debía dejar archivos")

# %% [markdown]
# ## Verificar las citas con código
# El archivo *parece* correcto. Aplicamos la idea de la clase 2: los IDs citados deben estar
# en el catálogo. Para encontrarlos usamos la expresión regular de la clase 4.

# %%
import re

from henry_agents.cultural import load_catalog

ids_catalogo = {ficha["id"] for ficha in load_catalog()}
citados = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", actividad))
inventados = citados - ids_catalogo
print("Citados:", sorted(citados))
print("Inventados:", sorted(inventados) or "ninguno")
comprobar(bool(citados) and not inventados, "Todas las citas existen en el catálogo.",
          f"Hay citas inexistentes {sorted(inventados)}: justo por esto verificamos con código.")
comprobar(any(i.startswith("MUS-") for i in citados), "La actividad cita una canción.",
          "Falta la canción que pedía el encargo.")
if MODE == "offline":
    confirmar(citados == {"BAT-01", "BAT-03", "MUS-02"}, "Offline debía citar BAT-01, BAT-03 y MUS-02")

# %% [markdown]
# ## Un especialista atascado
# ¿Qué pasa si un especialista entra en bucle? Lo simulamos con `ModeloReglas(falla="bucle")`
# (también en live: no queremos pagar por un bucle). Su **propio** límite lo corta.
#
# 🔮 **Predice:** con `max_llamadas_especialista=3`, ¿cuántas veces busca el investigador
# antes del corte? ¿Qué escribe el coordinador en la sección de fichas?

# %%
atascado = crear_equipo_profundo(
    MODE,
    models={"coordinador": ModeloCoordinador(), "investigador": ModeloReglas(falla="bucle"),
            "dj": ModeloReglas(), "general-purpose": ModeloReglas()},
    max_llamadas_especialista=3,
)
config_atascado = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 80}
pausa = ver_en_vivo(atascado, {"messages": [HumanMessage(PEDIDO)]}, config_atascado, ancho=90)
# pausa["pasos"] guarda (quién, mensaje) de cada paso, también de los subagentes.
busquedas = sum(
    1 for quien, mensaje in pausa["pasos"] if quien == "investigador" and mensaje.type == "tool"
)
print("Búsquedas del investigador antes del corte:", busquedas)

aprobar_todo = [{"type": "approve"} for _ in solicitudes_pendientes(pausa)]
resultado, _ = ejecutar_con_revision(
    atascado, Command(resume={"decisions": aprobar_todo}), config_atascado,
    decidir=lambda accion: {"type": "approve"},
)
print(resultado.get("files", {}).get("/actividad.md", {}).get("content", "(sin archivo)"))
confirmar(busquedas == 3, "El límite de 3 llamadas debía permitir 3 búsquedas")

# %% [markdown]
# 🔍 **Observa:** el investigador buscó 3 veces y su límite lo cortó; el archivo dice
# "Sin evidencia encontrada" en lugar de inventar. Sin ese límite, el bucle seguiría gastando.

# %% [markdown]
# ## ☕ Pausa

# %% [markdown]
# ## ✏️ Tu turno 1 · Otro pedido
# Escribe un pedido de **ciencia** con fichas de **los Fantásticos** y una canción. Antes de
# ejecutar, anota en `mi_prediccion` los IDs que esperas. Puedes averiguarlos con
# `search_catalog` (clase 1), por ejemplo buscando "ciencia" en la colección correcta.

# %%
mi_pedido = None  # ✏️ escribe aquí tu pedido, entre comillas
mi_prediccion = set()  # ✏️ los IDs que esperas, por ejemplo {"XXX-00", "YYY-00"}

# %%
if mi_pedido:
    mi_estado, _ = ejecutar_con_revision(
        crear_equipo_profundo(MODE), {"messages": [HumanMessage(mi_pedido)]},
        {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 80},
        decidir=lambda accion: {"type": "approve"},
    )
    mi_actividad = mi_estado.get("files", {}).get("/actividad.md", {}).get("content", "")
    mis_ids = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", mi_actividad))
    print("Predijiste:", sorted(mi_prediccion), "| Citó:", sorted(mis_ids))
    comprobar(bool(mis_ids) and mi_prediccion == mis_ids, "Tu predicción coincide con lo que citó el equipo.",
              "Compara las dos listas: ¿tu pedido dice 'ciencia' y 'Fantásticos'? ¿qué ID no anticipaste?")
else:
    comprobar(False, "", "Escribe tu pedido en mi_pedido y vuelve a ejecutar.")

# %%
ver_solucion("07_otro_pedido")

# %% [markdown]
# ## ✏️ Tu turno 2 · Tu herramienta de verificación
# Convierte la verificación de citas en una **herramienta** (clase 1) que un especialista
# "verificador" podría usar. Completa las dos líneas marcadas: los `inventados` son los
# citados que **no** están en `ids_catalogo`; `valido` es verdadero solo si hay citas y
# ninguna es inventada.

# %%
from langchain_core.tools import tool


@tool
def validar_ids(texto: str) -> dict:
    """Revisa que los IDs citados (formato ABC-01) existan en el catálogo del curso."""
    citados = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", texto))
    inventados = set()  # ✏️ completa: los citados que no están en ids_catalogo
    valido = None  # ✏️ completa: True si hay citas y no hay inventados
    return {"citados": sorted(citados), "inventados": sorted(inventados), "valido": valido}


# %%
con_inventado = validar_ids.invoke({"texto": "Según [BAT-01] y [ZZZ-01], el reloj estaba adelantado."})
sin_citas = validar_ids.invoke({"texto": "Un texto que no cita ninguna ficha."})
print(con_inventado)
print(sin_citas)
comprobar(
    con_inventado == {"citados": ["BAT-01", "ZZZ-01"], "inventados": ["ZZZ-01"], "valido": False}
    and sin_citas["valido"] is False,
    "Tu herramienta detecta el ID inventado y no acepta un texto sin citas.",
    "Revisa los dos casos: ¿qué devuelve 'valido' cuando no hay ninguna cita?",
)

# %%
ver_solucion("07_validar_ids")

# %% [markdown]
# ## Costos y modelos por rol
# En live, cada pedido hace varias llamadas al modelo (coordinador y especialistas). Mídelo tú
# envolviendo una ejecución con `medir_costo()` (clase 6). Para gastar con criterio:
#
# - **Modelos por rol:** Sol coordina; Luna hace tareas acotadas.
# - **Frenos:** límite del coordinador, límite de cada especialista y `recursion_limit`.
#   Ninguno es un presupuesto en dólares: configura límites y alertas de gasto en tu cuenta.
# - **Esfuerzo de razonamiento** `low`: subirlo multiplica los tokens.

# %% [markdown]
# | Necesidad | Mejor opción | Por qué |
# |---|---|---|
# | Pasos conocidos | Workflow de LangGraph (clase 3) | Más barato, predecible y fácil de probar |
# | Pocas acciones según lo que observa | `create_agent` con límites (clase 4) | Un bucle simple alcanza |
# | Tarea larga, subtareas separables, entregable | Deep agent (hoy) | Plan, archivos y delegación |
# | Acción con efecto externo | Cualquiera + aprobación humana | La responsabilidad es de una persona |
#
# **Herramientas externas (MCP).** El *Model Context Protocol* permite conectar herramientas
# que viven en otros servidores (un calendario, una base de datos) sin programarlas tú.
# No lo instalamos en este curso; aplican las mismas reglas: pocas herramientas, límites y
# aprobación para acciones con efecto.

# %% [markdown]
# ## 🧱 Proyecto · Paso 7: entrega final
# Arma tu equipo profundo para el Asistente del Archivo y presenta:
# 1. El prompt del coordinador y las descripciones de tus especialistas.
# 2. Una ejecución aprobada (con `/actividad.md`) y una rechazada (sin archivo).
# 3. La verificación de citas con `validar_ids` y el caso del especialista atascado.
# 4. Tu defensa: ¿por qué un workflow o un agente simple no alcanzaban? (o reconoce que sí).
#
# Guarda la evidencia en tu copia de `proyectos/asistente_archivo/mi_entrega.md` y usa la
# rúbrica de `proyectos/asistente_archivo/README.md`.

# %% [markdown]
# ## 🎟️ Ticket de salida
# - ¿Qué ingrediente resuelve "pierde el hilo" y cuál "acumula demasiado texto"?
# - ¿Por qué el coordinador no tiene `buscar_archivo`?
# - Nombra una tarea donde un deep agent sería exagerado.

# %% [markdown]
# ## 📖 Glosario de hoy
# | Término | En una frase |
# |---|---|
# | Deep agent | Agente que planifica, usa archivos y delega en subagentes |
# | Harness (arnés) | Todo lo que rodea al modelo para que trabaje ordenado |
# | `write_todos` | Herramienta para escribir y actualizar el plan |
# | `task` | Herramienta para delegar en un subagente |
# | Subagente | Agente especialista con su propio contexto, prompt y límites |
# | Archivo virtual | Archivo que vive en el estado del grafo, no en tu disco |
# | `interrupt_on` | Diccionario que dice qué herramientas necesitan aprobación humana |
# | MCP | Protocolo para conectar herramientas de servidores externos |

# %% [markdown]
# ## Límites de lo que hicimos
# - Offline, el coordinador decide con reglas; en live las decisiones pueden variar.
# - Archivos y pausas viven en memoria: se pierden al reiniciar el kernel.
# - Verificar IDs no prueba que cada frase sea fiel: para eso está el juez de la clase 6.
