# %% [markdown]
# # Clase 6 · Evaluación y control humano
#
# ¿Cómo sabes que tu asistente funciona, y quién decide antes de entregar?
#
# **Vas a construir:**
# - Un ciclo que revisa su propio borrador, lo corrige con un límite y escala si no puede.
# - Una pausa para que una persona apruebe o rechace, y que se puede retomar.
# - Un reporte de evaluación con casos, un juez y el costo medido.
#
# **Necesitas:** clases 2–5 (y la ruta 1). Modo offline por defecto; live si el docente lo activa.
#
# **Recorrido:**
# - Un error que una respuesta bonita puede esconder
# - Evaluador–optimizador en LangGraph: redactar, evaluar, corregir o escalar
# - ☕ Pausa
# - Pausar el grafo para una persona: checkpoint, `thread_id`, aprobar y rechazar
# - Medir con casos: coincidencia exacta y abstención
# - Un juez que revisa fidelidad, y cómo calibrarlo
# - ☕ Pausa
# - ✏️ Tu turno: casos nuevos y una etiqueta para el juez
# - 🧱 Proyecto y cierre

# %% [markdown]
# ## Un error que una respuesta bonita puede esconder
# Imagina este borrador: el texto es fluido, el formato es correcto y el programa no falló.
# Pero cita una ficha que **no existe**. ¿Está listo para entregar?
#
# 🔮 **Predice:** ¿qué tendría que comprobar el programa para detectarlo sin leer el texto?

# %%
from henry_agents.config import ROOT, configure, medir_costo
from henry_agents.cultural import SearchResult, compose, load_catalog, search_catalog
from henry_agents.practica import comprobar, confirmar, ver_solucion

MODE = configure()
evidencia = search_catalog("herramientas", universe="batman", top_k=1)
borrador_roto = {"text": "Un informe impecable.", "source_ids": ["INVENTADA"]}
print("Modo:", MODE)
print("Fichas recuperadas:", [h.id for h in evidencia.hits])
print("Fichas citadas:", borrador_roto["source_ids"])

# %% [markdown]
# 🔍 **Observa:** lo recuperado y lo citado no coinciden. Esa comparación la hace Python,
# sin entender el texto. Es el **evaluador** más simple: un criterio que se puede comprobar.
#
# 🐍 **Python nuevo:** `citados <= disponibles` pregunta si un conjunto está *dentro* de otro
# (todos los citados están entre los disponibles).


# %%
def citas_validas(borrador, evidencia):
    disponibles = {h["id"] for h in evidencia["hits"]}
    citados = set(borrador["source_ids"])
    return bool(citados) and citados <= disponibles


confirmar(not citas_validas(borrador_roto, evidencia.model_dump()), "La cita inventada debía fallar")
print("✅ El evaluador detecta la cita inventada.")

# %% [markdown]
# ## Evaluador–optimizador en LangGraph
# Ya conoces el patrón de la ruta 1. Ahora con un grafo y un **ciclo**:
#
# ```text
# START → recuperar → redactar → evaluar
#                        ↑          ├─ válido ─────────────────────→ listo → END
#                        └──────────┼─ inválido y quedan intentos
#                                   └─ sin evidencia o sin intentos → escalar → END
# ```
#
# El límite `max_intentos` viaja **en el estado**: cada solicitud trae el suyo.
# Para ver el ciclo de forma repetible, **inyectamos** una cita falsa en el primer borrador.

# %%
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class EstadoRevision(TypedDict, total=False):
    pedido: str
    evidencia: dict
    borrador: dict
    intentos: int
    max_intentos: int
    valido: bool
    decision: str
    falla_inyectada: bool


def recuperar(estado):
    return {"evidencia": search_catalog(estado["pedido"], top_k=2).model_dump(), "intentos": 0}


def redactar(estado):
    intento = estado["intentos"] + 1
    borrador = compose(SearchResult.model_validate(estado["evidencia"]), MODE).model_dump()
    if estado.get("falla_inyectada") and intento == 1:
        borrador["source_ids"] = ["INVENTADA"]  # error a propósito, solo en el primer intento
    return {"borrador": borrador, "intentos": intento}


def evaluar(estado):
    return {"valido": citas_validas(estado["borrador"], estado["evidencia"])}


# %% [markdown]
# La decisión tiene un orden. Léelo en voz alta: primero aceptar algo válido; después
# cortar si no hay evidencia o se acabaron los intentos; solo entonces, volver a redactar.
# Repetir sin evidencia no sirve: preguntar otra vez no crea fuentes.


# %%
def decidir(estado):
    if estado["valido"]:
        return "listo"
    if not estado["evidencia"]["hits"] or estado["intentos"] >= estado["max_intentos"]:
        return "escalar"
    return "redactar"


def listo(estado):
    return {"decision": "lista para revisión"}


def escalar(estado):
    # No entregamos un borrador inválido solo porque se acabaron los intentos.
    return {"decision": "escalada", "borrador": {"text": "Necesita revisión humana.", "source_ids": []}}


grafo = StateGraph(EstadoRevision)
for nombre, funcion in [("recuperar", recuperar), ("redactar", redactar), ("evaluar", evaluar),
                        ("listo", listo), ("escalar", escalar)]:
    grafo.add_node(nombre, funcion)
grafo.add_edge(START, "recuperar")
grafo.add_edge("recuperar", "redactar")
grafo.add_edge("redactar", "evaluar")
grafo.add_conditional_edges("evaluar", decidir, ["listo", "escalar", "redactar"])
grafo.add_edge("listo", END)
grafo.add_edge("escalar", END)
app_revision = grafo.compile()

# %% [markdown]
# 🔮 **Predice:** con la falla inyectada y `max_intentos=2`, ¿cuántos intentos hará?
# ¿Y con `max_intentos=1`?

# %%
corregido = app_revision.invoke(
    {"pedido": "investigación Batman", "max_intentos": 2, "falla_inyectada": True}
)
agotado = app_revision.invoke(
    {"pedido": "investigación Batman", "max_intentos": 1, "falla_inyectada": True}
)
sin_evidencia = app_revision.invoke({"pedido": "vacuna marciana", "max_intentos": 3})
for nombre, resultado in [("2 intentos", corregido), ("1 intento", agotado), ("sin evidencia", sin_evidencia)]:
    print(f"{nombre:14} → intentos={resultado['intentos']} decisión={resultado['decision']}")
confirmar(agotado["borrador"]["source_ids"] == [], "Un borrador escalado no debe llevar citas")
if MODE == "offline":
    confirmar(corregido["intentos"] == 2 and corregido["valido"], "Debía corregirse en el intento 2")
    confirmar(sin_evidencia["intentos"] == 1, "Sin evidencia no debía repetir")

# %% [markdown]
# 🔍 **Observa:**
# - Con 2 intentos, el segundo borrador corrige la cita y queda listo.
# - Con 1 intento, escala y **no** entrega el borrador malo.
# - Sin evidencia, escala de inmediato: no gasta intentos en vano.
#
# ## ☕ Pausa

# %% [markdown]
# ## Pausar el grafo para que decida una persona
# Un **checkpoint** es una foto del estado del grafo guardada en un momento. Gracias a esa
# foto, el grafo puede **pausarse** con `interrupt(...)` y **continuar** después con
# `Command(resume=...)`, aunque pasen minutos.
#
# - `InMemorySaver` guarda las fotos en la memoria del programa: si reinicias el kernel, se pierden.
# - `thread_id` es el nombre de la conversación: con él se retoma la solicitud correcta.
#
# Al continuar, el nodo pausado **vuelve a ejecutarse desde su inicio**. Por eso, antes de
# `interrupt` no pongas acciones que no deban repetirse (como enviar un correo).

# %%
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt


def revisar(estado):
    respuesta = interrupt({"propuesta": estado["borrador"], "opciones": ["aprobar", "rechazar"]})
    return {"decision": respuesta}


def decision_valida(estado):
    # Si la persona escribe otra cosa, el grafo vuelve a preguntar.
    return END if estado["decision"] in ("aprobar", "rechazar") else "revisar"


humano = StateGraph(EstadoRevision)
humano.add_node("revisar", revisar)
humano.add_edge(START, "revisar")
humano.add_conditional_edges("revisar", decision_valida, ["revisar", END])
app_humana = humano.compile(checkpointer=InMemorySaver())

hilo_a = {"configurable": {"thread_id": str(uuid4())}}
pausado = app_humana.invoke({"borrador": corregido["borrador"]}, hilo_a)
print("Propuesta esperando:", pausado["__interrupt__"][0].value["propuesta"]["source_ids"])
print("Próximo paso:", app_humana.get_state(hilo_a).next)

# %% [markdown]
# 🔍 **Observa:** el grafo no terminó: quedó detenido en `revisar`. Antes de continuar, una
# persona de la pareja lee la propuesta y dice qué verificó. Aquí simulamos su respuesta.

# %%
aprobado = app_humana.invoke(Command(resume="aprobar"), hilo_a)
print("Hilo A:", aprobado["decision"])

hilo_b = {"configurable": {"thread_id": str(uuid4())}}
app_humana.invoke({"borrador": corregido["borrador"]}, hilo_b)
otra_vez = app_humana.invoke(Command(resume="tal vez"), hilo_b)  # respuesta inválida
print("Hilo B tras 'tal vez': ¿sigue en pausa?", bool(otra_vez.get("__interrupt__")))
rechazado = app_humana.invoke(Command(resume="rechazar"), hilo_b)
print("Hilo B:", rechazado["decision"], "| Hilo A sigue:", app_humana.get_state(hilo_a).values["decision"])
confirmar(app_humana.get_state(hilo_a).values["decision"] == "aprobar", "Los hilos debían ser independientes")

# %% [markdown]
# 🔍 **Observa:** "tal vez" no se aceptó: el grafo volvió a preguntar. Y rechazar el hilo B
# no tocó la aprobación del hilo A: cada `thread_id` es una solicitud distinta.
#
# En una aplicación real, la decisión viene de una persona identificada, nunca del modelo.

# %% [markdown]
# ## Medir con casos
# Un conjunto de casos con respuesta conocida (*golden set*) es un contrato de comportamiento.
# Medimos dos cosas por caso:
#
# - **Coincidencia exacta:** los IDs recuperados son *exactamente* los esperados (ni más ni menos).
# - **Abstención correcta:** si se esperaba vacío, no devolvió nada; si se esperaba algo, no quedó vacío.

# %%
import json


def evaluar_casos(casos):
    filas = []
    for caso in casos:
        recibido = search_catalog(caso["query"], universe=caso["universe"],
                                  kind=caso.get("kind", "todos"), top_k=caso.get("top_k", 3))
        ids = sorted(h.id for h in recibido.hits)
        filas.append({
            "query": caso["query"],
            "esperado": sorted(caso["expected"]),
            "recibido": ids,
            "exacta": ids == sorted(caso["expected"]),
            "abstencion_ok": (not ids) == (not caso["expected"]),
        })
    return filas


golden = json.loads((ROOT / "src/henry_agents/data/cultural_golden.json").read_text(encoding="utf-8"))
filas = evaluar_casos(golden)
for fila in filas:
    print("✅" if fila["exacta"] else "❌", f"{fila['query']:24}", fila["recibido"])
print(f"Exactas: {sum(f['exacta'] for f in filas)}/{len(filas)} · "
      f"Abstención correcta: {sum(f['abstencion_ok'] for f in filas)}/{len(filas)}")

# %% [markdown]
# 🔍 **Observa:** "Batman vacuna marciana" espera vacío. Que exista la colección Batman no
# significa que exista el tema. Esos casos "trampa" valen más que diez casos fáciles.

# %% [markdown]
# ## Un juez que revisa fidelidad
# Las citas pueden existir y aun así la frase decir algo **falso** sobre la ficha. Para eso
# usamos un **juez**: un revisor que devuelve un veredicto estructurado.
#
# - **offline:** el juez son **reglas** (no entiende el texto): cada ID citado debe existir
#   y la frase debe compartir palabras clave con esa ficha.
# - **live:** el juez es GPT-6 con salida estructurada `Veredicto`.

# %%
import re

from pydantic import BaseModel, Field

from henry_agents.cultural import query_terms

FICHAS = {f["id"]: f for f in load_catalog()}


class Veredicto(BaseModel):
    fiel: bool = Field(description="True si cada afirmación está respaldada por la ficha citada")
    motivo: str = Field(description="Una frase que explica el veredicto")


def juez_reglas(respuesta):
    citados = re.findall(r"\b[A-Z]{3}-\d{2}\b", respuesta)
    if not citados:
        return Veredicto(fiel=False, motivo="No cita ninguna ficha.")
    for identificador in citados:
        if identificador not in FICHAS:
            return Veredicto(fiel=False, motivo=f"{identificador} no existe en el catálogo.")
        ficha = FICHAS[identificador]
        comunes = query_terms(respuesta) & query_terms(ficha["title"] + " " + ficha["text"])
        if not comunes - {identificador.lower()}:
            return Veredicto(fiel=False, motivo=f"La frase no se parece a lo que dice {identificador}.")
    return Veredicto(fiel=True, motivo="Las fichas existen y la frase coincide con su contenido.")


def juez_modelo(respuesta):
    from henry_agents.config import chat_model

    fichas = {i: FICHAS[i]["text"] for i in re.findall(r"\b[A-Z]{3}-\d{2}\b", respuesta) if i in FICHAS}
    instrucciones = "Eres un juez estricto. Decide si la respuesta solo afirma lo que dicen las fichas."
    pedido = f"Respuesta: {respuesta}\nFichas citadas (las inexistentes no aparecen): {fichas}"
    return chat_model().with_structured_output(Veredicto).invoke([("system", instrucciones), ("human", pedido)])


juez = juez_modelo if MODE == "live" else juez_reglas

# %% [markdown]
# **Calibrar** un juez es compararlo con etiquetas puestas por personas. Si no coincide en
# casos claros, su veredicto no sirve para evaluar nada.
#
# 🔮 **Predice:** ¿cuál de estas tres respuestas debería ser fiel?

# %%
etiquetadas = [
    ("Batman compara tres registros del observatorio y descubre un reloj adelantado [BAT-01].", True),
    ("Batman usa un satélite secreto para resolver el caso [BAT-99].", False),
    ("El Chavo vende helados en la playa [BAT-02].", False),
]
aciertos = 0
for respuesta, etiqueta in etiquetadas:
    veredicto = juez(respuesta)
    aciertos += veredicto.fiel == etiqueta
    print("✅" if veredicto.fiel == etiqueta else "❌", veredicto.fiel, "—", veredicto.motivo)
print(f"El juez coincide con las personas en {aciertos} de {len(etiquetadas)} casos.")
if MODE == "offline":
    confirmar(aciertos == 3, "El juez de reglas debía coincidir en los tres casos")

# %% [markdown]
# 🔍 **Observa:** el tercer caso cita una ficha **real** con una afirmación falsa. La
# validación de citas de la primera parte no lo detecta; el juez sí.
#
# ## Un reporte con costo medido
# Evaluamos respuestas completas (recuperar + redactar + juzgar) dentro de `medir_costo()`.
# Offline el costo es cero; en live verás tokens y dólares reales.

# %%
with medir_costo() as medicion:
    reporte = []
    for caso in golden[:5]:
        resultado = search_catalog(caso["query"], universe=caso["universe"], kind=caso["kind"])
        respuesta = compose(resultado, MODE)
        faltan = [f" [{i}]" for i in respuesta.source_ids if i not in respuesta.text]
        texto_con_citas = respuesta.text + "".join(faltan)
        veredicto = juez(texto_con_citas) if respuesta.source_ids else None
        reporte.append({"caso": caso["id"], "citas": respuesta.source_ids,
                        "fiel": veredicto.fiel if veredicto else None})

ruta = ROOT / "reports" / f"evaluacion-clase6-{MODE}.json"
ruta.parent.mkdir(exist_ok=True)
ruta.write_text(json.dumps({"modo": MODE, "casos": reporte, "usd": medicion["usd"]},
                           indent=2, ensure_ascii=False), encoding="utf-8")
print("Reporte guardado en", ruta.relative_to(ROOT))
confirmar(all(fila["citas"] for fila in reporte), "Los cinco casos tienen evidencia y debían citar")

# %% [markdown]
# ## ☕ Pausa

# %% [markdown]
# ## ✏️ Tu turno 1 · Dos casos nuevos
# Completa los dos casos. Uno debe recuperar **CHA-03** (herramientas en El Chavo). El otro debe
# **esperar vacío**: elige un tema que no exista en El Chavo (por ejemplo, "receta").
# Cambia solo los `None`. La celda siguiente te dice si quedó bien.

# %%
caso_herramientas = {"query": "herramientas", "universe": "chavo", "expected": None}  # ✏️ completa
caso_vacio = {"query": None, "universe": "chavo", "expected": []}  # ✏️ completa

# %%
completos = caso_herramientas["expected"] is not None and caso_vacio["query"] is not None
if completos:
    mis_filas = evaluar_casos([caso_herramientas, caso_vacio])
    comprobar(all(f["exacta"] for f in mis_filas), "Tus dos casos pasan.",
              "Mira 'recibido' en cada fila y ajusta lo esperado o el tema.")
else:
    comprobar(False, "", "Reemplaza los dos None antes de comprobar.")

# %%
ver_solucion("06_casos_nuevos")

# %% [markdown]
# ## ✏️ Tu turno 2 · Etiqueta como persona
# Lee esta respuesta y la ficha FAN-02 en el catálogo. Pon `True` si es fiel o `False` si no.
# Después comparamos tu etiqueta con el juez.
#
# > "Reed compara un sensor de temperatura y otro de energía [FAN-02]."

# %%
mi_etiqueta = None  # ✏️ True o False

# %%
respuesta_fan = "Reed compara un sensor de temperatura y otro de energía [FAN-02]."
comprobar(mi_etiqueta is True, "Correcto: la frase repite lo que dice FAN-02.",
          "Busca FAN-02 en el catálogo: ¿habla de dos sensores?")
print("El juez dice:", juez(respuesta_fan))

# %%
ver_solucion("06_etiqueta_juez")

# %% [markdown]
# ## 🧱 Proyecto · Paso 6: evaluación y aprobación
# Agrega a tu Asistente del Archivo:
# 1. Un ciclo evaluador–optimizador con `max_intentos` en el estado y escalación sin evidencia.
# 2. Una pausa de aprobación: muestra una aprobación y un rechazo en hilos distintos.
# 3. Un reporte con al menos 5 casos (uno que espere vacío), el veredicto del juez y el costo.
#
# **Evidencia que guardas:** el JSON de `reports/` y las salidas de aprobar y rechazar.
#
# ## 🎟️ Ticket de salida
# - ¿Por qué el borrador escalado no lleva las citas del último intento?
# - ¿Para qué sirve el `thread_id` al continuar una pausa?
# - Da un ejemplo de respuesta con citas válidas que el juez debería marcar como no fiel.
#
# ## 📖 Glosario de hoy
# | Término | En una frase |
# |---|---|
# | Evaluador–optimizador | Redactar, evaluar con un criterio y corregir con un límite |
# | Escalar | Pasar el caso a una persona en lugar de entregar algo inválido |
# | Checkpoint | Foto del estado del grafo que permite pausar y continuar |
# | `thread_id` | Nombre de una conversación o solicitud para retomarla |
# | `interrupt` / `Command(resume=...)` | Pausar el grafo / continuar con una respuesta |
# | Golden set | Casos con respuesta conocida para medir el comportamiento |
# | Juez (LLM-as-judge) | Revisor que devuelve un veredicto estructurado sobre una respuesta |
# | Calibrar | Comparar el juez con etiquetas humanas antes de confiar en él |
#
# ## Límites de lo que hicimos
# - El juez de reglas solo compara palabras: no entiende el sentido. El juez GPT-6 tampoco es
#   infalible; por eso se calibra.
# - Las pausas viven en memoria: si reinicias el kernel, se pierden.
# - Las aprobaciones son simuladas y no publican nada.
