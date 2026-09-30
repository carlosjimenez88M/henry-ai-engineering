# %% [markdown]
# # Clase 4 · Revisar antes de entregar: ciclos, aprobación humana y evaluación
#
# El sistema ya busca y coordina tareas. Ahora debe reconocer fallas, intentar una
# corrección acotada y dejar una propuesta lista para que una persona la revise.
#
# **Producto:** un ciclo de revisión, una aprobación/rechazo reanudable y un reporte
# local. No publicamos nada ni conectamos servicios externos de observabilidad.
#
# **Recorrido de la clase**
#
# - Detectar una respuesta con fuente inventada
# - Definir qué significa “aprobada por el evaluador”
# - Construir borrador y evaluación separados
# - Cerrar el ciclo con límite y observar una corrección
# - Pausa
# - Distinguir checkpoint de memoria durable
# - Interrumpir, inspeccionar y reanudar con una decisión humana
# - Probar rechazo y aislamiento de solicitudes
# - Pausa
# - Taller: diseñar y ejecutar casos de evaluación
# - Diagnosticar una regresión y guardar evidencia
# - Defender el proyecto final y sus límites
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
# reglas/extractos explícitos. Live usa OpenAI y consume API. El estudiante puede
# hacer toda la práctica offline; el docente demuestra live. No compartas el .env.

# %% [markdown]
# ## El error que una respuesta bonita puede esconder
# Una respuesta dice que una ficha inexistente respalda su conclusión. El texto es
# fluido, el JSON es válido y el programa terminó sin excepciones. ¿Está listo?
# La pregunta correcta es qué contrato falló y dónde lo podemos observar.

# %%
from henry_agents.config import ROOT, configure
from henry_agents.cultural import SearchResult, compose, search_catalog

MODE = configure()
fichas = search_catalog("herramientas", universe="batman", top_k=1)
borrador_roto = {"text": "Un informe sin fuente comprobable", "source_ids": ["INVENTADA"]}
print("Disponibles:", [h.id for h in fichas.hits])
print("Citadas:", borrador_roto["source_ids"])

# %% [markdown]
# ## Arquitectura 6: evaluador–optimizador
# Separamos producir un borrador de evaluar un criterio. Si falla, podemos corregir
# con un límite. Un loop sin criterio verificable puede consumir recursos sin mejorar.
#
# ```text
# START → buscar → redactar → evaluar ─ válido ─→ listo
#                     ↑         │
#                     └─ corregir (quedan intentos)
#                               └─ sin evidencia o límite ─→ escalar
# ```
#
# Nuestro evaluador comprueba IDs existentes, no toda la fidelidad semántica. Para
# calidad real sumaríamos revisión humana o una evaluación semántica contrastada.
# No vamos a llamar “verdadera” a una respuesta por pasar una comprobación de referencias.
#
# **Actividad:** propongan dos criterios diferentes: uno que Python pueda comprobar
# y otro que requiera leer el contenido. Anoten qué fallos puede detectar cada uno.


# %%
def ids_validos(borrador, evidencia):
    disponibles = {h["id"] for h in evidencia["hits"]}
    citados = set(borrador["source_ids"])
    return bool(citados) and citados <= disponibles


assert not ids_validos(borrador_roto, fichas.model_dump())
print("El evaluador detecta la referencia inventada.")

# %% [markdown]
# ## Producir y evaluar son nodos distintos
# Nuestro estado conservará evidence, draft, attempts y valid. Conservar la evidencia
# original permite revisar el borrador sin volver a buscar en cada iteración.
#
# Inyectamos un error únicamente en el primer borrador para ver el ciclo de manera
# reproducible. No fingimos que el LLM cometió ese error espontáneamente. En el segundo
# intento vuelve a generar desde la evidencia original.
# `MAX_INTENTOS` limita borradores por solicitud. Su valor se usa en la decisión:
# con dos hay oportunidad de corregir; con uno el primer borrador inválido escala.

# %%
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

MAX_INTENTOS = 2


class EstadoRevision(TypedDict, total=False):
    query: str
    evidence: dict
    draft: dict
    attempts: int
    valid: bool
    decision: str
    fault: bool


def recuperar(estado):
    return {"evidence": search_catalog(estado["query"], top_k=2).model_dump()}


# %%
def redactar(estado):
    intento = estado.get("attempts", 0) + 1
    respuesta = compose(SearchResult.model_validate(estado["evidence"]), MODE).model_dump()
    if estado.get("fault") and intento == 1:
        respuesta["source_ids"] = ["INVENTADA"]
    return {"draft": respuesta, "attempts": intento}


def evaluar(estado):
    return {"valid": ids_validos(estado["draft"], estado["evidence"])}


def decidir(estado):
    if estado["valid"]:
        return "listo"
    if not estado["evidence"]["hits"] or estado["attempts"] >= MAX_INTENTOS:
        return "escalar"
    return "redactar"


# %% [markdown]
# **Leé el orden de decidir:** primero acepta algo válido; luego evita repetir sin
# evidencia o después del límite; solo entonces corrige. Reintentar no es siempre
# la acción adecuada. Si falta información, repetir el prompt no la crea.
#
# ## Conectamos el ciclo y observamos dos intentos
# La escalación elimina el borrador inválido de la salida destinada al usuario.
# No lo entregamos como si fuera correcto simplemente porque agotamos los intentos.


# %%
def listo(estado):
    return {"decision": "ready"}


def escalar(estado):
    return {
        "decision": "escalate",
        "draft": {"text": "Revisión humana necesaria.", "source_ids": []},
    }


revision = StateGraph(EstadoRevision)
for nombre, funcion in [
    ("recuperar", recuperar),
    ("redactar", redactar),
    ("evaluar", evaluar),
    ("listo", listo),
    ("escalar", escalar),
]:
    revision.add_node(nombre, funcion)
revision.add_edge(START, "recuperar")
revision.add_edge("recuperar", "redactar")
revision.add_edge("redactar", "evaluar")
revision.add_conditional_edges("evaluar", decidir, ["listo", "escalar", "redactar"])
revision.add_edge("listo", END)
revision.add_edge("escalar", END)
app_revision = revision.compile()

# %%
corregido = app_revision.invoke(
    {"query": "investigación Batman", "attempts": 0, "fault": True}, config={"recursion_limit": 15}
)
assert corregido["attempts"] == 2
assert corregido["valid"]
assert corregido["decision"] == "ready"
print("Intentos:", corregido["attempts"], "| decisión:", corregido["decision"])
print(corregido["draft"])

# %% [markdown]
# **Experimento de límite:** conservamos la misma falla y reducimos el presupuesto
# a un intento. Anticipá si debería entregar el borrador o pedir revisión. Después
# restauramos el valor para continuar la clase con las mismas condiciones.

# %%
MAX_INTENTOS = 1
agotado = app_revision.invoke({"query": "investigación Batman", "attempts": 0, "fault": True})
assert agotado["attempts"] == 1
assert agotado["decision"] == "escalate"
assert agotado["draft"]["source_ids"] == []
MAX_INTENTOS = 2
print("Sin intentos disponibles:", agotado["decision"])

# %% [markdown]
# **Punto de reenganche 1:** podés localizar la condición que vuelve a redactar y la
# que obliga a terminar. MAX_INTENTOS es un límite de nuestra lógica; recursion_limit
# protege los pasos del grafo. Ninguno equivale a un presupuesto monetario.
#
# ## Pausa
#
# ## Una persona revisa la propuesta
# Inspiración: Alfred revisa el equipo propuesto antes de una salida. En esta demo
# solo revisamos una respuesta: no publicamos, compramos ni enviamos nada.
#
# Un checkpoint guarda estado de ejecución. thread_id identifica la solicitud que
# queremos continuar. InMemorySaver sirve en el proceso actual: **no sobrevive a un
# reinicio** y no es una base de datos compartida entre servidores.
#
# interrupt devuelve control al llamador; Command(resume=...) aporta la decisión.
# Al reanudar, el nodo interrumpido comienza otra vez: no pongas efectos externos
# antes de interrupt sin diseñar idempotencia.

# %%
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt


def revisar_con_persona(estado):
    solicitud = {"proposal": estado["draft"], "options": ["approve", "reject"]}
    while True:
        decision = interrupt(solicitud)
        if decision in ("approve", "reject"):
            return {"decision": decision}
        solicitud = {**solicitud, "error": "Elegí approve o reject; la solicitud sigue pendiente."}


# %% [markdown]
# ## Pausar no es bloquear el teclado
# Construimos un grafo pequeño para aislar la revisión. Recibe el borrador listo de
# la etapa anterior. Separar esta demo facilita entender el checkpoint antes de
# combinarlo con el ciclo completo.

# %%
humano = StateGraph(EstadoRevision)
humano.add_node("revisar", revisar_con_persona)
humano.add_edge(START, "revisar")
humano.add_edge("revisar", END)
app_humana = humano.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": str(uuid4())}}
pausado = app_humana.invoke({"draft": corregido["draft"]}, config)
assert pausado.get("__interrupt__")
print("Propuesta pendiente:", pausado["__interrupt__"][0].value)
print("Próximo nodo:", app_humana.get_state(config).next)

# %% [markdown]
# **Antes de reanudar:** una persona de la pareja lee la propuesta y explica qué
# verificó. La próxima celda usa una decisión simulada para que la demo se pueda
# ejecutar completa. En una aplicación real debe venir de una persona autenticada,
# no de una aprobación automática del modelo.

# %%
aprobado = app_humana.invoke(Command(resume="approve"), config)
assert aprobado["decision"] == "approve"
assert app_humana.get_state(config).next == ()
print("Decisión guardada:", aprobado["decision"])

# %% [markdown]
# ## Probar el rechazo es obligatorio
# Una segunda solicitud usa otro thread_id. Rechazarla no debe cambiar la aprobación
# anterior. No confundimos la cadena "reject" con True por ser texto no vacío.
# Primero enviamos "tal vez", que no es una opción permitida. La solicitud vuelve a
# interrumpirse y permite corregir la decisión. El while no hace llamadas automáticas:
# cada interrupt devuelve el control y espera una nueva respuesta de la persona.

# %%
otra_config = {"configurable": {"thread_id": str(uuid4())}}
app_humana.invoke({"draft": corregido["draft"]}, otra_config)
decision_invalida = app_humana.invoke(Command(resume="tal vez"), otra_config)
assert decision_invalida.get("__interrupt__")
assert any(tarea.interrupts for tarea in app_humana.get_state(otra_config).tasks)
print("Podemos corregir la entrada:", decision_invalida["__interrupt__"][0].value["error"])
rechazado = app_humana.invoke(Command(resume="reject"), otra_config)
assert rechazado["decision"] == "reject"
assert app_humana.get_state(config).values["decision"] == "approve"
print("Solicitudes aisladas: una aprobada y otra rechazada.")

# %% [markdown]
# **Punto de reenganche 2:** podés explicar para qué sirve el mismo thread_id al
# reanudar y por qué una nueva solicitud necesita otro identificador.
#
# ## Pausa
#
# ## Taller: diseñar pruebas antes de cambiar el prompt
# Una tabla de casos es un contrato de comportamiento. No basta con las preguntas
# que sabemos que funcionan. El conjunto necesita filtros, acentos y falta de evidencia.
#
# La tabla inicial tiene cuatro casos. Agregá un caso de canciones a tu copia,
# con sus IDs esperados, y ejecutala. La solución completa está en el siguiente
# bloque. No cambies las expectativas para esconder una falla sin entender la causa.
#
# **Pista 1:** investigación en canciones debería recuperar MUS-02.
# **Pista 2:** una expectativa vacía debe exigir salida vacía; no basta comprobar
# que un conjunto vacío sea subconjunto de cualquier resultado.
# **Entrega:** tabla con resultado y una hipótesis sobre cualquier diferencia.

# %%
casos_base = [
    {"query": "herramientas", "universe": "batman", "expected": ["BAT-02"]},
    {"query": "cooperación", "universe": "fantasticos", "expected": ["FAN-01", "FAN-03"]},
    {"query": "recado", "universe": "chavo", "expected": ["CHA-02"]},
    {"query": "vacuna marciana", "universe": "todos", "expected": []},
]
mis_casos = casos_base.copy()  # Agregá tu quinto diccionario a esta lista.

# %%
from time import perf_counter


def evaluar_casos(casos):
    filas = []
    for caso in casos:
        inicio = perf_counter()
        recuperado = search_catalog(caso["query"], universe=caso["universe"], top_k=3)
        obtenidos = {h.id for h in recuperado.hits}
        esperados = set(caso["expected"])
        correcto = obtenidos == esperados
        filas.append(
            {
                "query": caso["query"],
                "expected": sorted(esperados),
                "received": sorted(obtenidos),
                "ok": correcto,
                "latency_ms": round((perf_counter() - inicio) * 1000, 3),
            }
        )
        print(caso["query"], "→", "PASA" if correcto else "REVISAR", sorted(obtenidos))
    return filas


mi_reporte = evaluar_casos(mis_casos)

# %% [markdown]
# ## Solución, regresión y evidencia local
# El reporte mide coincidencia exacta de IDs en estos casos. No mide comprensión
# universal ni calidad de cada frase. La latencia es local; no incluye un LLM porque
# este conjunto está evaluando la herramienta.
#
# Para detectar una regresión del filtro, probamos una consulta de Batman en la
# colección equivocada y observamos qué cambia. Una señal útil identifica el primer
# componente que falló; no solo imprime “algo salió mal”.
# Comparamos los cinco casos de la solución con tu tabla. ¿Tu caso nuevo tendría
# una expectativa diferente? Explicá por qué antes de cambiarla.

# %%
casos_solucion = [
    *casos_base,
    {"query": "investigación", "universe": "canciones", "expected": ["MUS-02"]},
]
reporte = evaluar_casos(casos_solucion)
assert len(reporte) == 5
assert all(fila["ok"] for fila in reporte)
filtro_equivocado = search_catalog("herramientas", universe="canciones")
assert filtro_equivocado.hits == []
sin_evidencia = app_revision.invoke({"query": "vacuna marciana", "attempts": 0})
assert sin_evidencia["decision"] == "escalate"
assert sin_evidencia["attempts"] == 1
assert sin_evidencia["draft"]["source_ids"] == []
print("La falta de evidencia escala sin bucle inútil.")

# %% [markdown]
# El paquete incluye build_review_graph para integrar el ciclo y la aprobación en
# una sola aplicación. Las pruebas automáticas verifican rechazo, agotamiento de
# intentos e interrupciones. Aquí guardamos la evidencia del trabajo realizado.
# Los reportes quedan fuera de Git; el corpus docente sí se versiona.

# %%
import json

carpeta = ROOT / "reports"
carpeta.mkdir(exist_ok=True)
ruta = carpeta / f"cultural-evaluation-{MODE}.json"
ruta.write_text(
    json.dumps(
        {
            "mode": MODE,
            "cases": reporte,
            "review_attempts": corregido["attempts"],
            "approved": aprobado["decision"],
            "rejected": rechazado["decision"],
        },
        indent=2,
        ensure_ascii=False,
    )
)
print("Reporte guardado:", ruta.name)

# %% [markdown]
# ## Defensa del proyecto
# Prepará una explicación breve con evidencia visible:
# 1. El contrato de tu herramienta y un argumento inválido que rechace.
# 2. La arquitectura elegida y una alternativa que descartaste con una razón.
# 3. Una fuente real del catálogo y una consulta sin evidencia.
# 4. Un ciclo que termine, una aprobación y un rechazo.
# 5. Un reporte con al menos cinco casos; agregá dos más como trabajo posterior.
#
# **Rúbrica:** contrato y pruebas de herramienta 25%; elección/implementación de
# arquitectura 30%; fuentes y abstención 20%; evaluación y revisión humana 25%.
# Se evalúa explicar y comprobar, no recordar sintaxis ni terminar primero.
#
# **Crítica final:** aún no tenemos autenticación, almacenamiento durable ni una
# evaluación semántica completa. Son requisitos para otro alcance, no promesas de
# esta demo. Tampoco hicimos un buscador de todos los cómics: construimos un sistema
# acotado que permite ver dónde decide, dónde busca y dónde se detiene.
#
# Referencias técnicas del módulo: [patrones](https://docs.langchain.com/oss/python/langgraph/workflows-agents),
# [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) e
# [interrupciones](https://docs.langchain.com/oss/python/langgraph/interrupts).
# Los ejemplos y actividades de estas clases fueron diseñados para este repositorio.
