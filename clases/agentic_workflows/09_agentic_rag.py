# %% [markdown]
# # 09 · Agentic RAG: decidir la próxima búsqueda
#
# **Meta:** construir un agente que mire la evidencia antes de decidir si busca
# otra vez, consulta el catálogo, responde o se abstiene.
# **Necesitas:** clase 08 y poder leer un grafo pequeño. **Producto:** un LangGraph
# con decisiones observables, revisión de evidencia y presupuesto de consultas.
#
# RAG clásico: una búsqueda y una respuesta/abstención. Agentic RAG: un modelo
# puede decidir qué fuente consultar, reformular, completar una pregunta compuesta
# y detenerse. Poner una búsqueda dentro de un grafo no basta para hacerlo agéntico.
#
# **Offline:** el grafo y las herramientas son reales; una política de reglas sustituye
# al LLM y copia extractos. Es una simulación de control, no comprensión general.
# **Live:** GPT-6.1 Sol decide; GPT-6 Luna genera y revisa. Los nombres son
# configurables y se comprobaron en la documentación oficial del 6/10/2026.

# %%
from henry_agents.rag import (
    DecisionRAG,
    EstadoAgenticRAG,
    crear_agente_rag,
    crear_nodos_rag,
    estado_inicial,
    rag_clasico,
    ruta_rag,
)

pregunta = "¿Hacen mandados?"  # Aquí significa llevar compras; no está en el texto del manual.

# %% [markdown]
# ## 1. Compara con una búsqueda fija
# **Predice:** la búsqueda lexical no reconoce "mandados" como "envío". ¿Serviría
# pedir que el redactor sea más amable? Necesitamos mejorar recuperación.

# %%
base = rag_clasico(pregunta, k=1)
assert base["estado"] == "abstencion" and base["evidencia"] == []
print("RAG clásico:", base["estado"])

# %% [markdown]
# ## 2. Lee el estado antes de conectar el grafo
# `pregunta` conserva el pedido original. `necesidades` representa qué falta cubrir;
# en esta práctica se obtiene por una heurística visible o se define a mano.
# `evidencia` acumula resultados por ID sin duplicarlos. `eventos` registra acciones
# y observaciones; no es razonamiento privado del modelo.

# %%
entrada = estado_inicial(pregunta)
print(entrada)
assert entrada["busquedas"] == 0
assert entrada["evaluacion"]["faltantes"][0]["categoria"] == "entregas"

# %% [markdown]
# ## 3. Construye el mapa con sus nodos reales
# Cada nodo devuelve cambios en el estado. El programa valida la decisión antes
# de llamar una fuente. La recuperación vuelve a evaluación y luego a decisión.

# %%
from langgraph.graph import END, START, StateGraph

nodos = crear_nodos_rag(mode="offline", k=1, max_busquedas=3, max_decisiones=4)
constructor = StateGraph(EstadoAgenticRAG)
for nombre, funcion in nodos.items():
    constructor.add_node(nombre, funcion)
constructor.add_edge(START, "decidir")
constructor.add_conditional_edges("decidir", ruta_rag, ["recuperar", "responder", "abstenerse"])
constructor.add_edge("recuperar", "evaluar")
constructor.add_edge("evaluar", "decidir")
constructor.add_edge("responder", END)
constructor.add_edge("abstenerse", END)
app = constructor.compile()
print(app.get_graph().draw_mermaid())  # Texto local; no servicio web de imágenes.

# %% [markdown]
# ## Primera pausa · Localiza las decisiones
# Encuentra `politica_offline` en `src/henry_agents/rag.py`. Lee la regla que cambia
# una consulta tras ver la evidencia faltante. No viene de un modelo entrenado.
# El mismo registro de acciones se valida cuando la decisión viene de un LLM.
#
# ## 4. Observa la reformulación y conserva la pregunta original
# **Predice:** primera búsqueda vacía → reformular → recuperar → responder.
# Comparamos `k=1` en ambas arquitecturas para no atribuir al agente una ventaja
# que venga simplemente de darle más resultados por búsqueda.

# %%
resultado = app.invoke(estado_inicial(pregunta))
for evento in resultado["eventos"]:
    print(evento)
assert resultado["pregunta"] == pregunta
assert resultado["busquedas"] == 2
assert resultado["estado"] == "respondido"
assert resultado["llamadas_llm"] == 0
print(resultado["respuesta"])

# %% [markdown]
# ## 5. Una pregunta compuesta puede necesitar dos recuperaciones
# ¿El envío a Centro y el retiro tienen la misma política? No. `k=1` solo entrega
# un candidato por búsqueda. El evaluador indica qué necesidad sigue faltando.

# %%
compuesta = "¿Cuánto cuesta el envío a Centro y cómo retiro mi pedido?"
assert rag_clasico(compuesta, k=1)["estado"] == "abstencion"
equipo = crear_agente_rag(k=1)
completa = equipo.invoke(estado_inicial(compuesta))
assert completa["estado"] == "respondido" and completa["busquedas"] == 2
assert set(completa["fuentes"]) == {"ENVIO-2026-c01", "RETIRO-2026-c01"}
print(completa["respuesta"])

# %% [markdown]
# ## 6. Elegir otra herramienta: el stock no viene del manual
# Para stock y precio usa el catálogo exacto de clase 01. La herramienta devuelve
# una observación con ID; no confirma reservas. El corpus y catálogo son snapshots ficticios.

# %%
con_stock = crear_agente_rag(k=1).invoke(estado_inicial("¿Hay stock de arroz y envío a Centro?"))
assert con_stock["estado"] == "respondido"
assert "CAT-ARROZ" in con_stock["fuentes"]
assert any(e.get("accion") == "catalogo" for e in con_stock["eventos"])
print(con_stock["respuesta"])

# %% [markdown]
# ## Segunda pausa · Taller de fallas
# Prueba tres casos antes de mirar la solución: presupuesto de una búsqueda,
# tema sin fuente y decisor que insiste en repetir lo mismo. ¿Qué debe devolverse?
# **Pista:** agotar un límite no crea evidencia. Repetir una consulta idéntica
# se detecta antes de ejecutarla otra vez.
#
# ## Solución: límites y falta de progreso

# %%
limitado = crear_agente_rag(k=1, max_busquedas=1).invoke(estado_inicial(pregunta))
assert limitado["estado"] == "abstencion" and limitado["causa"] == "presupuesto_agotado"
desconocido = crear_agente_rag().invoke(estado_inicial("¿Qué garantía tiene un televisor?"))
assert desconocido["estado"] == "abstencion" and desconocido["fuentes"] == []


def decisor_atascado(estado):
    return DecisionRAG(accion="buscar", consulta="unicornios", motivo="Falla inyectada: repetir")


atascado = crear_agente_rag(decisor=decisor_atascado).invoke(estado_inicial(pregunta))
assert atascado["causa"] == "sin_progreso" and atascado["busquedas"] == 1
print("Límite:", limitado["causa"], "| desconocido:", desconocido["estado"],
      "| repetición:", atascado["causa"])

# %% [markdown]
# ## Experimento opcional con LLM: ahora el modelo elige la acción
# `mode="live"` conserva recuperación lexical para poder atribuir los cambios
# al decisor. Opcionalmente puedes pasar un `IndiceVectorial` de clase 08 y comparar
# después esa segunda variable. No cambia todo a la vez.
#
# Máximo: cuatro decisiones + una generación + una revisión de fidelidad = seis
# invocaciones de chat, y hasta tres consultas a fuentes. Reintentos de transporte
# del SDK pueden agregar solicitudes; estos límites no equivalen a un tope de facturación.
# El juez LLM es una señal adicional, no una prueba absoluta. Errores de API se propagan.

# %%
USAR_AGENTIC_RAG_REAL = False
if USAR_AGENTIC_RAG_REAL:
    from henry_agents.config import medir_costo

    with medir_costo() as medicion:
        real = crear_agente_rag(mode="live", k=1).invoke(estado_inicial(pregunta))
    print(real["estado"], real["respuesta"], "invocaciones de chat:", real["llamadas_llm"])
    print("Uso registrado:", medicion)
else:
    print("Live desactivado. El grafo completo se ejecutó sin consumir API.")

# %% [markdown]
# ## Proyecto y ticket de salida
# Sigue [el proyecto RAG](../../proyectos/tienda_rag/README.md): misma pregunta,
# mismo índice/k, comparar calidad y consultas; añadir dos casos propios.
# ¿Cuándo basta una búsqueda? ¿Qué aporta el agente y qué costo añade? ¿Qué parte
# de la evaluación todavía depende de criterios manuales o de un juez imperfecto?
#
# Documentación y límites: [RAG y Agentic RAG](../../docs/RAG_Y_AGENTIC_RAG.md).
