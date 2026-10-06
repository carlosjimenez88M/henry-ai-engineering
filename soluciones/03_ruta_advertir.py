"""Clase 3 · Tu turno 1: routing con tres salidas según cuántas fichas hay, conectado a un grafo."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from henry_agents.config import configure
from henry_agents.cultural import SearchResult, compose, search_catalog
from henry_agents.practica import confirmar

MODE = configure()


def mi_ruta(estado):
    cantidad = len(estado["evidence"]["hits"])
    if cantidad == 0:
        return "abstenerse"
    elif cantidad == 1:
        return "advertir"
    else:
        return "responder"


class EstadoRAG(TypedDict, total=False):
    query: str
    universe: str
    evidence: dict
    answer: str
    sources: list[str]
    status: str


def nodo_buscar(estado):
    return {"evidence": search_catalog(estado["query"], universe=estado.get("universe", "todos"), top_k=2).model_dump()}


def nodo_responder(estado):
    respuesta = compose(SearchResult.model_validate(estado["evidence"]), MODE)
    estado_final = "answered" if respuesta.source_ids else "abstained"
    return {"answer": respuesta.text, "sources": respuesta.source_ids, "status": estado_final}


def nodo_advertir(estado):
    respuesta = nodo_responder(estado)
    return {**respuesta, "answer": "⚠️ Evidencia escasa (una ficha). " + respuesta["answer"], "status": "warned"}


def nodo_abstenerse(estado):
    return {"answer": "No hay evidencia en el catálogo.", "sources": [], "status": "abstained"}


casos = {n: mi_ruta({"evidence": {"hits": [{}] * n}}) for n in (0, 1, 3)}
print("Rutas por cantidad de fichas:", casos)

grafo = StateGraph(EstadoRAG)
for nombre, funcion in [("buscar", nodo_buscar), ("responder", nodo_responder),
                        ("advertir", nodo_advertir), ("abstenerse", nodo_abstenerse)]:
    grafo.add_node(nombre, funcion)
grafo.add_edge(START, "buscar")
grafo.add_conditional_edges("buscar", mi_ruta, ["responder", "advertir", "abstenerse"])
for final in ["responder", "advertir", "abstenerse"]:
    grafo.add_edge(final, END)
app = grafo.compile()
recorrido = [list(p)[0] for p in app.stream({"query": "cooperación", "universe": "chavo"}, stream_mode="updates")]
print("Recorrido para 'cooperación' en chavo:", recorrido)

confirmar(casos == {0: "abstenerse", 1: "advertir", 3: "responder"}, "La regla cubre los tres casos")
confirmar(recorrido == ["buscar", "advertir"], "Una sola ficha debía llevar a advertir")
