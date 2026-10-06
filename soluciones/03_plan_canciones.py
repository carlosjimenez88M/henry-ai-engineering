"""Clase 3 · Tu turno 2: el mismo orquestador con tres colecciones en el plan."""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from henry_agents.cultural import search_catalog
from henry_agents.practica import confirmar


class EstadoEquipo(TypedDict, total=False):
    query: str
    universes: list[str]
    parts: Annotated[list[dict], operator.add]
    summary: str


def planificar(estado):
    pedidas = estado["universes"]
    if not pedidas or not set(pedidas) <= {"batman", "fantasticos", "chavo", "canciones"}:
        raise ValueError("Plan vacío o colección desconocida")
    return {"universes": sorted(set(pedidas))}


def repartir(estado):
    return [Send("worker", {"query": estado["query"], "universe": u}) for u in estado["universes"]]


def worker(estado):
    hits = search_catalog(estado["query"], universe=estado["universe"], top_k=1).hits
    return {"parts": [{"universe": estado["universe"], "ids": [h.id for h in hits]}]}


def reunir(estado):
    partes = sorted(estado["parts"], key=lambda parte: parte["universe"])
    return {"summary": "\n".join(f"{p['universe']}: {p['ids']}" for p in partes)}


grafo = StateGraph(EstadoEquipo)
grafo.add_node("planificar", planificar)
grafo.add_node("worker", worker)
grafo.add_node("reunir", reunir)
grafo.add_edge(START, "planificar")
grafo.add_conditional_edges("planificar", repartir, ["worker"])
grafo.add_edge("worker", "reunir")
grafo.add_edge("reunir", END)
app_equipo = grafo.compile()

mi_plan = {"query": "cooperación", "universes": ["fantasticos", "chavo", "canciones"], "parts": []}
resultado = app_equipo.invoke(mi_plan)
print(resultado["summary"])
confirmar(len(resultado["parts"]) == 3, "Tres colecciones debían crear tres workers")
