"""Clase 3 · Tu turno 2: el plan de una actividad de cooperación, sin workers vacíos."""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from henry_agents.cultural import search_catalog
from henry_agents.practica import confirmar

PERMITIDAS = {"batman", "fantasticos", "chavo", "canciones"}


class EstadoEquipo(TypedDict, total=False):
    query: str
    universes: list[str]
    parts: Annotated[list[dict], operator.add]
    summary: str


def planificar(estado):
    desconocidas = sorted(set(estado["universes"]) - PERMITIDAS)
    if not estado["universes"] or desconocidas:
        raise ValueError(f"Plan inválido: {desconocidas or 'vacío'}")
    return {"universes": sorted(set(estado["universes"]))}


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

# 1) Explorar con las cuatro colecciones: batman vuelve vacía para cooperación.
exploracion = app_equipo.invoke({"query": "cooperación", "universes": sorted(PERMITIDAS), "parts": []})
print("Exploración:\n" + exploracion["summary"])
utiles = [p["universe"] for p in exploracion["parts"] if p["ids"]]

# 2) El plan final solo con las colecciones que aportan algo.
resultado = app_equipo.invoke({"query": "cooperación", "universes": utiles, "parts": []})
print("Plan final:\n" + resultado["summary"])
confirmar(sorted(utiles) == ["canciones", "chavo", "fantasticos"], "Tres colecciones con cooperación")
confirmar(all(p["ids"] for p in resultado["parts"]), "Ningún worker debía volver vacío")
