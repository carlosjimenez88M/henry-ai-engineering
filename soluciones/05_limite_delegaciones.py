"""Solución · Clase 5 · Tu turno 1: con max_delegaciones=1 hay UNA delegación.

El supervisor delega en batman; al volver, el límite ya está alcanzado y el código manda a
redactar aunque falte canciones. El informe queda incompleto, y la bitácora dice por qué.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from henry_agents.cultural import search_catalog
from henry_agents.practica import confirmar


class Estado(TypedDict, total=False):
    tema: str
    necesita: list[str]
    asignada: str
    hallazgos: Annotated[list[dict], operator.add]
    bitacora: Annotated[list[str], operator.add]
    delegaciones: int
    max_delegaciones: int


def supervisor(estado) -> Command[Literal["especialista", "redactar"]]:
    hechas = estado.get("delegaciones", 0)
    if hechas >= estado["max_delegaciones"]:
        return Command(goto="redactar", update={"bitacora": ["Límite de delegaciones alcanzado"]})
    cubiertas = {h["universe"] for h in estado.get("hallazgos", [])}
    faltan = [u for u in estado["necesita"] if u not in cubiertas]
    if not faltan:
        return Command(goto="redactar", update={"bitacora": ["Redactar"]})
    return Command(
        goto="especialista",
        update={"asignada": faltan[0], "delegaciones": hechas + 1, "bitacora": [f"Delegar en {faltan[0]}"]},
    )


def especialista(estado):
    resultado = search_catalog(estado["tema"], universe=estado["asignada"], top_k=1)
    return {"hallazgos": [{"universe": estado["asignada"], "ids": [h.id for h in resultado.hits]}]}


grafo = StateGraph(Estado)
grafo.add_node("supervisor", supervisor)
grafo.add_node("especialista", especialista)
grafo.add_node("redactar", lambda estado: {})
grafo.add_edge(START, "supervisor")
grafo.add_edge("especialista", "supervisor")
grafo.add_edge("redactar", END)

mi_prediccion = 1
corto = grafo.compile().invoke(
    {"tema": "investigación", "necesita": ["batman", "canciones"], "max_delegaciones": 1}
)
print("Predicción:", mi_prediccion, "| real:", corto["delegaciones"])
print("Bitácora:", corto["bitacora"])
confirmar(corto["delegaciones"] == mi_prediccion, "Con límite 1 debía haber una sola delegación")
