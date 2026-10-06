"""Solución · Clase 5 · Tu turno 2: tres colecciones con el supervisor real.

Reconstruimos el mismo grafo de la clase (regla offline) para que este archivo sea autocontenido.
Con max_delegaciones=3 y tres colecciones pendientes, el supervisor delega tres veces.
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
        return Command(goto="redactar", update={"bitacora": ["Redactar: ya alcanza"]})
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

mis_colecciones = ["batman", "fantasticos", "chavo"]
mi_prediccion_tres = 3  # tres colecciones pendientes y el límite permite tres

tres = grafo.compile().invoke({"tema": "herramientas", "necesita": mis_colecciones, "max_delegaciones": 3})
for linea in tres["bitacora"]:
    print("🧭", linea)
print("Hallazgos:", tres["hallazgos"])
confirmar(tres["delegaciones"] == mi_prediccion_tres, "Debían ser tres delegaciones")
confirmar(
    [h["ids"] for h in tres["hallazgos"]] == [["BAT-02"], ["FAN-02"], ["CHA-03"]],
    "Cada colección debía aportar su ficha de herramientas",
)
