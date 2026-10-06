"""Solución · Clase 5 · Tu turno 2: tres colecciones → tres delegaciones (una por colección)."""

from henry_agents.cultural import search_catalog
from henry_agents.practica import confirmar

mis_colecciones = ["batman", "fantasticos", "chavo"]

# Lo mismo que hace el supervisor con la regla offline: una delegación por colección pedida,
# mientras no se supere max_delegaciones=3.
hallazgos = []
for coleccion in mis_colecciones[:3]:
    resultado = search_catalog("herramientas", universe=coleccion, top_k=1)
    hallazgos.append({"universe": coleccion, "ids": [h.id for h in resultado.hits]})
    print(f"Delegar en {coleccion}: {hallazgos[-1]['ids']}")

print("Delegaciones:", len(hallazgos))
confirmar(
    [h["ids"] for h in hallazgos] == [["BAT-02"], ["FAN-02"], ["CHA-03"]],
    "Cada colección debía aportar su ficha de herramientas",
)
