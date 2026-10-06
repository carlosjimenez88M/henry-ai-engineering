"""Solución · Clase 1 · El pedido del DJ.

Los valores de universe y kind van sin tilde porque así los define el contrato (Literal).
"""

from henry_agents.cultural import buscar_archivo
from henry_agents.practica import confirmar

entrada_dj = {
    "query": "investigación",  # el tema
    "universe": "canciones",  # solo la colección de canciones
    "kind": "cancion",  # solo canciones, nunca fichas
    "top_k": 2,
}
resultado_dj = buscar_archivo.invoke(entrada_dj)
print([(f["id"], f["title"]) for f in resultado_dj["hits"]])
confirmar([f["id"] for f in resultado_dj["hits"]] == ["MUS-02"], "Debía encontrar solo MUS-02")
