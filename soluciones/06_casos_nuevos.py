"""Solución · Clase 6 · Tu turno 1: dos casos nuevos para el golden set."""

from henry_agents.cultural import search_catalog
from henry_agents.practica import confirmar

caso_herramientas = {"query": "herramientas", "universe": "chavo", "expected": ["CHA-03"]}
caso_vacio = {"query": "receta", "universe": "chavo", "expected": []}

for caso in [caso_herramientas, caso_vacio]:
    recibido = sorted(h.id for h in search_catalog(caso["query"], universe=caso["universe"]).hits)
    print(f"{caso['query']:12} esperado={caso['expected']} recibido={recibido}")
    confirmar(recibido == sorted(caso["expected"]), f"El caso '{caso['query']}' debía pasar")

# Un caso que espera vacío obliga a comprobar que la salida ESTÉ vacía:
# no alcanza con revisar que lo esperado esté "dentro" de lo recibido.
