"""Solución · Clase 1 · Filtrar por colección."""

from henry_agents.cultural import load_catalog
from henry_agents.practica import confirmar


def buscar_en_coleccion(palabra, coleccion, fichas):
    encontradas = []
    for ficha in fichas:
        temas = " ".join(ficha["tags"])
        es_de_la_coleccion = ficha["universe"] == coleccion  # la condición que faltaba
        if palabra in temas and es_de_la_coleccion:
            encontradas.append(ficha["id"])
    return encontradas


resultado = buscar_en_coleccion("investigacion", "batman", load_catalog())
print(resultado)
confirmar(resultado == ["BAT-01", "BAT-03"], "Debían quedar solo BAT-01 y BAT-03")
