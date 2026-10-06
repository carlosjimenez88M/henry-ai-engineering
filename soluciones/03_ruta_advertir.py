"""Clase 3 · Tu turno 1: routing con tres salidas según cuántas fichas hay."""

from henry_agents.practica import confirmar


def mi_ruta(estado):
    cantidad = len(estado["evidence"]["hits"])
    if cantidad == 0:
        return "abstenerse"
    elif cantidad == 1:
        return "advertir"
    else:
        return "responder"


casos = {n: mi_ruta({"evidence": {"hits": [{}] * n}}) for n in (0, 1, 3)}
print("Rutas por cantidad de fichas:", casos)
confirmar(casos == {0: "abstenerse", 1: "advertir", 3: "responder"}, "La regla cubre los tres casos")
