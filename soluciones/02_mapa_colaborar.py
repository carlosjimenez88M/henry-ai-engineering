"""Clase 2 · Tu turno 2: ubicar "colaborar" en el mapa de significados."""

from henry_agents.practica import confirmar
from henry_agents.semantica import MAPA, buscar_por_significado

# (misterio, colaboración, tecnología): colaborar es casi pura colaboración.
MAPA["colaborar"] = (0.05, 0.9, 0.05)
resultado = buscar_por_significado("colaborar", mode="offline")
print("Más parecidas a 'colaborar':", resultado)
confirmar(
    resultado[0][1] in {"MUS-01", "CHA-01", "FAN-01", "FAN-03"},
    "La primera ficha debía ser de equipo o cooperación",
)
