"""Solución · Clase 6 · Tu turno 2: etiquetar una respuesta como persona."""

from henry_agents.cultural import load_catalog
from henry_agents.practica import confirmar

ficha = next(f for f in load_catalog() if f["id"] == "FAN-02")
print("FAN-02 dice:", ficha["text"])

# La respuesta afirma que Reed compara un sensor de temperatura y otro de energía.
# La ficha dice exactamente eso: la respuesta es fiel.
mi_etiqueta = True
print("Mi etiqueta:", mi_etiqueta)
confirmar("temperatura" in ficha["text"] and "energía" in ficha["text"], "La ficha debía mencionar ambos sensores")
