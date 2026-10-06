"""Clase 2 · Tu turno 1: encontrar las citas que no están en la evidencia."""

from henry_agents.cultural import GroundedAnswer, search_catalog
from henry_agents.practica import confirmar


def citas_inventadas(respuesta, evidencia):
    disponibles = {ficha.id for ficha in evidencia.hits}
    citadas = set(respuesta.source_ids)
    return citadas - disponibles  # lo citado que NO recuperamos


evidencia = search_catalog("investigación", universe="batman", top_k=2)
respuesta = GroundedAnswer(text="...", source_ids=["BAT-01", "BAT-03", "BAT-99"])
print("Citas inventadas:", citas_inventadas(respuesta, evidencia))
confirmar(citas_inventadas(respuesta, evidencia) == {"BAT-99"}, "Solo BAT-99 es inventada")
