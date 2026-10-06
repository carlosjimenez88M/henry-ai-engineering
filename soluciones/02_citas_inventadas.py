"""Clase 2 · Tu turno 1: encontrar las citas que no están en la evidencia."""

from langchain_core.messages import HumanMessage

from henry_agents.agentic import ModeloReglas, crear_agente
from henry_agents.config import configure
from henry_agents.cultural import GroundedAnswer, SearchResult
from henry_agents.practica import confirmar


def citas_inventadas(respuesta, evidencia):
    disponibles = {ficha.id for ficha in evidencia.hits}
    citadas = set(respuesta.source_ids)
    return citadas - disponibles  # resta de conjuntos: lo citado que NO recuperamos


# El mismo agente con la falla inventa_id que usa la clase (también en live).
agente = crear_agente(configure(), model=ModeloReglas(falla="inventa_id"), response_format=GroundedAnswer)
salida = agente.invoke({"messages": [HumanMessage("investigación de Batman")]})
respuesta = salida["structured_response"]
busqueda = next(m for m in salida["messages"] if m.type == "tool" and m.name == "buscar_archivo")
observacion = SearchResult.model_validate_json(busqueda.text)

print("Citas inventadas:", citas_inventadas(respuesta, observacion))
confirmar(citas_inventadas(respuesta, observacion) == {"BAT-99"}, "Solo BAT-99 es inventada")
