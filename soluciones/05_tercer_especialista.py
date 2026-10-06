"""Solución · Clase 5 · Tu turno 3: envolver al agente `cientifico` en una herramienta."""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from henry_agents.agentic import crear_agente
from henry_agents.config import configure
from henry_agents.practica import confirmar

MODE = configure()
cientifico = crear_agente(
    MODE,
    system_prompt="Eres el especialista en ciencia. Busca fichas de los Cuatro Fantásticos.",
)


@tool
def consultar_cientifico(pedido: str) -> str:
    """Pide al especialista en ciencia fichas de los Cuatro Fantásticos."""
    # Mismo patrón que consultar_dj: invocar al agente y devolver el texto de su última respuesta.
    return cientifico.invoke({"messages": [HumanMessage(pedido)]})["messages"][-1].text


resultado = consultar_cientifico.invoke({"pedido": "ciencia de los Fantásticos"})
print(resultado)
confirmar("[FAN-" in resultado, "El especialista debía citar fichas de los Fantásticos")
