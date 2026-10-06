"""Solución · Clase 0 · Cambia el pedido del agente."""

from langchain_core.messages import HumanMessage

from henry_agents.agentic import crear_agente, linea_de_tiempo
from henry_agents.config import configure
from henry_agents.practica import confirmar

MODE = configure()
agente = crear_agente(MODE)

mi_pedido = "Busca fichas de cooperación de El Chavo"  # tema + colección
mi_resultado = agente.invoke({"messages": [HumanMessage(mi_pedido)]})
linea_de_tiempo(mi_resultado["messages"])

confirmar("CHA-" in mi_resultado["messages"][-1].text, "La respuesta debía citar una ficha CHA-")
