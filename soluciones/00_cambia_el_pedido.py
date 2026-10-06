"""Solución · Clase 0 · Cambia el pedido del agente."""

import re

from langchain_core.messages import HumanMessage

from henry_agents.agentic import crear_agente, linea_de_tiempo
from henry_agents.config import configure
from henry_agents.cultural import load_catalog
from henry_agents.practica import confirmar

MODE = configure()
agente = crear_agente(MODE)

mi_pedido = "Busca fichas de cooperación de El Chavo"  # tema + colección
mi_resultado = agente.invoke({"messages": [HumanMessage(mi_pedido)]})
linea_de_tiempo(mi_resultado["messages"])

observado = " ".join(m.text for m in mi_resultado["messages"] if m.type == "tool")
respuesta = mi_resultado["messages"][-1].text
if MODE == "offline":
    confirmar("CHA-" in observado and "CHA-01" in respuesta, "Debía encontrar y citar CHA-01")
else:
    # En live el texto varía: comprobamos que toda cita exista en el catálogo.
    ids_catalogo = {f["id"] for f in load_catalog()}
    citados = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", respuesta))
    confirmar(citados <= ids_catalogo, "Toda cita debe existir en el catálogo")
