"""Solución · Clase 4 · Tu turno 3: pedir aprobación humana antes de publicar."""

from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from henry_agents.agentic import (
    HumanInTheLoopMiddleware,
    ModeloReglas,
    crear_agente,
    leer_resenas,
    publicar_anuncio,
    solicitudes_pendientes,
)
from henry_agents.config import configure
from henry_agents.practica import confirmar

# La herramienta que ACTÚA es publicar_anuncio: esa es la que necesita una persona.
proteger = {"publicar_anuncio": True}

protegido = crear_agente(
    configure(),  # Cualquier modo: el modelo ingenuo es simulado también en live.
    model=ModeloReglas(falla="obedece_inyeccion"),
    tools=[leer_resenas, publicar_anuncio],
    checkpointer=InMemorySaver(),
    middleware=[HumanInTheLoopMiddleware(proteger)],
)
estado = protegido.invoke(
    {"messages": [HumanMessage("Resume las reseñas de BAT-01")]},
    {"configurable": {"thread_id": str(uuid4())}},
)
pendientes = solicitudes_pendientes(estado)
for accion in pendientes:
    print("⏸️ Espera aprobación:", accion["name"], accion["args"])
confirmar([a["name"] for a in pendientes] == ["publicar_anuncio"], "El anuncio debía quedar en pausa")
