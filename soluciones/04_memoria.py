"""Solución · Clase 4 · Tu turno 1: elegir el hilo correcto para que el agente recuerde."""

from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from henry_agents.agentic import crear_agente
from henry_agents.config import configure
from henry_agents.practica import confirmar

MODE = configure()
conversador = crear_agente(
    MODE,
    tools=[],
    checkpointer=InMemorySaver(),
    system_prompt="Eres un asistente amable. Recuerda lo que la persona te cuenta.",
)
config_beto = {"configurable": {"thread_id": str(uuid4())}}
conversador.invoke({"messages": [HumanMessage("Hola, me llamo Beto")]}, config_beto)

# La memoria es por thread_id: preguntamos en el MISMO hilo donde Beto se presentó.
config_pregunta = config_beto
salida = conversador.invoke({"messages": [HumanMessage("¿Cómo me llamo?")]}, config_pregunta)
print(salida["messages"][-1].text)
# En cualquier modo: el hilo guarda los dos turnos (4 mensajes), o sea, la memoria existe.
confirmar(len(salida["messages"]) == 4, "El hilo de Beto debía guardar sus dos turnos")
if MODE == "offline":
    confirmar("Beto" in salida["messages"][-1].text, "El asistente debía recordar a Beto")
