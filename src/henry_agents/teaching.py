"""Puentes pequeños para el recorrido inicial; la infraestructura se estudia después."""

from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from henry_agents.config import chat_model, configure
from henry_agents.graph import route_query
from henry_agents.retrieval import answer_question


@tool
def consultar_guia(tema: Literal["vpn"]) -> str:
    """Consulta la guía ficticia de soporte para un problema de VPN."""
    return (
        "Reinicia el cliente VPN y verifica tu conexión a Internet. Nunca compartas tu contraseña."
    )


def support_demo(mode=None):
    """Demo breve de propuesta → herramienta → respuesta; la app ejecuta, no el modelo."""
    mode = configure(mode)
    messages = [
        SystemMessage(
            content="Consulta consultar_guia para VPN y responde en español con esa guía."
        ),
        HumanMessage(content="Mi VPN se desconecta. ¿Qué puedo hacer?"),
    ]
    model = chat_model().bind_tools([consultar_guia]) if mode == "live" else None
    events = []
    for _ in range(3):
        if model:
            response = model.invoke(messages)
        elif isinstance(messages[-1], ToolMessage):
            response = AIMessage(content=messages[-1].content)
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "consultar_guia",
                        "args": {"tema": "vpn"},
                        "id": "guia-1",
                    }
                ],
            )
        messages.append(response)
        if not response.tool_calls:
            return {"answer": response.content, "events": events, "mode": mode}
        for call in response.tool_calls:
            if call["name"] != consultar_guia.name:
                raise ValueError("Herramienta fuera de la lista permitida")
            observation = consultar_guia.invoke(call["args"])
            events.append({"tool": call["name"], "result": observation})
            messages.append(ToolMessage(content=observation, tool_call_id=call["id"]))
    raise RuntimeError("Se alcanzó el límite de tres pasos sin respuesta final")


class SimpleState(TypedDict, total=False):
    question: str
    area: str
    answer: str
    sources: list[str]


def choose_area(question):
    """Reglas visibles para aprender routing; casos mixtos se derivan a una persona."""
    domains = route_query(question, mode="offline")
    return domains[0] if len(domains) == 1 and domains[0] in ("it", "rrhh") else "humano"


def build_beginner_graph(mode=None):
    """Dos especialistas y una salida humana; sin paralelismo ni memoria."""
    mode = configure(mode)

    def classify(state):
        return {"area": choose_area(state["question"])}

    def it(state):
        result = answer_question(state["question"], mode, domain="it")
        return {"answer": result.answer, "sources": result.sources}

    def hr(state):
        result = answer_question(state["question"], mode, domain="rrhh")
        return {"answer": result.answer, "sources": result.sources}

    def human(state):
        return {"answer": "Necesito ayuda de una persona para esta consulta.", "sources": []}

    builder = StateGraph(SimpleState)
    builder.add_node("clasificar", classify)
    builder.add_node("it", it)
    builder.add_node("rrhh", hr)
    builder.add_node("humano", human)
    builder.add_edge(START, "clasificar")
    builder.add_conditional_edges(
        "clasificar", lambda state: state["area"], ["it", "rrhh", "humano"]
    )
    for destination in ["it", "rrhh", "humano"]:
        builder.add_edge(destination, END)
    return builder.compile()
