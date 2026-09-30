"""Clase 1: decidir → ejecutar una herramienta → observar → responder."""

import json
from decimal import Decimal
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from henry_agents.config import chat_model, configure


@tool
def sales_total(week: Literal["2026-W39"]) -> dict:
    """Consulta el total USD y cantidad de ventas de la semana 2026-W39 del dataset ficticio."""
    amounts = [Decimal("125.50"), Decimal("80.00"), Decimal("249.90")]
    return {"week": week, "count": len(amounts), "total_usd": str(sum(amounts))}


class DemoToolModel:
    """Guion determinista para inspeccionar el protocolo; NO es un LLM."""

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            result = json.loads(messages[-1].content)
            return AIMessage(content=f"Reporte de ventas: {result}")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sales_total",
                    "args": {"week": "2026-W39"},
                    "id": "sales-1",
                    "type": "tool_call",
                }
            ],
        )


def run_agent(query, mode=None, max_steps=4, model=None):
    """El modelo propone; Python valida y ejecuta una allowlist de herramientas."""
    mode = configure(mode)
    if not query.strip() or len(query) > 2000:
        raise ValueError("La consulta debe tener entre 1 y 2000 caracteres")
    if max_steps < 1:
        raise ValueError("max_steps debe ser positivo")
    tools = {"sales_total": sales_total}
    model = model or (
        chat_model().bind_tools(list(tools.values())) if mode == "live" else DemoToolModel()
    )
    messages = [
        SystemMessage(
            content=(
                "Eres analista de ventas. Consulta sales_total antes de reportar cifras. "
                "Responde en español usando solo los resultados observados. No inventes tendencias: "
                "solo hay una semana. No puedes enviar correos."
            )
        ),
        HumanMessage(content=query),
    ]
    events = []
    tokens = 0
    for step in range(max_steps):
        reply = model.invoke(messages)
        tokens += (reply.usage_metadata or {}).get("total_tokens", 0)
        messages.append(reply)
        if not reply.tool_calls:
            return {
                "answer": reply.content,
                "status": "completed",
                "steps": step + 1,
                "events": events,
                "tokens": tokens,
                "mode": mode,
            }
        for call in reply.tool_calls:
            try:
                if call["name"] not in tools:
                    raise ValueError("Herramienta no permitida")
                result = tools[call["name"]].invoke(call["args"])
            except (ValueError, TypeError) as exc:
                result = {"error": type(exc).__name__}
            events.append({"tool": call["name"], "args": call["args"], "result": result})
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))
    return {
        "answer": "Límite de pasos alcanzado; no hay respuesta final verificada.",
        "status": "step_limit",
        "steps": max_steps,
        "events": events,
        "tokens": tokens,
        "mode": mode,
    }
