"""Agentes modernos: create_agent con límites, Deep Agents, aprobación y configuración GPT-6."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from henry_agents import agentic, config
from henry_agents.agentic import (
    ModeloGuionado,
    build_deep_researcher,
    build_prebuilt_agent,
    ejecutar_con_revision,
    llamar,
    mostrar_grafo,
    solicitudes_pendientes,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("COURSE_MODE", "offline")
    monkeypatch.setenv("HENRY_GRAPH_PNG", "0")


def thread():
    return {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 60}


def test_scripted_model_follows_steps_and_ends_safely():
    modelo = ModeloGuionado(pasos=[AIMessage(content="uno"), lambda m: AIMessage(content=str(len(m)))])
    assert modelo.invoke("hola").text == "uno"
    assert modelo.invoke([HumanMessage("a"), HumanMessage("b")]).text == "2"
    assert "Fin del guion" in modelo.invoke("otra").text
    assert modelo.bind_tools([]) is modelo


def test_prebuilt_agent_uses_tool_then_answers_from_real_observation():
    salida = build_prebuilt_agent().invoke({"messages": [HumanMessage("investigación")]})
    herramientas = [m for m in salida["messages"] if isinstance(m, ToolMessage)]
    assert len(herramientas) == 1
    assert "[BAT-01]" in salida["messages"][-1].text


def test_model_call_limit_stops_a_stuck_model():
    atascado = ModeloGuionado(pasos=[llamar("buscar_archivo", {"query": "equipo"}, f"x{i}") for i in range(20)])
    salida = build_prebuilt_agent(model=atascado, max_model_calls=3, max_tool_calls=10).invoke(
        {"messages": [HumanMessage("equipo")]}
    )
    assert sum(isinstance(m, ToolMessage) for m in salida["messages"]) == 3
    assert "limit" in salida["messages"][-1].text.lower()


@pytest.mark.parametrize("valor", [0, -1, 1.5, True, "3"])
def test_prebuilt_agent_rejects_invalid_limits(valor):
    with pytest.raises(ValueError):
        build_prebuilt_agent(max_model_calls=valor)


def test_deep_agent_plans_delegates_and_waits_for_approval():
    agente, cfg = build_deep_researcher(), thread()
    estado = agente.invoke({"messages": [HumanMessage("Prepará una actividad")]}, cfg)
    assert [a["name"] for a in solicitudes_pendientes(estado)] == ["write_file"]
    assert not estado.get("files")
    assert len(estado["todos"]) == 3
    llamadas_task = [
        c for m in estado["messages"] if isinstance(m, AIMessage) for c in m.tool_calls if c["name"] == "task"
    ]
    assert {c["args"]["subagent_type"] for c in llamadas_task} == {"investigador", "dj"}

    final = agente.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
    contenido = final["files"]["/actividad.md"]["content"]
    for identificador in ("BAT-01", "BAT-03", "MUS-02"):
        assert identificador in contenido
    assert "escribí" in final["messages"][-1].text


def test_deep_agent_rejection_writes_nothing_and_reports_it():
    agente = build_deep_researcher(tema="cooperación", universe="chavo")
    estado, registro = ejecutar_con_revision(
        agente,
        {"messages": [HumanMessage("Prepará una actividad")]},
        thread(),
        decidir=lambda accion: {"type": "reject", "message": "No todavía"},
    )
    assert registro == [("write_file", {"type": "reject", "message": "No todavía"})]
    assert not estado.get("files")
    assert "No escribí" in estado["messages"][-1].text


def test_deep_agent_approval_changes_with_topic():
    estado, _ = ejecutar_con_revision(
        build_deep_researcher(tema="cooperación", universe="chavo"),
        {"messages": [HumanMessage("x")]},
        thread(),
        decidir=lambda accion: {"type": "approve"},
    )
    contenido = estado["files"]["/actividad.md"]["content"]
    assert "CHA-01" in contenido and "MUS-01" in contenido and "BAT-" not in contenido


def test_review_loop_is_bounded():
    class Insistente:
        def __init__(self):
            self.invocaciones = 0

        def invoke(self, entrada, cfg):
            self.invocaciones += 1
            pausa = type("P", (), {"value": {"action_requests": [{"name": "write_file"}]}})()
            return {"__interrupt__": [pausa]}

    agente = Insistente()
    with pytest.raises(RuntimeError):
        ejecutar_con_revision(agente, {}, {}, decidir=lambda a: {"type": "approve"}, max_pausas=2)
    assert agente.invocaciones == 3


def test_live_deep_agent_uses_capable_coordinator_and_cheap_specialists(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    roles = []

    def falso(role="default", **_):
        roles.append(role)
        return ModeloGuionado(pasos=[AIMessage(content="ok")])

    monkeypatch.setattr(agentic, "chat_model", falso)
    build_deep_researcher("live")
    assert sorted(roles) == ["agent", "default"]


def test_graph_drawing_works_without_internet(capsys):
    mostrar_grafo(build_prebuilt_agent())
    assert "model" in capsys.readouterr().out


def test_chat_model_targets_gpt6_with_responses_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_AGENT", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    modelo = config.chat_model()
    assert modelo.model_name == "gpt-6-luna"
    assert modelo.use_responses_api is True
    assert modelo.reasoning_effort == "low"
    assert config.chat_model("agent").model_name == "gpt-6.1-sol"
    payload = modelo._get_request_payload([HumanMessage("hola")])
    assert payload["reasoning"] == {"effort": "low"} and payload["max_output_tokens"] == 8000


def test_invalid_reasoning_effort_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "muchisimo")
    with pytest.raises(ValueError):
        config.chat_model()
