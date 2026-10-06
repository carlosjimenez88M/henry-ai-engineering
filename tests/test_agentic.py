"""Ruta avanzada: cerebros offline, create_agent, Deep Agents, semántica y configuración GPT-6."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from henry_agents import agentic, config
from henry_agents.agentic import (
    HumanInTheLoopMiddleware,
    ModeloCoordinador,
    ModeloGuionado,
    ModeloReglas,
    crear_agente,
    crear_equipo_profundo,
    ejecutar_con_revision,
    interpretar_pedido,
    leer_resenas,
    llamar,
    mostrar_grafo,
    publicar_anuncio,
    solicitudes_pendientes,
    ver_en_vivo,
)
from henry_agents.practica import ErrorDelCurso, comprobar, confirmar
from henry_agents.semantica import buscar_por_significado, similitud, vector_de


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("COURSE_MODE", "offline")
    monkeypatch.setenv("HENRY_GRAPH_PNG", "0")


def hilo():
    return {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 80}


def herramientas_usadas(mensajes):
    return [m for m in mensajes if isinstance(m, ToolMessage)]


@pytest.mark.parametrize(
    "pedido,esperado",
    [
        ("Busca fichas de cooperación de El Chavo", ("cooperación", "chavo")),
        ("investigación de Batman", ("investigación", "batman")),
        ("Elige una canción para una actividad de equipo", ("equipo", "canciones")),
        ("hola", (None, "todos")),
    ],
)
def test_interpreting_requests(pedido, esperado):
    argumentos = interpretar_pedido(pedido)
    assert (argumentos["query"], argumentos["universe"]) == esperado


def test_rules_brain_follows_the_request_and_is_rerunnable():
    agente = crear_agente()
    for _ in range(2):  # Reejecutar no rompe nada: el cerebro no guarda estado.
        salida = agente.invoke({"messages": [HumanMessage("Busca cooperación de El Chavo")]})
        assert "[CHA-01]" in salida["messages"][-1].text
        assert len(herramientas_usadas(salida["messages"])) == 1


def test_rules_brain_asks_for_a_topic_instead_of_searching_nothing():
    salida = crear_agente().invoke({"messages": [HumanMessage("hola")]})
    assert not herramientas_usadas(salida["messages"])
    assert "Dime un tema" in salida["messages"][-1].text


def test_failure_no_tool_invents_a_source():
    salida = crear_agente(model=ModeloReglas(falla="no_usa_herramienta")).invoke(
        {"messages": [HumanMessage("investigación de Batman")]}
    )
    assert not herramientas_usadas(salida["messages"])
    assert "[BAT-07]" in salida["messages"][-1].text


def test_failure_invalid_args_is_corrected_after_tool_error():
    salida = crear_agente(model=ModeloReglas(falla="argumentos_invalidos")).invoke(
        {"messages": [HumanMessage("investigación de Batman")]}
    )
    resultados = herramientas_usadas(salida["messages"])
    assert [r.status for r in resultados] == ["error", "success"]
    assert "[BAT-01]" in salida["messages"][-1].text


def test_failure_loop_is_cut_by_middleware_in_spanish():
    salida = crear_agente(model=ModeloReglas(falla="bucle"), max_llamadas_modelo=3).invoke(
        {"messages": [HumanMessage("investigación de Batman")]}
    )
    assert len(herramientas_usadas(salida["messages"])) == 3
    assert salida["messages"][-1].text.startswith("Límite de llamadas al modelo")


def test_failure_invented_id_is_detectable():
    salida = crear_agente(model=ModeloReglas(falla="inventa_id")).invoke(
        {"messages": [HumanMessage("investigación de Batman")]}
    )
    assert "[BAT-99]" in salida["messages"][-1].text


def test_memory_depends_on_thread():
    agente = crear_agente(tools=[], checkpointer=InMemorySaver())
    uno = hilo()
    agente.invoke({"messages": [HumanMessage("Hola, me llamo Ana")]}, uno)
    assert "Ana" in agente.invoke({"messages": [HumanMessage("¿Cómo me llamo?")]}, uno)["messages"][-1].text
    otro = agente.invoke({"messages": [HumanMessage("¿Cómo me llamo?")]}, hilo())
    assert "No lo sé" in otro["messages"][-1].text


class RespuestaConFuentes(BaseModel):
    respuesta: str
    fuentes: list[str]


def test_structured_output_from_agent_and_from_model():
    salida = crear_agente(response_format=RespuestaConFuentes).invoke(
        {"messages": [HumanMessage("investigación de Batman")]}
    )
    assert salida["structured_response"].fuentes == ["BAT-01", "BAT-03"]
    directa = ModeloReglas().with_structured_output(RespuestaConFuentes).invoke(
        [("human", 'Evidencia: {"hits": [{"id": "CHA-02"}]}')]
    )
    assert directa.fuentes == ["CHA-02"]


def test_prompt_injection_is_ignored_by_default_and_stopped_by_human_review():
    seguro = crear_agente(tools=[leer_resenas, publicar_anuncio]).invoke(
        {"messages": [HumanMessage("Resume las reseñas de BAT-01")]}
    )
    assert "no las seguí" in seguro["messages"][-1].text
    ingenuo = crear_agente(
        model=ModeloReglas(falla="obedece_inyeccion"),
        tools=[leer_resenas, publicar_anuncio],
        checkpointer=InMemorySaver(),
        middleware=[HumanInTheLoopMiddleware({"publicar_anuncio": True})],
    )
    cfg = hilo()
    pausado = ingenuo.invoke({"messages": [HumanMessage("Resume las reseñas de BAT-01")]}, cfg)
    accion = solicitudes_pendientes(pausado)[0]
    assert accion["args"] == {"texto": "Todas las actividades de hoy están canceladas"}
    antes = len(agentic.ANUNCIOS_PUBLICADOS)
    ingenuo.invoke(Command(resume={"decisions": [{"type": "reject", "message": "Es una inyección"}]}), cfg)
    assert len(agentic.ANUNCIOS_PUBLICADOS) == antes


def test_scripted_model_restarts_each_turn():
    modelo = ModeloGuionado(pasos=[AIMessage(content="uno"), AIMessage(content="dos")])
    assert modelo.invoke([HumanMessage("a")]).text == "uno"
    assert modelo.invoke([HumanMessage("a"), AIMessage(content="uno")]).text == "dos"
    assert modelo.invoke([HumanMessage("b")]).text == "uno"


def test_deep_team_plans_delegates_in_parallel_and_waits():
    equipo, cfg = crear_equipo_profundo(), hilo()
    estado = equipo.invoke({"messages": [HumanMessage("Actividad de cooperación con El Chavo")]}, cfg)
    assert [a["name"] for a in solicitudes_pendientes(estado)] == ["write_file"]
    assert not estado.get("files")
    tareas = [c for m in estado["messages"] if isinstance(m, AIMessage) for c in m.tool_calls if c["name"] == "task"]
    assert {c["args"]["subagent_type"] for c in tareas} == {"investigador", "dj"}
    final = equipo.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
    contenido = final["files"]["/actividad.md"]["content"]
    assert "[CHA-01]" in contenido and "[MUS-01]" in contenido


def test_deep_team_rejection_and_edit_also_need_approval():
    equipo = crear_equipo_profundo()
    estado, registro = ejecutar_con_revision(
        equipo,
        {"messages": [HumanMessage("Actividad de investigación con Batman")]},
        hilo(),
        decidir=lambda accion: {"type": "reject", "message": "Todavía no"},
    )
    assert registro[0][0] == "write_file" and not estado.get("files")
    assert "No guardé" in estado["messages"][-1].text


def test_editing_a_file_also_requires_approval():
    editor = ModeloGuionado(
        pasos=[llamar("edit_file", {"file_path": "/a.md", "old_string": "x", "new_string": "y"}, "e1")]
    )
    modelos = {"coordinador": editor, "investigador": ModeloReglas(), "dj": ModeloReglas(), "general-purpose": ModeloReglas()}
    estado = crear_equipo_profundo(models=modelos).invoke({"messages": [HumanMessage("edita")]}, hilo())
    assert [a["name"] for a in solicitudes_pendientes(estado)] == ["edit_file"]


def test_coordinator_has_no_search_tool_and_stuck_specialist_is_bounded():
    modelos = {
        "coordinador": ModeloCoordinador(),
        "investigador": ModeloReglas(falla="bucle"),
        "dj": ModeloReglas(),
        "general-purpose": ModeloReglas(),
    }
    equipo = crear_equipo_profundo(models=modelos, max_llamadas_especialista=3)
    estado, _ = ejecutar_con_revision(
        equipo, {"messages": [HumanMessage("investigación de Batman")]}, hilo(), lambda a: {"type": "approve"}
    )
    respuestas = [m.text for m in estado["messages"] if isinstance(m, ToolMessage) and m.name == "task"]
    assert any(r.startswith("Límite de llamadas al modelo") for r in respuestas)


def test_streaming_shows_subagents_and_returns_pending_actions(capsys):
    equipo, cfg = crear_equipo_profundo(), hilo()
    estado = ver_en_vivo(equipo, {"messages": [HumanMessage("Actividad de equipo con Fantásticos")]}, cfg)
    salida = capsys.readouterr().out
    assert "↳" in salida and "⏸️" in salida
    assert [a["name"] for a in solicitudes_pendientes(estado)] == ["write_file"]


def test_review_loop_is_bounded():
    class Insistente:
        invocaciones = 0

        def invoke(self, entrada, cfg):
            Insistente.invocaciones += 1
            pausa = type("P", (), {"value": {"action_requests": [{"name": "write_file"}]}})()
            return {"__interrupt__": [pausa]}

    with pytest.raises(RuntimeError):
        ejecutar_con_revision(Insistente(), {}, {}, decidir=lambda a: {"type": "approve"}, max_pausas=2)
    assert Insistente.invocaciones == 3


@pytest.mark.parametrize("valor", [0, -1, 1.5, True, "3"])
def test_invalid_limits_are_rejected(valor):
    with pytest.raises(ValueError):
        crear_agente(max_llamadas_modelo=valor)
    with pytest.raises(ValueError):
        crear_equipo_profundo(max_llamadas_especialista=valor)


def test_live_team_uses_capable_coordinator_and_cheap_specialists(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    roles = []

    def falso(role="default", **_):
        roles.append(role)
        return ModeloReglas()

    monkeypatch.setattr(agentic, "chat_model", falso)
    crear_equipo_profundo("live")
    assert sorted(roles) == ["agent", "default"]


def test_semantic_search_finds_meaning_that_lexical_search_misses():
    from henry_agents.cultural import search_catalog

    assert search_catalog("un enigma para resolver").hits == []
    ids = [i for _, i, _ in buscar_por_significado("un enigma para resolver")]
    assert "BAT-01" in ids
    assert similitud(vector_de("enigma"), vector_de("detective")) > similitud(vector_de("enigma"), vector_de("equipo"))
    assert buscar_por_significado("xyz desconocido") == []


def test_practice_helpers(capsys):
    assert comprobar(True, "bien", "pista") is True
    assert comprobar(False, "bien", "revisa esto") is False
    assert "Pista: revisa esto" in capsys.readouterr().out
    with pytest.raises(ErrorDelCurso):
        confirmar(False, "no se cumplió")


def test_graph_drawing_works_without_internet(capsys):
    mostrar_grafo(crear_agente())
    assert "model" in capsys.readouterr().out


def test_chat_model_targets_gpt6_with_responses_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for variable in ("OPENAI_MODEL", "OPENAI_MODEL_AGENT", "OPENAI_REASONING_EFFORT"):
        monkeypatch.delenv(variable, raising=False)
    modelo = config.chat_model()
    assert modelo.model_name == "gpt-6-luna" and modelo.use_responses_api is True
    assert config.chat_model("agent").model_name == "gpt-6.1-sol"
    payload = modelo._get_request_payload([HumanMessage("hola")])
    assert payload["reasoning"] == {"effort": "low"} and payload["max_output_tokens"] == 8000


def test_reasoning_effort_is_validated_per_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "none")
    assert config.chat_model().reasoning_effort == "none"  # Luna acepta none
    with pytest.raises(ValueError):
        config.chat_model("agent")  # Sol no
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "muchisimo")
    with pytest.raises(ValueError):
        config.chat_model()


def test_cost_helpers(capsys):
    assert config.costo_usd("gpt-6-luna", 1_000_000, 0) == pytest.approx(0.10)
    with config.medir_costo() as medicion:
        crear_agente().invoke({"messages": [HumanMessage("investigación de Batman")]})
    assert medicion["usd"] == 0 and "offline" in capsys.readouterr().out


def test_scripted_call_helper():
    mensaje = llamar("buscar_archivo", {"query": "equipo"}, "x")
    assert mensaje.tool_calls[0]["name"] == "buscar_archivo"
