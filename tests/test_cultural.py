"""Contratos de herramienta y terminación/estado de las arquitecturas culturales."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import ValidationError

from henry_agents.cultural import (
    SearchArgs,
    build_review_graph,
    build_team,
    build_tool_agent,
    buscar_archivo,
    compose,
    search_catalog,
)


@pytest.mark.parametrize(
    "args",
    [
        {"query": "  "},
        {"query": "x" * 241},
        {"query": "equipo", "universe": "internet"},
        {"query": "equipo", "top_k": 0},
        {"query": "equipo", "top_k": 6},
        {"query": "equipo", "top_k": True},
        {"query": "equipo", "top_k": "2"},
        {"query": "equipo", "path": "/etc/passwd"},
        {"query": "equipo", "kind": "codigo"},
    ],
)
def test_tool_rejects_invalid_contract(args):
    with pytest.raises(ValidationError):
        SearchArgs(**args)


def test_filters_limit_and_stable_accent_normalization():
    first = search_catalog("investigación", universe="canciones", kind="cancion", top_k=1)
    second = search_catalog("investigacion", universe="canciones", kind="cancion", top_k=1)
    assert first.hits == second.hits
    assert [h.id for h in first.hits] == ["MUS-02"]
    assert first.inspected == 3
    assert all(len(h.text) <= 600 for h in first.hits)
    assert search_catalog("investigación", universe="batman", kind="cancion").hits == []


@pytest.mark.parametrize("query", ["Batman vacuna marciana", "canciones receta de sopa"])
def test_collection_name_does_not_count_as_evidence_for_an_unknown_topic(query):
    result = search_catalog(query)
    assert result.status == "no_results"
    assert result.hits == []


def test_collection_queries_browse_or_filter_the_topic():
    assert [h.id for h in search_catalog("Batman").hits] == ["BAT-01", "BAT-02", "BAT-03"]
    assert [h.id for h in search_catalog("Batman herramientas").hits] == ["BAT-02"]
    assert search_catalog("Batman herramientas", universe="chavo").hits == []


@pytest.mark.parametrize("mode", ["offline", "live"])
def test_empty_retrieval_does_not_generate(monkeypatch, mode):
    import henry_agents.cultural as cultural

    monkeypatch.setenv("OPENAI_API_KEY", "fake-for-unit-test")
    monkeypatch.setattr(cultural, "chat_model", lambda: pytest.fail("No debe llamar al LLM"))
    result = search_catalog("vacuna marciana")
    assert result.status == "no_results"
    assert compose(result, mode).source_ids == []


def test_structured_tool_returns_citable_data():
    result = buscar_archivo.invoke({"query": "herramientas", "universe": "batman", "top_k": 1})
    assert [h["id"] for h in result["hits"]] == ["BAT-02"]


def test_fabricated_citation_rejected(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    import henry_agents.cultural as cultural

    class FakeModel:
        def with_structured_output(self, schema):
            return RunnableLambda(
                lambda _: cultural.GroundedAnswer(text="Inventado", source_ids=["X"])
            )

    monkeypatch.setenv("OPENAI_API_KEY", "fake-for-unit-test")
    monkeypatch.setattr(cultural, "chat_model", lambda: FakeModel())
    response = compose(search_catalog("Batman"), "live")
    assert not response.source_ids


@pytest.mark.parametrize("source_id", ["BAT-02", "INVENTADA"])
def test_custom_prompt_reaches_model_and_preserves_citation_validation(monkeypatch, source_id):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    import henry_agents.cultural as cultural

    received = []

    class FakeModel:
        def with_structured_output(self, schema):
            assert schema is cultural.GroundedAnswer

            def answer(messages):
                received.append(messages.to_string())
                return cultural.GroundedAnswer(text="Resumen docente", source_ids=[source_id])

            return RunnableLambda(answer)

    monkeypatch.setenv("OPENAI_API_KEY", "fake-for-unit-test")
    monkeypatch.setattr(cultural, "chat_model", lambda: FakeModel())
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Formato docente personalizado"), ("human", "{pregunta}\n{contexto}")]
    )
    response = compose(search_catalog("Batman herramientas"), "live", prompt=prompt)
    assert "Formato docente personalizado" in received[0]
    assert "Batman herramientas" in received[0] and "BAT-02" in received[0]
    assert response.source_ids == (["BAT-02"] if source_id == "BAT-02" else [])


def test_custom_prompt_is_checked_offline_without_calling_model(monkeypatch):
    from langchain_core.prompts import ChatPromptTemplate

    import henry_agents.cultural as cultural

    monkeypatch.setattr(cultural, "chat_model", lambda: pytest.fail("No debe llamar al LLM"))
    prompt = ChatPromptTemplate.from_messages([("human", "{variable_desconocida}")])
    with pytest.raises(KeyError):
        compose(search_catalog("Batman"), "offline", prompt=prompt)


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "2"])
def test_graph_budgets_require_positive_integers(limit):
    with pytest.raises(ValueError, match="entero positivo"):
        build_tool_agent("offline", max_calls=limit)
    with pytest.raises(ValueError, match="entero positivo"):
        build_review_graph("offline", max_attempts=limit)


def test_agent_executes_tool_then_answers():
    result = build_tool_agent("offline").invoke(
        {
            "messages": [HumanMessage(content="investigación Batman")],
            "calls": 0,
        }
    )
    assert result["calls"] == 2
    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    assert "BAT-01" in result["messages"][-1].content


def test_agent_rejects_a_final_answer_without_observing_a_tool():
    class NoToolModel:
        def invoke(self, messages):
            return AIMessage(content="Una afirmación inventada sin buscar")

    result = build_tool_agent("offline", model=NoToolModel()).invoke(
        {"messages": [HumanMessage(content="investigación Batman")], "calls": 0}
    )
    assert result["messages"][-1].content.startswith("No hay una observación válida")


def test_agent_accepts_a_model_answer_after_a_successful_tool_observation():
    class EvidenceModel:
        def invoke(self, messages):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="[BAT-02] Alfred confirma la selección del inventario.")
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "buscar_archivo", "args": {"query": "Batman herramientas"}, "id": "s1"}
                ],
            )

    result = build_tool_agent("offline", model=EvidenceModel()).invoke(
        {"messages": [HumanMessage(content="Batman herramientas")], "calls": 0}
    )
    assert result["calls"] == 2
    assert result["messages"][-1].content.startswith("[BAT-02]")


@pytest.mark.parametrize("query", ["x", "vacuna marciana"])
def test_agent_does_not_trust_a_final_answer_after_failed_or_empty_search(query):
    class InventingModel:
        def invoke(self, messages):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="Inventado a pesar del resultado de la herramienta")
            return AIMessage(
                content="",
                tool_calls=[{"name": "buscar_archivo", "args": {"query": query}, "id": "s1"}],
            )

    result = build_tool_agent("offline", model=InventingModel()).invoke(
        {"messages": [HumanMessage(content=query)], "calls": 0}
    )
    assert result["messages"][-1].content.startswith("No hay")
    assert "Inventado" not in result["messages"][-1].content


def test_agent_requires_new_evidence_for_a_new_user_turn():
    class NoToolModel:
        def invoke(self, messages):
            return AIMessage(content="Uso la evidencia del turno anterior")

    history = build_tool_agent("offline").invoke(
        {"messages": [HumanMessage(content="herramientas Batman")], "calls": 0}
    )["messages"]
    result = build_tool_agent("offline", model=NoToolModel()).invoke(
        {"messages": [*history, HumanMessage(content="vacuna marciana")], "calls": 0}
    )
    assert result["messages"][-1].content.startswith("No hay una observación válida")


def test_agent_stops_a_model_that_never_finishes():
    class LoopModel:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "buscar_archivo",
                        "args": {"query": "equipo"},
                        "id": str(len(messages)),
                    }
                ],
            )

    result = build_tool_agent("offline", max_calls=2, model=LoopModel()).invoke(
        {
            "messages": [HumanMessage(content="equipo")],
            "calls": 0,
        }
    )
    assert result["calls"] == 2
    assert result["messages"][-1].content.startswith("Límite")
    assert not result["messages"][-1].tool_calls


def test_dynamic_workers_deduplicate_plan_and_preserve_all_results():
    result = build_team().invoke(
        {
            "query": "cooperación",
            "universes": ["fantasticos", "chavo", "canciones", "chavo"],
            "parts": [],
        }
    )
    assert len(result["parts"]) == 3
    assert {p["universe"] for p in result["parts"]} == {"fantasticos", "chavo", "canciones"}
    assert all(p["hits"] for p in result["parts"])


@pytest.mark.parametrize("universes", [[], ["internet"]])
def test_invalid_worker_plan(universes):
    with pytest.raises(ValueError):
        build_team().invoke({"query": "equipo", "universes": universes, "parts": []})


def test_revision_corrects_injected_error_within_bound():
    result = build_review_graph("offline", require_approval=False).invoke(
        {
            "query": "investigación Batman",
            "fault": True,
            "attempts": 0,
        }
    )
    assert result["attempts"] == 2
    assert result["valid"]
    assert result["decision"] == "ready"
    assert "INVENTADA" not in result["draft"]["source_ids"]


def test_exhausted_revision_never_returns_invalid_draft():
    result = build_review_graph("offline", require_approval=False, max_attempts=1).invoke(
        {
            "query": "investigación Batman",
            "fault": True,
            "attempts": 0,
        }
    )
    assert result["decision"] == "escalate"
    assert result["attempts"] == 1
    assert result["draft"]["source_ids"] == []


def test_missing_evidence_escalates_without_retry_loop():
    result = build_review_graph("offline", require_approval=False).invoke(
        {
            "query": "vacuna marciana",
            "attempts": 0,
        }
    )
    assert result["decision"] == "escalate"
    assert result["attempts"] == 1


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_human_decision_is_checkpointed(decision):
    graph = build_review_graph("offline")
    config = {"configurable": {"thread_id": decision}}
    paused = graph.invoke({"query": "herramientas Batman", "attempts": 0}, config)
    assert paused["__interrupt__"]
    assert paused["decision"] == "pending"
    result = graph.invoke(Command(resume=decision), config)
    assert result["decision"] == decision
    assert graph.get_state(config).next == ()


def test_human_threads_isolated_and_invalid_decision_can_be_corrected():
    graph = build_review_graph("offline")
    first = {"configurable": {"thread_id": "first"}}
    second = {"configurable": {"thread_id": "second"}}
    for config in [first, second]:
        graph.invoke({"query": "equipo", "attempts": 0}, config)
    graph.invoke(Command(resume="reject"), first)
    assert graph.get_state(second).next == ("review",)
    retry = graph.invoke(Command(resume="yes"), second)
    assert retry["decision"] == "pending"
    assert retry["__interrupt__"][0].value["error"].startswith("Decisión inválida")
    result = graph.invoke(Command(resume="approve"), second)
    assert result["decision"] == "approve"
    assert graph.get_state(second).next == ()
    assert graph.get_state(first).values["decision"] == "reject"


def test_new_request_on_same_thread_does_not_inherit_approval_or_attempts():
    graph = build_review_graph("offline")
    config = {"configurable": {"thread_id": "reused-thread"}}
    first = graph.invoke({"query": "investigación Batman", "fault": True}, config)
    assert first["attempts"] == 2
    graph.invoke(Command(resume="approve"), config)
    second = graph.invoke({"query": "herramientas Batman"}, config)
    assert second["__interrupt__"]
    assert second["decision"] == "pending"
    assert second["attempts"] == 1
    assert second["draft"]["source_ids"] == ["BAT-02"]


def test_agent_reports_validation_error_without_inventing_evidence():
    result = build_tool_agent("offline").invoke(
        {"messages": [HumanMessage(content="x")], "calls": 0}
    )
    assert result["calls"] == 2
    assert result["messages"][-1].content.startswith("La herramienta rechazó")


def test_cultural_golden_dataset():
    from henry_agents.cultural import evaluate_catalog

    report = evaluate_catalog("offline")
    assert report["n"] == 10
    assert report["retrieval_exact"] == report["citations_valid"] == report["abstention_ok"] == 1
