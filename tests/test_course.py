import json
from importlib.resources import files

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.types import Command

from henry_agents.api import app
from henry_agents.fundamentals import run_agent
from henry_agents.graph import build_graph, route_query
from henry_agents.operations import evaluate, retry_demo, run_observed
from henry_agents.retrieval import HashEmbeddings, answer_question, retrieve


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("COURSE_MODE", "offline")


def test_tool_protocol_and_step_limit():
    result = run_agent("Ventas semanales")
    assert result["events"][0]["result"]["total_usd"] == "455.40"
    assert result["steps"] == 2
    assert run_agent("Ventas", max_steps=1)["status"] == "step_limit"


@pytest.mark.parametrize(
    "name,args", [("shell", {"cmd": "bad"}), ("sales_total", {"week": "invalid"})]
)
def test_invalid_tool_is_returned_as_error(name, args):
    class BadModel:
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "x"}])

    result = run_agent("Ventas", max_steps=1, model=BadModel())
    assert "error" in result["events"][0]["result"]
    assert result["status"] == "step_limit"


@pytest.mark.parametrize("query", ["", " ", "x" * 2001])
def test_agent_rejects_bad_input(query):
    with pytest.raises(ValueError):
        run_agent(query)


def test_retrieval_filters_domain_and_abstains():
    docs = retrieve("VPN y recibo de sueldo", domain="it")
    assert docs
    assert all(d.metadata["domain"] == "it" for d in docs)
    result = answer_question("capital de Marte")
    assert result.abstained and result.sources == []
    assert retrieve("¿?") == []


def test_embedding_deterministic_and_accent_normalized():
    embed = HashEmbeddings()
    assert embed.embed_query("contraseña") == embed.embed_query("contrasena")


CASES = json.loads(files("henry_agents").joinpath("data/golden.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_graph_against_golden(case):
    result = build_graph().invoke({"query": case["query"], "results": []})
    assert set(result["domains"]) == set(case["domains"])
    assert set(case["sources"]) <= set(result["sources"])
    assert result["abstained"] == case["abstain"]
    assert len(result.get("results", [])) == len(case["domains"])


def test_router_uses_whole_words():
    assert route_query("Evitar un delito") == []  # No clasificar por substring "it".


@pytest.mark.parametrize("decision,expected", [("approve", "approved"), ("reject", "rejected")])
def test_interrupt_resume(decision, expected):
    graph = build_graph(approval=True)
    config = {"configurable": {"thread_id": decision}}
    paused = graph.invoke({"query": "Solicito una silla", "results": []}, config)
    assert paused["__interrupt__"]
    assert "approval" not in paused
    assert graph.invoke(Command(resume=decision), config)["approval"] == expected


def test_invalid_approval_is_not_truthy_approval():
    graph = build_graph(approval=True)
    config = {"configurable": {"thread_id": "invalid"}}
    graph.invoke({"query": "Solicito una silla", "results": []}, config)
    with pytest.raises(ValueError):
        graph.invoke(Command(resume="false"), config)


def test_threads_are_isolated():
    graph = build_graph(approval=True)
    first = {"configurable": {"thread_id": "first"}}
    second = {"configurable": {"thread_id": "second"}}
    graph.invoke({"query": "Solicito una silla", "results": []}, first)
    result = graph.invoke({"query": "VPN", "results": []}, second)
    assert result["domains"] == ["it"]
    assert graph.get_state(first).next == ("review",)


def test_bounded_retries_and_fallback():
    assert retry_demo(1) == {"result": "recovered", "attempts": 2}
    result = retry_demo(5)
    assert result["attempts"] == 3 and result["result"].startswith("fallback")


def test_observation_has_root_and_no_fake_cost():
    report = run_observed("La VPN falla")
    assert report["metrics"]["llm_calls"] == 0
    roots = [e for e in report["events"] if e.get("parent_id") == "None"]
    assert len(roots) == 1
    assert report["latency_ms"] >= 0
    assert len(report["events"]) > 4


def test_complete_evaluation():
    report = evaluate()
    assert report["n"] == 8
    assert report["route_ok"] == report["retrieval_ok"] == report["abstention_ok"] == 1


def test_http_contract():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.post("/ask", json={"query": ""}).status_code == 422
    assert client.post("/ask", json={"query": "x" * 2001}).status_code == 422
    response = client.post("/ask", json={"query": "La VPN falla"})
    assert response.status_code == 200
    assert "it-vpn" in response.json()["result"]["sources"]


def test_invalid_source_is_rejected_in_live_pipeline(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    import henry_agents.retrieval as retrieval
    from henry_agents.retrieval import Answer

    class StubModel:
        def with_structured_output(self, schema):
            return RunnableLambda(
                lambda _: Answer(answer="Dato sin respaldo", sources=["invented"], abstained=False)
            )

    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(retrieval, "chat_model", lambda: StubModel())
    result = retrieval.answer_question("La VPN falla", mode="live")
    assert result.abstained and not result.sources


def test_whitespace_http_input():
    assert TestClient(app).post("/ask", json={"query": "   "}).status_code == 422


def test_notebook_sources_match_scripts():
    from pathlib import Path

    import jupytext
    import nbformat

    root = Path(__file__).resolve().parents[1]
    for script in (root / "clases").glob("0*.py"):
        expected = jupytext.read(script)
        actual = nbformat.read(script.with_suffix(".ipynb"), as_version=4)
        assert [(c.cell_type, c.source) for c in actual.cells] == [
            (c.cell_type, c.source) for c in expected.cells
        ]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("La VPN falla", "it"),
        ("Mi recibo de sueldo está mal", "rrhh"),
        ("Capital de Marte", "humano"),
        ("La VPN falla y mi recibo de sueldo está mal", "humano"),
        ("Solicito una silla", "humano"),
    ],
)
def test_beginner_graph_routes_or_escalates(question, expected):
    from henry_agents.teaching import build_beginner_graph

    result = build_beginner_graph().invoke({"question": question})
    assert result["area"] == expected
    assert result["answer"]
    if expected == "humano":
        assert result["sources"] == []
    else:
        assert result["sources"]


def test_beginner_tool_demo_observes_before_answering():
    from henry_agents.teaching import support_demo

    result = support_demo()
    assert result["events"][0]["tool"] == "consultar_guia"
    assert result["answer"] == result["events"][0]["result"]
