"""Clase 4: callbacks, métricas, fallas controladas y evaluación reproducible."""

import json
import os
import time
from importlib.resources import files
from threading import Lock
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from henry_agents.config import configure
from henry_agents.graph import build_graph


class Metrics(BaseCallbackHandler):
    """Callbacks sin almacenar prompts ni claves. Seguros para workers paralelos."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.errors = 0
        self.events = []
        self.started = {}
        self.lock = Lock()

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        with self.lock:
            self.started[str(run_id)] = time.perf_counter()
            self.events.append(
                {
                    "run_id": str(run_id),
                    "parent_id": str(parent_run_id),
                    "name": kwargs.get("name", "chain"),
                    "event": "start",
                }
            )

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        with self.lock:
            start = self.started.pop(str(run_id), time.perf_counter())
            self.events.append(
                {
                    "run_id": str(run_id),
                    "event": "end",
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            )

    def on_chain_error(self, error, *, run_id, **kwargs):
        with self.lock:
            self.errors += 1
            self.started.pop(str(run_id), None)
            self.events.append(
                {"run_id": str(run_id), "event": "error", "type": type(error).__name__}
            )

    def on_llm_end(self, response, **kwargs):
        with self.lock:
            self.calls += 1
            for generation in response.generations:
                for item in generation:
                    usage = getattr(item.message, "usage_metadata", None) or {}
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)

    def summary(self):
        price_in = os.getenv("INPUT_USD_PER_MILLION")
        price_out = os.getenv("OUTPUT_USD_PER_MILLION")
        estimated = None
        if price_in and price_out:
            estimated = (
                self.input_tokens * float(price_in) + self.output_tokens * float(price_out)
            ) / 1_000_000
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls": self.calls,
            "chain_errors": self.errors,
            "estimated_cost_usd": estimated,
        }


def run_observed(query, mode=None):
    """Registra el recorrido localmente; no envía trazas a servicios externos."""
    mode = configure(mode)
    metrics = Metrics()
    request_id = str(uuid4())
    start = time.perf_counter()
    result = build_graph(mode).invoke(
        {"query": query, "results": []},
        config={
            "callbacks": [metrics],
            "run_name": "support-desk",
            "recursion_limit": 12,
            "metadata": {"request_id": request_id, "course": "henry-m3", "mode": mode},
        },
    )
    return {
        "request_id": request_id,
        "mode": mode,
        "result": result,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "metrics": metrics.summary(),
        "events": metrics.events,
    }


def evaluate(mode=None, limit=None):
    """Mide contratos de routing/retrieval; no sustituye revisión semántica humana."""
    mode = configure(mode)
    cases = json.loads(files("henry_agents").joinpath("data/golden.json").read_text())
    if limit is not None:
        if limit < 1:
            raise ValueError("limit debe ser positivo")
        cases = cases[:limit]
    rows = []
    for case in cases:
        report = run_observed(case["query"], mode)
        result = report["result"]
        rows.append(
            {
                "id": case["id"],
                "route_ok": set(result["domains"]) == set(case["domains"]),
                "retrieval_ok": set(case["sources"]) <= set(result["sources"]),
                "abstention_ok": result["abstained"] == case["abstain"],
                "latency_ms": report["latency_ms"],
                "metrics": report["metrics"],
                "request_id": report["request_id"],
            }
        )
    return {
        "mode": mode,
        "n": len(rows),
        "cases": rows,
        **{
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in ("route_ok", "retrieval_ok", "abstention_ok")
        },
    }


class RetryState(TypedDict):
    result: str


def retry_demo(failures=1, max_attempts=3):
    """Inyecta TimeoutError sin red: retries solo para errores transitorios."""
    attempts = []

    def flaky(state):
        attempts.append(len(attempts) + 1)
        if len(attempts) <= failures:
            raise TimeoutError("Fallo transitorio simulado")
        return {"result": "recovered"}

    builder = StateGraph(RetryState)
    builder.add_node(
        "service",
        flaky,
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            initial_interval=0.01,
            jitter=False,
            retry_on=TimeoutError,
        ),
    )
    builder.add_edge(START, "service")
    builder.add_edge("service", END)
    try:
        result = builder.compile().invoke({})
    except TimeoutError:
        result = {"result": "fallback: escalar a una persona"}
    return {**result, "attempts": len(attempts)}
