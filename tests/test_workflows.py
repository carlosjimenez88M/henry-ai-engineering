"""Contratos y fallas de la ruta inicial, sin API ni servicios externos."""

import importlib.util
import json
from importlib.resources import files
from pathlib import Path
from threading import Barrier

import pytest

from henry_agents import workflows as wf

ROOT = Path(__file__).resolve().parents[1]
PEDIDO = {"sku": "arroz", "cantidad": 2, "barrio": "Centro"}


@pytest.mark.parametrize("cantidad", [0, -1, 21, True, 1.5, "2", None])
def test_tool_rejects_invalid_quantity(cantidad):
    with pytest.raises(ValueError):
        wf.cotizar({**PEDIDO, "cantidad": cantidad})


@pytest.mark.parametrize("pedido", [None, {}, {**PEDIDO, "extra": 1}, {**PEDIDO, "barrio": " "}])
def test_tool_rejects_malformed_contract(pedido):
    with pytest.raises(ValueError):
        wf.validar_pedido(pedido)


def test_money_and_missing_evidence_are_distinct():
    assert wf.cotizar(PEDIDO)["total_usd"] == "5.60"
    assert wf.cotizar({**PEDIDO, "barrio": "Retiro"})["total_usd"] == "3.60"
    assert wf.cotizar({**PEDIDO, "sku": "leche"}) == {"estado": "sin_producto", "fuentes": []}
    assert wf.cotizar({**PEDIDO, "sku": "cuaderno"})["estado"] == "sin_stock"
    assert wf.cotizar({**PEDIDO, "barrio": "Sur"})["estado"] == "fuera_de_cobertura"


def test_fresh_catalog_and_read_only_quote():
    tienda = wf.cargar_tienda()
    original = json.loads(json.dumps(tienda))
    wf.preparar_pedido(PEDIDO, "aprobar", tienda)
    assert tienda == original
    tienda["productos"][0]["stock"] = 0
    assert wf.cargar_tienda()["productos"][0]["stock"] == 8


@pytest.mark.parametrize("mensaje,area", [
    ("¿CAFÉ?", "productos"), ("¿ENVÍOS?", "entregas"),
    ("Cafetería", "humano"), ("Precio y entrega", "humano"),
    ("Reclamo", "humano"), ("No quiero café", "productos"),
])
def test_router_normalization_ambiguity_and_known_limit(mensaje, area):
    assert wf.clasificar_por_reglas(mensaje) == area


def test_model_route_schema_and_no_silent_fallback():
    class Modelo:
        def with_structured_output(self, schema):
            assert schema is wf.RutaModelo
            return self

        def invoke(self, messages):
            assert messages[-1] == ("human", "¿Tienen arroz?")
            return {"area": "productos", "motivo": "Consulta de disponibilidad"}

    assert wf.clasificar_con_modelo("¿Tienen arroz?", Modelo())["area"] == "productos"

    class ModeloRoto(Modelo):
        def invoke(self, messages):
            raise ConnectionError("Falla simulada")

    with pytest.raises(ConnectionError):
        wf.clasificar_con_modelo("¿Tienen arroz?", ModeloRoto())
    with pytest.raises(ValueError):
        wf.RutaModelo(area="ejecutar_codigo", motivo="Fuera del contrato")


def test_workers_overlap_without_timing_assumptions():
    barrera = Barrier(2)

    def worker(valor):
        barrera.wait(timeout=5)
        return valor

    result = wf.ejecutar_paralelo({"a": lambda: worker(1), "b": lambda: worker(2)})
    assert result == {"a": 1, "b": 2}


def test_worker_failure_is_observable():
    def roto():
        raise RuntimeError("Falla de worker")

    with pytest.raises(RuntimeError, match="Falla de worker"):
        wf.ejecutar_paralelo({"bien": lambda: 1, "roto": roto})


def test_revision_uses_feedback_and_does_not_publish_on_exhaustion():
    cotizacion = wf.cotizar(PEDIDO)
    feedback_recibido = []

    def generar(feedback):
        feedback_recibido.append(list(feedback))
        return {
            "total_usd": cotizacion["total_usd"] if feedback else "0.00",
            "fuentes": cotizacion["fuentes"], "requiere_aprobacion": True,
        }

    result = wf.revisar_hasta_limite(generar, cotizacion)
    assert result["estado"] == "validado" and len(result["historial"]) == 2
    assert feedback_recibido[0] == [] and feedback_recibido[1]
    result = wf.revisar_hasta_limite(lambda f: {}, cotizacion, 2)
    assert result["estado"] == "limite_alcanzado" and result["borrador"] is None
    assert len(result["historial"]) == 2
    assert wf.evaluar_borrador({"fuentes": [{}]}, cotizacion)


@pytest.mark.parametrize("limite", [0, -1, True, 1.5, 6])
def test_revision_limit_is_validated(limite):
    with pytest.raises(ValueError):
        wf.revisar_hasta_limite(lambda f: {}, wf.cotizar(PEDIDO), limite)


@pytest.mark.parametrize("plan", [[], ["producto"], ["producto", "cobrar"],
                                  ["producto", "producto"], ["entrega"], [None]])
def test_incomplete_or_unknown_plan_rejected_before_workers(plan, monkeypatch):
    llamados = []
    monkeypatch.setattr(wf, "consultar_producto", lambda *args: llamados.append(True))
    with pytest.raises(ValueError):
        wf.ejecutar_plan(PEDIDO, plan)
    assert llamados == []


def test_plan_varies_by_delivery_and_reject_does_not_release_message():
    assert wf.planificar(PEDIDO) == ["producto", "entrega"]
    assert wf.planificar({**PEDIDO, "barrio": "Retiro"}) == ["producto"]
    assert wf.preparar_pedido(PEDIDO)["mensaje"] is None
    assert wf.preparar_pedido(PEDIDO, "rechazar")["mensaje"] is None
    assert wf.preparar_pedido(PEDIDO, "aprobar")["accion_externa"] is False
    with pytest.raises(ValueError):
        wf.preparar_pedido(PEDIDO, "false")


CASOS = json.loads(files("henry_agents").joinpath("data/workflows_casos.json").read_text("utf-8"))


@pytest.mark.parametrize("caso", CASOS, ids=lambda c: c["id"])
def test_golden_project_cases(caso):
    if caso["estado"] == "entrada_invalida":
        with pytest.raises(ValueError):
            wf.preparar_pedido(caso["pedido"], caso["decision"])
        return
    salida = wf.preparar_pedido(caso["pedido"], caso["decision"])
    assert salida["estado"] == caso["estado"]
    assert salida["accion_externa"] is False
    if "total_usd" in caso:
        assert salida["cotizacion"]["total_usd"] == caso["total_usd"]
    if caso["estado"] != "aprobado":
        assert salida["mensaje"] is None


def cargar_modulo(ruta):
    spec = importlib.util.spec_from_file_location(ruta.stem, ROOT / ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_project_solution_passes_and_unfinished_starter_fails():
    evaluador = cargar_modulo(Path("scripts/evaluate_workflows.py"))
    solucion = cargar_modulo(Path("proyectos/tienda_workflows/solution.py"))
    starter = cargar_modulo(Path("proyectos/tienda_workflows/starter.py"))
    assert evaluador.evaluar(solucion.resolver)["correctos"] == 10
    assert evaluador.evaluar(starter.resolver)["correctos"] == 0


def test_grader_cannot_pass_unexpected_exceptions():
    evaluador = cargar_modulo(Path("scripts/evaluate_workflows.py"))

    def roto(pedido, decision):
        raise RuntimeError("Falla simulada")

    reporte = evaluador.evaluar(roto)
    assert reporte["correctos"] == 0
    assert all(r["errores"] for r in reporte["resultados"])
