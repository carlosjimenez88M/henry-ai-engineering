"""Recuperación, fidelidad y control del Agentic RAG, incluidas fallas de modelos."""

import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import pytest

from henry_agents import config, rag
from henry_agents.retrieval import HashEmbeddings

ROOT = Path(__file__).resolve().parents[1]
PREGUNTA = "¿Cuánto cuesta el envío a Centro?"


@pytest.mark.parametrize("tamano,overlap", [(0, 0), (4, 0), (201, 0), (10, 10),
                                          (10, -1), (True, 0), (10, 1.5)])
def test_invalid_chunk_sizes_do_not_loop(tamano, overlap):
    with pytest.raises(ValueError):
        rag.fragmentar(palabras=tamano, solapamiento=overlap)


def test_chunk_metadata_stable_overlap_and_no_duplicate_ids():
    docs = [{"id": "D", "titulo": "T", "categoria": "retiro", "vigente": True,
             "texto": "uno dos tres cuatro cinco seis siete ocho nueve diez once doce"}]
    fragments = rag.fragmentar(docs, palabras=5, solapamiento=2)
    assert fragments[0]["texto"].split()[-2:] == fragments[1]["texto"].split()[:2]
    assert all(f["doc_id"] == "D" for f in fragments)
    assert len({f["id"] for f in fragments}) == len(fragments)
    assert fragments == rag.fragmentar(docs, palabras=5, solapamiento=2)
    with pytest.raises(ValueError, match="duplicados"):
        rag.fragmentar(docs + docs)


def test_archived_policy_excluded_even_when_query_matches():
    indice = rag.IndiceLexico()
    hits = indice.buscar("envío gratis gratuito Centro", categoria="entregas", k=5)
    assert hits and {f["doc_id"] for f in hits} == {"ENVIO-2026"}
    assert "USD 2.00" in hits[0]["texto"]
    assert all(f["vigente"] for f in indice.fragments)


@pytest.mark.parametrize("k", [0, 6, True, 1.5])
def test_retrieval_limit_validated(k):
    with pytest.raises(ValueError):
        rag.IndiceLexico().buscar(PREGUNTA, k=k)


def test_bm25_determinism_filters_and_empty_index():
    indice = rag.IndiceLexico()
    assert indice.buscar("ENVÍO CENTRO") == indice.buscar("envio centro")
    assert indice.buscar("unicornios intergalácticos") == []
    assert rag.IndiceLexico([]).buscar(PREGUNTA) == []
    assert all(f["categoria"] == "retiro" for f in indice.buscar("retiro", categoria="retiro"))
    with pytest.raises(ValueError):
        indice.buscar(PREGUNTA, categoria="ejecutar_codigo")


def test_vector_store_integration_uses_same_active_chunks_with_test_embeddings():
    # HashEmbeddings es un doble lexical, NO embeddings semánticos reales.
    indice = rag.IndiceVectorial(embeddings=HashEmbeddings())
    hits = indice.buscar("envío Centro", categoria="entregas", k=1)
    assert {h["doc_id"] for h in hits} == {"ENVIO-2026"}
    assert hits[0]["vigente"] is True


def test_empty_vector_index_does_not_embed_documents_or_query():
    class NoEmbeddings(HashEmbeddings):
        def embed_documents(self, texts):
            raise AssertionError("Un índice vacío no debe enviar documentos")

        def embed_query(self, text):
            raise AssertionError("Un índice vacío no debe consultar la API")

    assert rag.IndiceVectorial([], embeddings=NoEmbeddings()).buscar(PREGUNTA) == []


def test_retrieval_coverage_is_not_merely_nonempty():
    needs = [rag.Necesidad(consulta="envío Centro", categoria="entregas",
                          terminos_requeridos=["Centro", "USD 2.00"])]
    irrelevant = rag.IndiceLexico().buscar("horario")
    assert irrelevant and not rag.evaluar_evidencia(needs, irrelevant)["suficiente"]
    relevant = rag.IndiceLexico().buscar("envío Centro")
    assert rag.evaluar_evidencia(needs, relevant)["suficiente"]
    assert not rag.evaluar_evidencia(needs, [{**relevant[0], "vigente": False}])["suficiente"]


def test_valid_id_without_actual_quote_or_fidelity_is_insufficient():
    evidence = rag.IndiceLexico().buscar(PREGUNTA, categoria="entregas", k=1)
    needs = [n.model_dump() for n in rag.inferir_necesidades(PREGUNTA)]
    invented = rag.RespuestaRAG(abstencion=False, afirmaciones=[rag.Afirmacion(
        texto="Envío gratuito", fuente=evidence[0]["id"],
        cita_literal="Una cita que no aparece en el manual",
    )])
    assert not rag.validar_respuesta(invented, needs, evidence)
    wrong_claim = rag.RespuestaRAG(abstencion=False, afirmaciones=[rag.Afirmacion(
        texto="El envío a Centro cuesta USD 99.00", fuente=evidence[0]["id"],
        cita_literal=evidence[0]["texto"],
    )])
    assert rag.validar_respuesta(wrong_claim, needs, evidence)
    assert not rag.revisar_fidelidad(wrong_claim, evidence).respaldada


def test_compound_answer_must_cover_every_need():
    pregunta = "¿Cuánto cuesta el envío a Centro y cómo retiro mi pedido?"
    assert rag.rag_clasico(pregunta, k=1)["estado"] == "abstencion"
    result = rag.crear_agente_rag(k=1).invoke(rag.estado_inicial(pregunta))
    assert result["estado"] == "respondido" and result["busquedas"] == 2
    assert set(result["fuentes"]) == {"ENVIO-2026-c01", "RETIRO-2026-c01"}
    assert result["pregunta"] == pregunta


def test_query_rewrite_comes_after_failed_observation():
    result = rag.crear_agente_rag(k=1).invoke(rag.estado_inicial("¿Hacen mandados?"))
    recoveries = [e for e in result["eventos"] if e["nodo"] == "recuperar"]
    assert recoveries[0]["ids"] == [] and recoveries[1]["ids"] == ["ENVIO-2026-c01"]
    assert result["estado"] == "respondido" and result["llamadas_llm"] == 0
    assert result["revision_fidelidad"]["respaldada"]


def test_catalog_tool_for_stock_does_not_invent_inventory_from_docs():
    pregunta = "¿Cuál es el stock de arroz?"
    assert rag.rag_clasico(pregunta)["estado"] == "abstencion"
    result = rag.crear_agente_rag().invoke(rag.estado_inicial(pregunta))
    assert result["fuentes"] == ["CAT-ARROZ"]
    assert "8 unidades" in result["respuesta"]
    assert result["busquedas"] == 1


def test_budget_exhaustion_abstains_without_partial_answer():
    result = rag.crear_agente_rag(max_busquedas=1).invoke(rag.estado_inicial("¿Hacen mandados?"))
    assert result["causa"] == "presupuesto_agotado"
    assert result["fuentes"] == [] and result["estado"] == "abstencion"
    result = rag.crear_agente_rag(max_decisiones=1).invoke(rag.estado_inicial(PREGUNTA))
    assert result["causa"] == "presupuesto_agotado" and result["decisiones"] == 1


def test_repeated_search_blocked_before_second_execution():
    def stuck(state):
        return rag.DecisionRAG(accion="buscar", consulta="UNICORNIOS", motivo="Falla inyectada")

    result = rag.crear_agente_rag(decisor=stuck).invoke(rag.estado_inicial(PREGUNTA))
    assert result["busquedas"] == 1 and result["causa"] == "sin_progreso"


def test_premature_answer_blocked_by_program():
    def eager(state):
        return rag.DecisionRAG(accion="responder", motivo="Intento de responder sin buscar")

    result = rag.crear_agente_rag(decisor=eager).invoke(rag.estado_inicial(PREGUNTA))
    assert result["causa"] == "respuesta_prematura"
    assert result["fuentes"] == [] and result["busquedas"] == 0


@pytest.mark.parametrize("valor", [0, -1, True, 1.5, 7])
def test_agent_budgets_validated(valor):
    with pytest.raises(ValueError):
        rag.crear_agente_rag(max_busquedas=valor)


def test_unknown_action_and_provider_error_do_not_become_success():
    with pytest.raises(ValueError):
        rag.DecisionRAG(accion="pagar", motivo="Acción no permitida")

    def broken(state):
        raise ConnectionError("Falla de API simulada")

    with pytest.raises(ConnectionError):
        rag.crear_agente_rag(decisor=broken).invoke(rag.estado_inicial(PREGUNTA))


class ModeloDoble:
    """Doble de protocolo: observa prompts/schemas, sin acceder a la API."""

    def __init__(self, reject=False, forged=False):
        self.calls = []
        self.reject = reject
        self.forged = forged

    def with_structured_output(self, schema):
        parent = self

        class Bound:
            def invoke(self, messages):
                parent.calls.append(schema.__name__)
                assert messages[0][0] == "system" and messages[-1][0] == "human"
                payload = json.loads(messages[-1][1])
                if schema is rag.DecisionRAG:
                    return rag.politica_offline(payload)
                if schema is rag.RevisionFidelidad:
                    return {"respaldada": not parent.reject, "motivo": "Dictamen de prueba"}
                return rag.RespuestaRAG(abstencion=False, afirmaciones=[rag.Afirmacion(
                    texto=f["texto"], cita_literal=f["texto"],
                    fuente="FUENTE-INVENTADA" if parent.forged else f["id"],
                ) for f in payload["evidencia"]])

        return Bound()


def test_live_protocol_routes_rewrites_generates_and_grades_with_doubles(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    decision, generation = ModeloDoble(), ModeloDoble()
    app = rag.crear_agente_rag(mode="live", modelo=decision, modelo_respuesta=generation, k=1)
    result = app.invoke(rag.estado_inicial("¿Hacen mandados?"))
    assert result["estado"] == "respondido"
    assert len(decision.calls) == 3 and generation.calls == ["RespuestaRAG", "RevisionFidelidad"]
    assert result["llamadas_llm"] == 5


@pytest.mark.parametrize("forged,cause", [(True, "respuesta_no_validada"),
                                         (False, "fidelidad_no_aprobada")])
def test_live_forged_citation_and_negative_judge_fail_closed(monkeypatch, forged, cause):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = ModeloDoble(reject=True, forged=forged)
    result = rag.rag_clasico(PREGUNTA, mode="live", modelo=model)
    assert result["estado"] == "abstencion" and result["causa"] == cause
    assert result["fuentes"] == []


def test_current_model_roles_and_embedding_endpoint_are_distinct(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for name in ("OPENAI_MODEL", "OPENAI_MODEL_AGENT", "OPENAI_MODEL_RAG", "OPENAI_EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert config.model_name("default") == "gpt-6-luna"
    assert config.model_name("agent") == "gpt-6.1-sol"
    assert config.model_name("rag") == "gpt-6-luna"
    assert config.model_name("embeddings") == "text-embedding-3-large"
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    assert config.model_name("embeddings") == "text-embedding-3-small"
    with pytest.raises(ValueError, match="embeddings"):
        config.chat_model("embeddings")


CASOS = json.loads(files("henry_agents").joinpath("data/rag_casos.json").read_text("utf-8"))


@pytest.mark.parametrize("caso", CASOS, ids=lambda c: c["id"])
def test_same_corpus_and_k_comparison_cases(caso):
    indice = rag.IndiceLexico()
    base = rag.rag_clasico(caso["pregunta"], indice=indice, k=1)
    agent = rag.crear_agente_rag(indice=indice, k=1).invoke(rag.estado_inicial(caso["pregunta"]))
    assert base["estado"] == caso["clasico"]
    assert agent["estado"] == caso["agentico"]
    if agent["estado"] == "respondido":
        cited = {s.rsplit("-c", 1)[0] if "-c" in s else s for s in agent["fuentes"]}
        assert set(caso["documentos"]) <= cited
    assert agent["busquedas"] <= 3 and agent["decisiones"] <= 4


def test_evaluation_report_separates_recovery_from_expected_abstention():
    spec = importlib.util.spec_from_file_location("eval_rag", ROOT / "scripts/evaluate_rag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.evaluar()
    assert report["n"] == 12 and report["status"] == "passed"
    assert report["metricas"]["clasico"]["respondidas"] == 7
    assert report["metricas"]["agentico"]["respondidas"] == 11
    assert report["metricas"]["agentico"]["llamadas_llm"] == 0
