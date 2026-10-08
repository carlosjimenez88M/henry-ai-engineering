"""Verifica caché real, persistencia vectorial, filtros y controles de evidencia sin red."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from qdrant_client import models

from henry_agents import rag_taller as taller

SMALL, LARGE = taller.MODELOS_EMBEDDING


@pytest.fixture
def fragmentos():
    return taller.fragmentar(taller.cargar_documentos())


@pytest.fixture
def base(tmp_path, fragmentos):
    with taller.abrir_base(tmp_path / "qdrant") as client:
        taller.crear_indice(client, fragmentos,
                            taller.vectores_para([f["texto"] for f in fragmentos], SMALL),
                            "biblioteca", SMALL)
        yield client


def test_corpus_is_original_fictitious_and_independent_each_read(fragmentos):
    assert len(fragmentos) == 14
    assert len({f["id"] for f in fragmentos}) == 14
    assert sum(not f["vigente"] for f in fragmentos) == 2
    assert next(f for f in fragmentos if f["id"] == "CC-01-P1")["texto"].find("siete") >= 0
    primero = taller.cargar_documentos()
    primero[0]["texto"] = "cambio de estudiante"
    assert taller.cargar_documentos()[0]["texto"] != primero[0]["texto"]


def test_fragmenting_preserves_paragraphs_and_rejects_duplicate_ids():
    documentos = taller.cargar_documentos()
    with pytest.raises(ValueError, match="únicos"):
        taller.fragmentar([documentos[0], documentos[0]])


def test_real_cache_provenance_dimensions_queries_and_fresh_vectors(fragmentos):
    cache = taller.leer_cache()
    assert cache["corpus_sha256"] == taller.sha_corpus()
    for modelo, dimensiones in taller.MODELOS_EMBEDDING.items():
        datos = cache["modelos"][modelo]
        assert datos["origen"] == "openai_embeddings_api"
        assert datos["dimensiones"] == dimensiones
        assert datos["cantidad_textos"] == 23
        assert datos["usage"]["prompt_tokens"] > 0
        assert datos["usage"]["total_tokens"] >= datos["usage"]["prompt_tokens"]
        assert datos["generado_utc"]
        textos = [f["texto"] for f in fragmentos] + taller.CONSULTAS + taller.FRASES
        assert all(len(v) == dimensiones for v in taller.vectores_para(textos, modelo))
        original = taller.vectores_para([textos[0]], modelo)
        original[0][0] = 100000
        assert taller.vectores_para([textos[0]], modelo)[0][0] != 100000


def test_missing_new_query_does_not_silently_use_fake_embedding():
    with pytest.raises(taller.CacheNoDisponible, match="mode='live'"):
        taller.vectores_para(["Texto de un alumno que nunca se ha embebido"], SMALL)
    with pytest.raises(ValueError, match="offline o live"):
        taller.vectores_para(taller.CONSULTAS[:1], SMALL, mode="automático")
    with pytest.raises(taller.EspacioVectorialIncompatible):
        taller.vectores_para(taller.CONSULTAS[:1], "otro_modelo")


@pytest.mark.parametrize("cambio", ["corpus", "texto", "dimension", "origen", "vacio"])
def test_cache_corruption_is_detected(tmp_path, cambio):
    import json
    cache = taller.leer_cache()
    modelo = cache["modelos"][SMALL]
    primera = next(iter(modelo["entradas"].values()))
    if cambio == "corpus":
        cache["corpus_sha256"] = "corpus antiguo"
    elif cambio == "texto":
        primera["texto"] += " cambio"
    elif cambio == "dimension":
        modelo["dimensiones"] = 10
    elif cambio == "origen":
        modelo["origen"] = "vector manual"
    else:
        modelo["entradas"] = {}
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    with pytest.raises(taller.CacheNoDisponible):
        taller.leer_cache(path)


def test_missing_cache_is_actionable(tmp_path):
    with pytest.raises(taller.CacheNoDisponible, match="--live"):
        taller.leer_cache(tmp_path / "no_existe.json")


def test_qdrant_upsert_is_idempotent_and_filters_archived(base, fragmentos):
    assert base.count("biblioteca").count == len(fragmentos)
    taller.crear_indice(base, fragmentos,
                        taller.vectores_para([f["texto"] for f in fragmentos], SMALL),
                        "biblioteca", SMALL)
    assert base.count("biblioteca").count == len(fragmentos)
    filtrados = taller.buscar(base, "biblioteca", taller.CONSULTAS[1], SMALL, k=10)
    historicos = taller.buscar(base, "biblioteca", taller.CONSULTAS[1], SMALL, k=10, vigentes=False)
    assert all(h.vigente for h in filtrados)
    assert any(not h.vigente for h in historicos)
    assert all(-1.0001 <= h.score <= 1.0001 for h in historicos)
    assert filtrados == sorted(filtrados, key=lambda h: h.score, reverse=True)


def test_qdrant_persistence_reopens_same_points_and_releases_file_even_on_failure(tmp_path, fragmentos):
    path = tmp_path / "qdrant"
    vectores = taller.vectores_para([f["texto"] for f in fragmentos], SMALL)
    with pytest.raises(RuntimeError, match="error del ejercicio"):
        with taller.abrir_base(path) as base:
            taller.crear_indice(base, fragmentos, vectores, "persistente", SMALL)
            before = taller.buscar(base, "persistente", taller.CONSULTAS[0], SMALL)
            raise RuntimeError("error del ejercicio")
    with taller.abrir_base(path) as base:
        assert base.count("persistente").count == 14
        after = taller.buscar(base, "persistente", taller.CONSULTAS[0], SMALL)
        assert [h.model_dump(exclude={"score"}) for h in before] == [
            h.model_dump(exclude={"score"}) for h in after
        ]
        assert [h.score for h in before] == pytest.approx([h.score for h in after], abs=1e-6)
        assert base.get_collection("persistente").config.metadata["embedding_modelo"] == SMALL


def test_query_and_existing_collection_enforce_same_model(base, fragmentos):
    with pytest.raises(taller.EspacioVectorialIncompatible):
        taller.buscar(base, "biblioteca", taller.CONSULTAS[0], LARGE)
    with pytest.raises(taller.EspacioVectorialIncompatible):
        taller.crear_indice(base, fragmentos,
                            taller.vectores_para([f["texto"] for f in fragmentos], LARGE),
                            "biblioteca", LARGE)
    # Una etiqueta diferente se rechaza incluso cuando las dimensiones todavía coinciden.
    punto = base.scroll("biblioteca", limit=1)[0][0]
    base.set_payload("biblioteca", {"embedding_modelo": LARGE}, [punto.id])
    with pytest.raises(taller.EspacioVectorialIncompatible, match="otro modelo"):
        taller.buscar(base, "biblioteca", taller.CONSULTAS[0], SMALL)


def test_same_size_collection_with_missing_manifest_is_rejected(base):
    base.create_collection("sin_manifiesto",
                           vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))
    with pytest.raises(taller.EspacioVectorialIncompatible, match="manifiesto"):
        taller.buscar(base, "sin_manifiesto", taller.CONSULTAS[0], SMALL)


@pytest.mark.parametrize("k", [0, 11, True, 3.5, "3"])
def test_k_is_validated_before_query(base, k):
    with pytest.raises(ValueError, match="entero"):
        taller.buscar(base, "biblioteca", taller.CONSULTAS[0], SMALL, k=k)


@pytest.mark.parametrize("tipo", ["faltante", "dimension", "nan", "cero"])
def test_malformed_vectors_are_rejected_before_creating_collection(base, fragmentos, tipo):
    vectores = taller.vectores_para([f["texto"] for f in fragmentos], SMALL)
    if tipo == "faltante":
        vectores.pop()
    elif tipo == "dimension":
        vectores[0].pop()
    elif tipo == "nan":
        vectores[0][0] = float("nan")
    else:
        vectores[0] = [0.0] * 1536
    with pytest.raises(ValueError):
        taller.crear_indice(base, fragmentos, vectores, "invalida", SMALL)
    assert not base.collection_exists("invalida")


def test_both_real_embedding_spaces_retrieve_paraphrase_exact_code_and_two_sources(tmp_path, fragmentos):
    with taller.abrir_base(tmp_path / "comparacion") as base:
        for modelo in taller.MODELOS_EMBEDDING:
            taller.crear_indice(base, fragmentos,
                                taller.vectores_para([f["texto"] for f in fragmentos], modelo),
                                modelo, modelo)
            assert "CC-01-P1" in {h.id for h in taller.buscar(base, modelo, taller.CONSULTAS[0], modelo)}
            assert taller.buscar(base, modelo, taller.CONSULTAS[2], modelo)[0].id == "CC-05-P1"
            assert {"CC-01-P1", "CC-01-P2"} <= {
                h.id for h in taller.buscar(base, modelo, taller.CONSULTAS[3], modelo, k=4)
            }
            # Top-k todavía devuelve resultados para una pregunta fuera del corpus.
            fuera = taller.buscar(base, modelo, taller.CONSULTAS[4], modelo)
            assert len(fuera) == 3 and all("Mongolia" not in h.texto for h in fuera)


def test_empty_context_abstains_without_model_or_config(monkeypatch):
    def no_llamar(*args, **kwargs):
        raise AssertionError("No debería intentar una llamada")
    monkeypatch.setattr(taller, "configure", no_llamar)
    monkeypatch.setattr(taller, "chat_model", no_llamar)
    for mode in ("offline", "live"):
        respuesta = taller.responder("¿Cuál es el plazo?", [], mode=mode)
        assert respuesta.estado == "sin_evidencia" and respuesta.citas == []
        assert respuesta.origen == "control_sin_contexto" and respuesta.llamadas_modelo == 0
        assert respuesta.modelo_usado is None and respuesta.uso_tokens is None


def test_offline_extractor_is_literal_and_does_not_claim_an_llm(base):
    hallazgos = taller.buscar(base, "biblioteca", taller.CONSULTAS[0], SMALL)
    respuesta = taller.responder(taller.CONSULTAS[0], hallazgos)
    assert respuesta.origen == "extractor_literal" and respuesta.llamadas_modelo == 0
    assert respuesta.respuesta == "\n\n".join(h.texto for h in hallazgos)
    assert taller.validar_citas(respuesta, hallazgos) == []


class DobleChat:
    model_name = "doble_local_sin_red"

    def __init__(self, respuesta, *, error=None, raw=None):
        self.respuesta = respuesta
        self.error = error
        self.raw = raw
        self.llamadas = []

    def with_structured_output(self, schema, include_raw):
        assert schema is taller.RespuestaRAG and include_raw
        return self

    def invoke(self, mensajes):
        self.llamadas.append(mensajes)
        if self.error:
            raise self.error
        return {"parsed": self.respuesta, "parsing_error": None, "raw": self.raw}


def respuesta_valida(hallazgo):
    return {"estado": "respondido", "respuesta": "El plazo es de siete días calendario.",
            "citas": [{"fragmento_id": hallazgo.id, "cita_literal": hallazgo.texto}],
            "origen": "modelo"}


def test_live_with_double_records_real_reported_usage_not_model_claims(monkeypatch, base):
    monkeypatch.setattr(taller, "configure", lambda mode: mode)
    hallazgos = taller.buscar(base, "biblioteca", taller.CONSULTAS[0], SMALL)
    salida = {**respuesta_valida(hallazgos[0]), "modelo_usado": "modelo_inventado",
              "uso_tokens": {"total_tokens": 999999}, "llamadas_modelo": 0}
    uso = {"input_tokens": 70, "output_tokens": 20, "total_tokens": 90}
    doble = DobleChat(salida, raw=AIMessage(content="", usage_metadata=uso))
    respuesta = taller.responder(taller.CONSULTAS[0], hallazgos, mode="live", modelo=doble)
    assert respuesta.modelo_usado == doble.model_name
    assert respuesta.uso_tokens == uso and respuesta.llamadas_modelo == 1
    assert len(doble.llamadas) == 1
    assert "No uses emojis" in doble.llamadas[0][0][1]
    assert hallazgos[0].texto in doble.llamadas[0][1][1]


def test_usage_missing_is_unknown_not_fake_zero(monkeypatch, base):
    monkeypatch.setattr(taller, "configure", lambda mode: mode)
    hallazgos = taller.buscar(base, "biblioteca", taller.CONSULTAS[0], SMALL)
    respuesta = taller.responder(taller.CONSULTAS[0], hallazgos, mode="live",
                                modelo=DobleChat(respuesta_valida(hallazgos[0])))
    assert respuesta.uso_tokens is None and respuesta.llamadas_modelo == 1


@pytest.mark.parametrize("cambio", ["id", "cita", "sin_citas", "abstencion_con_citas", "origen"])
def test_invalid_llm_evidence_is_rejected(monkeypatch, base, cambio):
    monkeypatch.setattr(taller, "configure", lambda mode: mode)
    hallazgos = taller.buscar(base, "biblioteca", taller.CONSULTAS[0], SMALL)
    salida = respuesta_valida(hallazgos[0])
    if cambio == "id":
        salida["citas"][0]["fragmento_id"] = "FUENTE-INVENTADA"
    elif cambio == "cita":
        salida["citas"][0]["cita_literal"] = "El plazo es de noventa días."
    elif cambio == "sin_citas":
        salida["citas"] = []
    elif cambio == "abstencion_con_citas":
        salida["estado"] = "sin_evidencia"
    else:
        salida["origen"] = "extractor_literal"
    with pytest.raises(taller.EvidenciaInvalida):
        taller.responder(taller.CONSULTAS[0], hallazgos, mode="live", modelo=DobleChat(salida))


def test_structural_citation_checks_do_not_claim_semantic_faithfulness(base):
    hallazgos = taller.buscar(base, "biblioteca", taller.CONSULTAS[0], SMALL)
    cita_correcta_respuesta_erronea = respuesta_valida(hallazgos[0])
    cita_correcta_respuesta_erronea["respuesta"] = "El plazo es de noventa días."
    assert taller.validar_citas(cita_correcta_respuesta_erronea, hallazgos) == []
    # Esta limitación deliberada se discute en clase: una cita real no demuestra cada afirmación.


def test_live_semantic_abstention_and_provider_errors_are_explicit_and_private(monkeypatch, base):
    monkeypatch.setattr(taller, "configure", lambda mode: mode)
    hallazgos = taller.buscar(base, "biblioteca", taller.CONSULTAS[4], SMALL)
    doble = DobleChat({"estado": "sin_evidencia", "respuesta": "El corpus no contiene ese dato.",
                      "citas": [], "origen": "modelo"})
    respuesta = taller.responder(taller.CONSULTAS[4], hallazgos, mode="live", modelo=doble)
    assert respuesta.estado == "sin_evidencia" and respuesta.llamadas_modelo == 1
    with pytest.raises(taller.ProveedorRAGError) as error:
        taller.responder(taller.CONSULTAS[4], hallazgos, mode="live",
                         modelo=DobleChat(None, error=RuntimeError("credencial_privada_ejemplo")))
    assert "credencial_privada_ejemplo" not in str(error.value)


def test_live_embedding_batch_uses_sdk_order_and_provider_reported_tokens(monkeypatch):
    monkeypatch.setattr(taller, "configure", lambda mode: mode)
    peticiones = []
    def crear(**kwargs):
        peticiones.append(kwargs)
        return SimpleNamespace(model=SMALL, usage=SimpleNamespace(prompt_tokens=9, total_tokens=9),
                               data=[SimpleNamespace(index=1, embedding=[0.2]*1536),
                                     SimpleNamespace(index=0, embedding=[0.1]*1536)])
    cliente = SimpleNamespace(embeddings=SimpleNamespace(create=crear))
    resultado = taller.solicitar_embeddings(["uno", "dos"], SMALL, cliente=cliente)
    assert resultado["vectores"][0][0] == 0.1 and resultado["vectores"][1][0] == 0.2
    assert resultado["usage"] == {"prompt_tokens": 9, "total_tokens": 9}
    assert peticiones == [{"model": SMALL, "input": ["uno", "dos"]}]


def test_live_embedding_provider_error_does_not_silently_substitute_cache(monkeypatch):
    monkeypatch.setattr(taller, "configure", lambda mode: mode)
    def crear(**kwargs):
        raise RuntimeError("credencial_privada_ejemplo")
    cliente = SimpleNamespace(embeddings=SimpleNamespace(create=crear))
    with pytest.raises(taller.ProveedorRAGError) as error:
        taller.solicitar_embeddings(taller.CONSULTAS[:1], SMALL, cliente=cliente)
    assert "credencial_privada_ejemplo" not in str(error.value)


def test_modified_hallazgo_instances_are_revalidated(base):
    hallazgos = taller.buscar(base, "biblioteca", taller.CONSULTAS[0], SMALL)
    modificados = deepcopy(hallazgos)
    modificados[0].vigente = {"dato": "invalido"}
    with pytest.warns(UserWarning, match="Pydantic serializer"):
        with pytest.raises(ValidationError):
            taller.responder(taller.CONSULTAS[0], modificados)
