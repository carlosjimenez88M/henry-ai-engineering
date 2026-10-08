"""Replay comprobable: mismo contexto y corpus, errores claros, cero red y cero API."""

import json
from copy import deepcopy

import pytest

from henry_agents import rag_taller as taller
from henry_agents import rag_taller_replay as replay


@pytest.fixture(autouse=True)
def prohibir_modelos_y_red(monkeypatch):
    def prohibido(*args, **kwargs):
        raise AssertionError("El replay nunca debe llamar a modelos ni a embeddings")
    monkeypatch.setattr(taller, "configure", prohibido)
    monkeypatch.setattr(taller, "chat_model", prohibido)
    monkeypatch.setattr(taller, "solicitar_embeddings", prohibido)
    import openai
    monkeypatch.setattr(openai, "OpenAI", prohibido)


@pytest.fixture
def datos():
    return replay.leer_respuestas()


@pytest.fixture
def registros(datos):
    return list(datos["entradas"].values())


def guardar_cache(tmp_path, monkeypatch, datos):
    path = tmp_path / "respuestas.json"
    path.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(replay, "RESPUESTAS_PATH", path)
    return path


@pytest.mark.parametrize("numero", range(5))
def test_prepared_answers_replay_the_actual_qdrant_context_without_api(tmp_path, datos, numero):
    modelo = datos["embedding_modelo"]
    fragmentos = taller.fragmentar(taller.cargar_documentos())
    pregunta = taller.CONSULTAS[numero]
    with taller.abrir_base(tmp_path / "qdrant") as db:
        taller.crear_indice(db, fragmentos,
                            taller.vectores_para([f["texto"] for f in fragmentos], modelo),
                            "replay", modelo)
        contexto = taller.buscar(db, "replay", pregunta, modelo, k=datos["k"])
    respuesta = replay.reproducir_respuesta(pregunta, contexto)
    assert respuesta.origen == "modelo" and respuesta.modelo_usado
    # Son los metadatos históricos del archivo, no llamadas de la reproducción actual.
    assert respuesta.llamadas_modelo == 1 and respuesta.uso_tokens["total_tokens"] > 0
    assert taller.validar_citas(respuesta, contexto) == []
    assert respuesta.estado == ("sin_evidencia" if numero == 4 else "respondido")
    if numero == 4:
        assert respuesta.citas == []
    elif numero == 2:
        assert "COMIC-27-BARRIO" in respuesta.respuesta
    else:
        assert "siete" in respuesta.respuesta


@pytest.mark.parametrize("cambio", ["pregunta", "orden", "id", "texto", "vigencia", "menos_fragmentos"])
def test_different_question_or_evidence_never_replays_silently(registros, cambio):
    registro = deepcopy(registros[0])
    pregunta, contexto = registro["pregunta"], registro["contexto"]
    if cambio == "pregunta":
        pregunta += " Por favor."
    elif cambio == "orden":
        contexto.reverse()
    elif cambio == "id":
        contexto[0]["id"] = "CC-ID-DIFERENTE"
    elif cambio == "texto":
        contexto[0]["texto"] += " Texto adicional."
    elif cambio == "vigencia":
        contexto[0]["vigente"] = False
    else:
        contexto.pop()
    with pytest.raises(taller.CacheNoDisponible, match="mode='live'"):
        replay.reproducir_respuesta(pregunta, contexto)


def test_float32_score_variations_and_display_metadata_are_not_part_of_evidence_signature(registros):
    registro = deepcopy(registros[0])
    original = replay.firma_contexto(registro["pregunta"], registro["contexto"])
    for h in registro["contexto"]:
        h["score"] += 1e-7
        h["titulo"] = "Título para mostrar localmente"
        h["categoria"] = "otra etiqueta local"
    assert replay.firma_contexto(registro["pregunta"], registro["contexto"]) == original
    assert replay.reproducir_respuesta(registro["pregunta"], registro["contexto"]).estado == "respondido"


@pytest.mark.parametrize("valor", [None, "", 42, ["pregunta"]])
def test_invalid_question_shape_is_actionable(registros, valor):
    with pytest.raises(taller.CacheNoDisponible, match="pregunta"):
        replay.reproducir_respuesta(valor, registros[0]["contexto"])


@pytest.mark.parametrize("valor", [None, {}, "texto", [{"id": "fragmento_sin_campos"}]])
def test_invalid_context_shape_is_actionable(registros, valor):
    with pytest.raises(taller.CacheNoDisponible, match="contexto"):
        replay.reproducir_respuesta(registros[0]["pregunta"], valor)


def test_boolean_vigency_is_strict_not_coerced_from_strings(registros):
    contexto = deepcopy(registros[0]["contexto"])
    contexto[0]["vigente"] = "true"
    with pytest.raises(taller.CacheNoDisponible, match="contexto"):
        replay.firma_contexto(registros[0]["pregunta"], contexto)


@pytest.mark.parametrize("contenido", ["{", '"json escalar"', "[]", "null"])
def test_malformed_or_non_object_json_has_actionable_error(tmp_path, contenido):
    path = tmp_path / "respuestas.json"
    path.write_text(contenido, encoding="utf-8")
    with pytest.raises(taller.CacheNoDisponible, match="Revisar rag_taller_respuestas.json"):
        replay.leer_respuestas(path)


def test_missing_and_non_utf8_files_have_actionable_errors(tmp_path):
    with pytest.raises(taller.CacheNoDisponible, match="docente"):
        replay.leer_respuestas(tmp_path / "no_existe.json")
    path = tmp_path / "respuestas.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(taller.CacheNoDisponible, match="JSON"):
        replay.leer_respuestas(path)


@pytest.mark.parametrize("cambio", [
    {"version": 2}, {"version": True}, {"origen": "respuestas_inventadas"},
    {"corpus_sha256": "otro_corpus"}, {"embedding_modelo": []}, {"k": True},
    {"k": 0}, {"vigentes": "true"}, {"generado_utc": None}, {"entradas": []}, {"entradas": {}},
])
def test_invalid_manifest_or_entries_fail_before_replay(tmp_path, monkeypatch, datos, cambio):
    datos.update(cambio)
    guardar_cache(tmp_path, monkeypatch, datos)
    with pytest.raises(taller.CacheNoDisponible, match="Revisar rag_taller_respuestas.json"):
        replay.leer_respuestas()


def test_corpus_change_invalidates_all_prepared_answers(monkeypatch, registros):
    monkeypatch.setattr(replay, "sha_corpus", lambda: "sha_del_corpus_modificado")
    with pytest.raises(taller.CacheNoDisponible, match="otro corpus"):
        replay.reproducir_respuesta(registros[0]["pregunta"], registros[0]["contexto"])


@pytest.mark.parametrize("cambio", ["registro", "resultado", "contexto", "pregunta", "firma", "cita", "id_cita",
                                    "origen", "llamadas", "cantidad", "duplicado"])
def test_bad_record_shapes_signatures_and_citations_are_actionable(tmp_path, monkeypatch, datos, cambio):
    firma = next(iter(datos["entradas"]))
    registro = datos["entradas"][firma]
    if cambio == "registro":
        datos["entradas"][firma] = []
    elif cambio == "resultado":
        registro["resultado"] = {"dato": "incompleto"}
    elif cambio == "contexto":
        registro["contexto"] = None
    elif cambio == "pregunta":
        registro.pop("pregunta")
    elif cambio == "firma":
        registro["pregunta"] += " pregunta modificada"
    elif cambio == "cita":
        registro["resultado"]["citas"][0]["cita_literal"] = "El plazo es de noventa días."
    elif cambio == "id_cita":
        registro["resultado"]["citas"][0]["fragmento_id"] = "ID-INVENTADO"
    elif cambio == "origen":
        registro["resultado"]["origen"] = "extractor_literal"
    elif cambio == "llamadas":
        registro["resultado"]["llamadas_modelo"] = 0
    elif cambio == "cantidad":
        datos["k"] = 3
    else:
        registro["contexto"][1] = deepcopy(registro["contexto"][0])
    guardar_cache(tmp_path, monkeypatch, datos)
    with pytest.raises(taller.CacheNoDisponible):
        replay.leer_respuestas()


def test_forged_context_cannot_be_accepted_by_recomputing_its_signature(tmp_path, monkeypatch, datos):
    original = next(iter(datos["entradas"]))
    registro = datos["entradas"].pop(original)
    registro["contexto"][0]["texto"] += " El plazo inventado es de noventa días."
    firma_forged = replay.firma_contexto(registro["pregunta"], registro["contexto"])
    datos["entradas"][firma_forged] = registro
    guardar_cache(tmp_path, monkeypatch, datos)
    with pytest.raises(taller.CacheNoDisponible, match="respecto del corpus"):
        replay.reproducir_respuesta(registro["pregunta"], registro["contexto"])


def test_replayed_objects_are_independent_between_calls(registros):
    registro = registros[0]
    primera = replay.reproducir_respuesta(registro["pregunta"], registro["contexto"])
    primera.respuesta = "Cambio local del alumno."
    primera.citas.clear()
    segunda = replay.reproducir_respuesta(registro["pregunta"], registro["contexto"])
    assert segunda.respuesta != primera.respuesta and segunda.citas
