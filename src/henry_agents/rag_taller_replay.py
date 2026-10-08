"""Reproduce respuestas reales guardadas, sin llamadas nuevas ni reemplazos ocultos."""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from henry_agents.rag_taller import (
    MODELOS_EMBEDDING,
    CacheNoDisponible,
    Hallazgo,
    RespuestaRAG,
    cargar_documentos,
    fragmentar,
    sha_corpus,
    validar_citas,
)

RESPUESTAS_PATH = Path(__file__).with_name("data") / "rag_taller_respuestas.json"
REPARAR_ARCHIVO = (
    "Revisar rag_taller_respuestas.json o regenerar las respuestas con el docente "
    "mediante llamadas live explícitas."
)


def _contexto_validado(hallazgos):
    if not isinstance(hallazgos, (list, tuple)):
        raise CacheNoDisponible("El contexto debe ser una lista de hallazgos. Inspecciona el contexto.")
    try:
        return [Hallazgo.model_validate(
            h.model_dump(warnings=False) if isinstance(h, Hallazgo) else h, strict=True
        ) for h in hallazgos]
    except (ValidationError, TypeError, ValueError):
        raise CacheNoDisponible("El contexto no tiene la forma esperada. Inspecciona los hallazgos.") from None


def firma_contexto(pregunta, hallazgos):
    """Pregunta y orden/IDs/textos/vigencia exactos; scores float32 no integran la firma."""
    if not isinstance(pregunta, str) or not pregunta.strip():
        raise CacheNoDisponible("Se necesita una pregunta de texto no vacía. Revisa la pregunta.")
    hallazgos = _contexto_validado(hallazgos)
    datos = {
        "pregunta": pregunta,
        "fragmentos": [
            {"id": h.id, "texto": h.texto, "vigente": h.vigente} for h in hallazgos
        ],
    }
    texto = json.dumps(datos, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _validar_registro(firma, registro, corpus):
    if not isinstance(firma, str) or not isinstance(registro, dict):
        raise CacheNoDisponible("Registro de respuesta incompatible. " + REPARAR_ARCHIVO)
    if not {"pregunta", "contexto", "resultado"} <= registro.keys():
        raise CacheNoDisponible("Registro de respuesta incompleto. " + REPARAR_ARCHIVO)
    hallazgos = _contexto_validado(registro["contexto"])
    if len({h.id for h in hallazgos}) != len(hallazgos):
        raise CacheNoDisponible("El contexto preparado contiene IDs duplicados. " + REPARAR_ARCHIVO)
    for h in hallazgos:
        original = corpus.get(h.id)
        if original is None or (h.texto, h.vigente) != (original["texto"], original["vigente"]):
            raise CacheNoDisponible("El contexto preparado cambió respecto del corpus. " + REPARAR_ARCHIVO)
    if firma_contexto(registro["pregunta"], hallazgos) != firma:
        raise CacheNoDisponible("La firma no coincide con pregunta y contexto guardados. " + REPARAR_ARCHIVO)
    try:
        respuesta = RespuestaRAG.model_validate(registro["resultado"], strict=True)
    except (ValidationError, TypeError, ValueError):
        raise CacheNoDisponible("La respuesta preparada no cumple el contrato. " + REPARAR_ARCHIVO) from None
    if respuesta.origen != "modelo" or respuesta.llamadas_modelo != 1:
        raise CacheNoDisponible("El registro debe documentar una generación real anterior. " + REPARAR_ARCHIVO)
    if validar_citas(respuesta, hallazgos):
        raise CacheNoDisponible("Las citas preparadas no son literales del contexto. " + REPARAR_ARCHIVO)
    return hallazgos


def leer_respuestas(path=None):
    """Valida manifiesto, corpus actual, firmas y contratos de todas las respuestas."""
    path = RESPUESTAS_PATH if path is None else Path(path)
    if not path.is_file():
        raise CacheNoDisponible("Faltan las respuestas preparadas por el docente. " + REPARAR_ARCHIVO)
    try:
        datos = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CacheNoDisponible("No se puede leer el JSON de respuestas. " + REPARAR_ARCHIVO) from None
    if not isinstance(datos, dict):
        raise CacheNoDisponible("El archivo debe contener un objeto JSON. " + REPARAR_ARCHIVO)
    if (type(datos.get("version")) is not int or datos["version"] != 1
            or datos.get("origen") != "openai_chat_api"):
        raise CacheNoDisponible("El manifiesto no declara respuestas reales compatibles. " + REPARAR_ARCHIVO)
    if datos.get("corpus_sha256") != sha_corpus():
        raise CacheNoDisponible("Las respuestas corresponden a otro corpus. " + REPARAR_ARCHIVO)
    if (not isinstance(datos.get("embedding_modelo"), str)
            or datos["embedding_modelo"] not in MODELOS_EMBEDDING
            or type(datos.get("k")) is not int or not 1 <= datos["k"] <= 10
            or type(datos.get("vigentes")) is not bool
            or not isinstance(datos.get("generado_utc"), str) or not datos["generado_utc"]):
        raise CacheNoDisponible("El manifiesto de preparación está incompleto. " + REPARAR_ARCHIVO)
    entradas = datos.get("entradas")
    if not isinstance(entradas, dict) or not entradas:
        raise CacheNoDisponible("Faltan registros de respuestas válidos. " + REPARAR_ARCHIVO)
    corpus = {f["id"]: f for f in fragmentar(cargar_documentos())}
    for firma, registro in entradas.items():
        hallazgos = _validar_registro(firma, registro, corpus)
        if len(hallazgos) != datos["k"] or (datos["vigentes"] and any(not h.vigente for h in hallazgos)):
            raise CacheNoDisponible("El contexto contradice el manifiesto de recuperación. " + REPARAR_ARCHIVO)
    return datos


def reproducir_respuesta(pregunta, hallazgos):
    """Repite la respuesta al MISMO contexto: cero API, sin fallback.

    Los tokens, modelo y llamadas_modelo del resultado describen la generación
    original guardada. La reproducción actual no consume tokens ni llama modelos.
    """
    hallazgos = _contexto_validado(hallazgos)
    datos = leer_respuestas()
    registro = datos["entradas"].get(firma_contexto(pregunta, hallazgos))
    if registro is None:
        raise CacheNoDisponible(
            "No existe una respuesta preparada para esta pregunta y estos fragmentos. "
            "Inspecciona el contexto o ejecuta responder(..., mode='live') explícitamente."
        )
    respuesta = RespuestaRAG.model_validate(registro["resultado"], strict=True)
    if validar_citas(respuesta, hallazgos):
        raise CacheNoDisponible("Las citas de la respuesta no coinciden con el contexto. " + REPARAR_ARCHIVO)
    return respuesta
