"""Cache de embeddings del taller: completo, coherente y sin llamadas en offline."""

import math

import pytest

from henry_agents import embeddings_comics


def test_cache_covers_every_phrase_and_query_with_one_dimension():
    cache = embeddings_comics.leer_cache()
    textos = [frase["texto"] for frase in embeddings_comics.FRASES] + embeddings_comics.CONSULTAS
    assert set(cache["vectores"]) == set(textos)
    assert {len(vector) for vector in cache["vectores"].values()} == {cache["dimensiones"]}
    assert cache["modelo"].startswith("text-embedding-")


def test_cached_vectors_are_unit_length_so_dot_product_is_cosine():
    for vector in embeddings_comics.leer_cache()["vectores"].values():
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, abs_tol=1e-3)


def test_phrases_cross_three_themes_with_three_heroes():
    pares = {(f["heroe"], f["tema"]) for f in embeddings_comics.FRASES}
    for heroe in ("batman", "spiderman", "fantasticos", "ninguno"):
        for tema in ("investigar", "comunidad", "ciencia"):
            assert (heroe, tema) in pares


def test_offline_reads_cache_in_order_without_calling_openai(monkeypatch):
    def prohibido(_textos):
        raise AssertionError("offline no debe llamar al proveedor")

    monkeypatch.setattr(embeddings_comics, "_pedir_a_openai", prohibido)
    textos = [embeddings_comics.CONSULTAS[1], embeddings_comics.FRASES[0]["texto"]]
    vectores = embeddings_comics.obtener_embeddings(textos, mode="offline")
    cache = embeddings_comics.leer_cache()["vectores"]
    assert vectores == [cache[textos[0]], cache[textos[1]]]


def test_offline_unknown_phrase_explains_how_to_continue():
    with pytest.raises(ValueError, match="modo live"):
        embeddings_comics.obtener_embeddings(["Una frase que no está guardada"], mode="offline")


def test_live_uses_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    monkeypatch.setattr(embeddings_comics, "_pedir_a_openai", lambda textos: [[1.0, 0.0]] * len(textos))
    assert embeddings_comics.obtener_embeddings(["a", "b"], mode="live") == [[1.0, 0.0], [1.0, 0.0]]
