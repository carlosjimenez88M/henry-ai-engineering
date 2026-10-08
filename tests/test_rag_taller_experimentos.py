"""Casos que muestran pérdida de contexto, consenso de rankings y cobertura parcial."""

from copy import deepcopy

import pytest

from henry_agents.rag_taller_experimentos import (
    dividir_por_caracteres,
    evaluar_recuperacion,
    fusionar_rankings,
)


def test_character_cut_can_separate_a_rule_and_overlap_can_preserve_its_phrase():
    texto = "Álbum comunitario. El préstamo dura siete días. Se necesita identificación."
    frase = "El préstamo dura siete días."
    inicio_frase = texto.index(frase)
    fin_frase = inicio_frase + len(frase)
    corte = inicio_frase + 12

    separados = dividir_por_caracteres(texto, tamano=corte, solapamiento=0)
    assert separados[0]["fin"] < fin_frase
    assert not any(frase in parte["texto"] for parte in separados)
    solapados = dividir_por_caracteres(texto, tamano=corte, solapamiento=12)
    assert any(frase in parte["texto"] for parte in solapados)
    for parte in solapados:
        assert parte["texto"] == texto[parte["inicio"]:parte["fin"]]
    assert "Á" in solapados[0]["texto"] and "días" in texto


@pytest.mark.parametrize("tamano,solapamiento", [(1, 0), (5, 0), (5, 2), (5, 4), (40, 0)])
def test_chunk_walk_covers_every_character_progresses_and_stops_at_last_end(tamano, solapamiento):
    texto = "áéíóú: préstamo."
    partes = dividir_por_caracteres(texto, tamano=tamano, solapamiento=solapamiento)
    assert partes[0]["inicio"] == 0
    assert partes[-1]["fin"] == len(texto)
    assert len([p for p in partes if p["fin"] == len(texto)]) == 1
    cubiertos = set()
    for parte in partes:
        cubiertos.update(range(parte["inicio"], parte["fin"]))
        assert 0 < len(parte["texto"]) <= tamano
        assert parte["texto"] == texto[parte["inicio"]:parte["fin"]]
    assert cubiertos == set(range(len(texto)))
    for anterior, siguiente in zip(partes, partes[1:]):
        assert anterior["inicio"] < siguiente["inicio"]
        assert siguiente["inicio"] == anterior["fin"] - solapamiento


def test_chunks_preserve_whitespace_and_have_local_predictable_ids():
    partes = dividir_por_caracteres("  á\n\n b  ", tamano=4, solapamiento=1)
    assert [p["id"] for p in partes] == ["CAR-01", "CAR-02", "CAR-03"]
    assert partes[0] == {"id": "CAR-01", "inicio": 0, "fin": 4, "texto": "  á\n"}
    assert partes[-1]["texto"].endswith("  ")


def test_empty_short_and_exact_length_text_do_not_get_duplicate_last_chunks():
    assert dividir_por_caracteres("") == []
    assert dividir_por_caracteres("corto", tamano=10, solapamiento=9) == [
        {"id": "CAR-01", "inicio": 0, "fin": 5, "texto": "corto"},
    ]
    assert len(dividir_por_caracteres("cuatro", tamano=6, solapamiento=5)) == 1


@pytest.mark.parametrize("tamano", [0, -1, True, False, 4.0, "4", None])
def test_chunk_size_must_be_positive_integer_not_boolean(tamano):
    with pytest.raises(ValueError, match="tamano"):
        dividir_por_caracteres("texto", tamano=tamano, solapamiento=0)


@pytest.mark.parametrize("solapamiento", [-1, 5, 6, True, False, 1.0, "1", None])
def test_overlap_cannot_skip_or_prevent_progress(solapamiento):
    with pytest.raises(ValueError, match="solapamiento"):
        dividir_por_caracteres("texto", tamano=5, solapamiento=solapamiento)


@pytest.mark.parametrize("texto", [None, 42, b"texto", ["texto"]])
def test_chunks_reject_non_text_input(texto):
    with pytest.raises(ValueError, match="texto"):
        dividir_por_caracteres(texto)


def test_rrf_rewards_agreement_across_rankings_without_combining_raw_scores():
    rankings = [["literal-primer-lugar", "consenso", "otro"], ["vector-primer-lugar", "consenso"]]
    original = deepcopy(rankings)
    fusion = fusionar_rankings(rankings)
    assert fusion[0]["id"] == "consenso"
    assert fusion[0]["posiciones"] == [2, 2]
    assert fusion[0]["rrf_score"] == pytest.approx(2 / 62)
    assert next(p for p in fusion if p["id"] == "literal-primer-lugar")["posiciones"] == [1, None]
    assert next(p for p in fusion if p["id"] == "vector-primer-lugar")["posiciones"] == [None, 1]
    assert rankings == original


def test_rrf_ties_follow_first_appearance_and_have_no_extra_duplicates():
    rankings = [["Z", "A"], ["A", "Z"]]
    fusion = fusionar_rankings(rankings, k=10)
    assert [p["id"] for p in fusion] == ["Z", "A"]
    assert fusion[0]["rrf_score"] == fusion[1]["rrf_score"]
    assert fusionar_rankings(rankings, k=1) == fusion[:1]
    assert [p["id"] for p in fusionar_rankings(list(reversed(rankings)))] == ["A", "Z"]


def test_rrf_records_empty_rankings_and_accepts_no_candidates():
    assert fusionar_rankings([]) == []
    assert fusionar_rankings([[], []]) == []
    fusion = fusionar_rankings([[], ["A"], []])
    assert fusion == [{"id": "A", "rrf_score": 1 / 61, "posiciones": [None, 1, None]}]


def test_rrf_rejects_duplicate_ids_within_one_ranking_but_accepts_cross_ranking_repeats():
    with pytest.raises(ValueError, match="duplicados"):
        fusionar_rankings([["A", "A"], ["B"]])
    assert len(fusionar_rankings([["A"], ["A"]])) == 1


@pytest.mark.parametrize("campo", ["k", "rrf_constante"])
@pytest.mark.parametrize("valor", [0, -1, True, False, 2.5, "2", None])
def test_rrf_parameters_must_be_positive_integers(campo, valor):
    with pytest.raises(ValueError, match=campo):
        fusionar_rankings([["A"]], **{campo: valor})


@pytest.mark.parametrize("rankings", ["A", None, {"A": 0.9}, ["A"], [["A", ""]],
                                       [[True]], [[{"id": "A", "score": 0.8}]], [{"A", "B"}],
                                       {("A", "B"), ("B", "A")}])
def test_rrf_rejects_raw_score_records_non_ids_and_unordered_rankings(rankings):
    with pytest.raises(ValueError):
        fusionar_rankings(rankings)


def test_incomplete_recovery_can_have_perfect_precision_but_partial_recall():
    resultado = evaluar_recuperacion(["plazo"], ["plazo", "requisitos"])
    assert resultado == {"precision": 1.0, "recall": 0.5, "cobertura_completa": False}


def test_extra_results_lower_precision_even_with_complete_evidence():
    resultado = evaluar_recuperacion(["plazo", "requisitos", "irrelevante"], {"plazo", "requisitos"})
    assert resultado["precision"] == pytest.approx(2 / 3)
    assert resultado["recall"] == 1.0
    assert resultado["cobertura_completa"] is True


def test_empty_or_unrelated_recovery_is_not_complete_and_has_zero_precision_recall():
    esperado = {"precision": 0.0, "recall": 0.0, "cobertura_completa": False}
    assert evaluar_recuperacion([], ["A"]) == esperado
    assert evaluar_recuperacion(["B"], ["A"]) == esperado


def test_metrics_are_invariant_to_order_and_duplicate_expected_ids():
    antes = evaluar_recuperacion(["A", "B"], ["A", "C"])
    despues = evaluar_recuperacion(["B", "A"], ["C", "A", "A"])
    assert antes == despues == {"precision": 0.5, "recall": 0.5, "cobertura_completa": False}


def test_metrics_reject_duplicate_recovery_and_empty_ground_truth():
    with pytest.raises(ValueError, match="duplicados"):
        evaluar_recuperacion(["A", "A"], ["A"])
    with pytest.raises(ValueError, match="vacío"):
        evaluar_recuperacion(["A"], [])


@pytest.mark.parametrize("recuperados,esperados", [("A", ["A"]), (["A"], "A"),
                                                (["A"], None), ([True], ["A"]),
                                                (["A"], [""]), ([{"id": "A"}], ["A"])])
def test_metrics_reject_wrong_collection_or_id_shapes(recuperados, esperados):
    with pytest.raises(ValueError):
        evaluar_recuperacion(recuperados, esperados)
