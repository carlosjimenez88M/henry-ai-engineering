"""Experimentos pequeños de fragmentación, fusión y evaluación para el taller RAG.

Son funciones puras: no leen archivos, construyen modelos ni llaman a una API.
Las métricas describen los IDs de un caso, no la calidad general de un sistema.
"""

from collections.abc import Iterable, Mapping
from math import fsum


def _entero_positivo(valor, nombre):
    if isinstance(valor, bool) or not isinstance(valor, int) or valor <= 0:
        raise ValueError(f"{nombre} debe ser un entero positivo, sin booleanos")


def _como_lista(valores, nombre):
    if isinstance(valores, (str, bytes, Mapping)) or not isinstance(valores, Iterable):
        raise ValueError(f"{nombre} debe ser una colección de IDs, no texto ni un diccionario")
    return list(valores)


def _lista_ids(valores, nombre, *, permitir_duplicados=False):
    resultado = _como_lista(valores, nombre)
    if any(not isinstance(valor, str) or not valor.strip() for valor in resultado):
        raise ValueError(f"{nombre} debe contener IDs de texto no vacíos")
    if not permitir_duplicados and len(resultado) != len(set(resultado)):
        raise ValueError(f"{nombre} no admite IDs duplicados")
    return resultado


def dividir_por_caracteres(texto: str, tamano: int = 80, solapamiento: int = 20) -> list[dict]:
    """Divide sin alterar el texto, con posiciones Python: inicio incluido y fin excluido.

    El tamaño cuenta caracteres de Python, no bytes ni tokens. Los IDs CAR-01,
    CAR-02, etc. identifican fragmentos de esta entrada. Una cadena vacía devuelve
    una lista vacía. La última ventana termina una sola vez al alcanzar el final.
    """
    if not isinstance(texto, str):
        raise ValueError("texto debe ser una cadena")
    _entero_positivo(tamano, "tamano")
    if (isinstance(solapamiento, bool) or not isinstance(solapamiento, int)
            or not 0 <= solapamiento < tamano):
        raise ValueError("solapamiento debe ser un entero entre 0 y tamano - 1, sin booleanos")

    resultado = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        resultado.append({
            "id": f"CAR-{len(resultado) + 1:02d}",
            "inicio": inicio,
            "fin": fin,
            "texto": texto[inicio:fin],
        })
        if fin == len(texto):
            break
        inicio = fin - solapamiento
    return resultado


def fusionar_rankings(listas_ids: Iterable[Iterable[str]], k: int = 4,
                      rrf_constante: int = 60) -> list[dict]:
    """Fusiona posiciones mediante RRF: suma 1 / (constante + posición).

    Las posiciones empiezan en 1; None indica ausencia en una lista. Los rankings
    contienen IDs únicos y ordenados, sin scores de coseno o conteos de palabras.
    Ante scores iguales se conserva la primera aparición al recorrer las listas
    en su orden de entrada. Ninguna de las listas originales se modifica.
    """
    _entero_positivo(k, "k")
    _entero_positivo(rrf_constante, "rrf_constante")
    if isinstance(listas_ids, (set, frozenset)):
        raise ValueError("listas_ids debe conservar el orden de los rankings; no uses un conjunto")
    rankings = _como_lista(listas_ids, "listas_ids")
    posiciones_por_id = {}
    for indice_lista, ranking in enumerate(rankings):
        if isinstance(ranking, (set, frozenset)):
            raise ValueError("Cada ranking debe conservar un orden; no uses un conjunto")
        ids = _lista_ids(ranking, f"ranking {indice_lista + 1}")
        for posicion, identificador in enumerate(ids, start=1):
            if identificador not in posiciones_por_id:
                posiciones_por_id[identificador] = [None] * len(rankings)
            posiciones_por_id[identificador][indice_lista] = posicion

    resultado = [
        {
            "id": identificador,
            "rrf_score": fsum(1 / (rrf_constante + posicion)
                              for posicion in posiciones if posicion is not None),
            "posiciones": posiciones,
        }
        for identificador, posiciones in posiciones_por_id.items()
    ]
    # sorted conserva el orden de entrada cuando las claves de ordenación empatan.
    return sorted(resultado, key=lambda fila: fila["rrf_score"], reverse=True)[:k]


def evaluar_recuperacion(ids_recuperados: Iterable[str], ids_esperados: Iterable[str]) -> dict:
    """Calcula precisión y recall contra las fuentes esperadas para una pregunta.

    Precisión = aciertos / recuperados; recall = aciertos / esperados. Los esperados
    forman un conjunto no vacío: repetir un ID esperado no cambia la medida.
    Recuperar IDs duplicados se rechaza para no inflar los resultados. Sin ningún
    recuperado, ambas medidas son cero y la cobertura completa es falsa.
    """
    recuperados = set(_lista_ids(ids_recuperados, "ids_recuperados"))
    esperados = set(_lista_ids(ids_esperados, "ids_esperados", permitir_duplicados=True))
    if not esperados:
        raise ValueError("ids_esperados no puede estar vacío")
    aciertos = len(recuperados & esperados)
    return {
        "precision": aciertos / len(recuperados) if recuperados else 0.0,
        "recall": aciertos / len(esperados),
        "cobertura_completa": esperados <= recuperados,
    }
