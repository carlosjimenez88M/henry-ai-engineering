"""Búsqueda por significado (embeddings) para la clase 2.

Un embedding convierte un texto en una lista de números: textos con significado parecido
quedan cerca. En offline usamos un MAPA HECHO A MANO de tres dimensiones para ver la idea
sin magia; en live, embeddings reales de OpenAI (text-embedding-3-small, 1536 números).
"""

import math

from henry_agents.config import MODELO_EMBEDDINGS, configure
from henry_agents.cultural import load_catalog
from henry_agents.retrieval import tokens

# Cada palabra ubicada a mano en tres ejes: (misterio, colaboración, tecnología).
MAPA = {
    "investigacion": (0.9, 0.1, 0.2),
    "detective": (0.95, 0.0, 0.1),
    "pistas": (0.9, 0.1, 0.1),
    "enigma": (0.95, 0.05, 0.05),
    "misterio": (1.0, 0.0, 0.0),
    "sospechoso": (0.9, 0.0, 0.0),
    "evidencia": (0.7, 0.2, 0.3),
    "preguntas": (0.6, 0.3, 0.0),
    "revision": (0.5, 0.4, 0.3),
    "equipo": (0.1, 0.95, 0.1),
    "cooperacion": (0.05, 1.0, 0.0),
    "juntos": (0.0, 0.9, 0.0),
    "vecinos": (0.1, 0.8, 0.0),
    "compartir": (0.0, 0.85, 0.05),
    "organizacion": (0.1, 0.8, 0.2),
    "herramientas": (0.1, 0.2, 0.9),
    "sensores": (0.3, 0.0, 0.95),
    "laboratorio": (0.2, 0.2, 0.9),
    "ciencia": (0.3, 0.2, 0.9),
    "inventario": (0.1, 0.2, 0.8),
    "seguridad": (0.3, 0.2, 0.6),
    "energia": (0.0, 0.2, 0.8),
    "calma": (0.6, 0.2, 0.0),
    "alegre": (0.0, 0.7, 0.1),
}
EJES = ("misterio", "colaboración", "tecnología")


def vector_de(texto):
    """Promedio de los vectores de las palabras conocidas del texto (None si no conoce ninguna)."""
    vectores = [MAPA[t] for t in tokens(texto) if t in MAPA]
    if not vectores:
        return None
    return tuple(sum(eje) / len(vectores) for eje in zip(*vectores, strict=True))


def similitud(a, b):
    """Similitud coseno: 1 = misma dirección (mismo significado), 0 = nada en común."""
    producto = sum(x * y for x, y in zip(a, b, strict=True))
    normas = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return producto / normas if normas else 0.0


def _texto_de_ficha(ficha):
    return " ".join([ficha["title"], ficha["text"], *ficha["tags"]])


def buscar_por_significado(consulta, top_k=3, mode=None):
    """Devuelve [(similitud, id, título)] ordenado de más a menos parecido.

    offline: mapa hecho a mano (vector de las etiquetas de cada ficha).
    live: embeddings reales de OpenAI sobre el texto completo de cada ficha.
    """
    mode = configure(mode)
    fichas = load_catalog()
    if mode == "live":
        from langchain_openai import OpenAIEmbeddings

        modelo = OpenAIEmbeddings(model=MODELO_EMBEDDINGS)
        consulta_vec = modelo.embed_query(consulta)
        fichas_vec = modelo.embed_documents([_texto_de_ficha(f) for f in fichas])
    else:
        consulta_vec = vector_de(consulta)
        if consulta_vec is None:
            return []
        fichas_vec = [vector_de(" ".join(f["tags"])) for f in fichas]
    puntajes = [
        (round(similitud(consulta_vec, vec), 3), f["id"], f["title"])
        for f, vec in zip(fichas, fichas_vec, strict=True)
        if vec is not None
    ]
    return sorted(puntajes, key=lambda p: (-p[0], p[1]))[:top_k]
