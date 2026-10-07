"""Frases de cómics y sus embeddings para el bloque visual del taller Python.

El notebook hace las cuentas y los gráficos; este módulo solo entrega vectores:
- offline: los lee de un cache JSON generado una vez con el modelo real (gratis y reproducible).
- live: los pide a OpenAI con el modelo de embeddings configurado en .env.

Regenerar el cache (requiere OPENAI_API_KEY):
    uv run python -m henry_agents.embeddings_comics
"""

import json
from datetime import date
from pathlib import Path

from henry_agents.config import configure, model_name

CACHE = Path(__file__).parent / "data" / "embeddings_comics.json"

# Tres temas cruzados con tres héroes, más frases sin héroe y fuera del tema.
# Así se puede preguntar: ¿los vectores se agrupan por héroe o por significado?
FRASES = [
    {"texto": "Batman sigue huellas en un callejón para descubrir quién apagó las luces.",
     "heroe": "batman", "tema": "investigar"},
    {"texto": "Batman organiza a los vecinos para repartir agua después de la tormenta.",
     "heroe": "batman", "tema": "comunidad"},
    {"texto": "Batman calibra un sensor nuevo en su laboratorio subterráneo.",
     "heroe": "batman", "tema": "ciencia"},
    {"texto": "Spider-Man interroga a los testigos para resolver el robo de la biblioteca.",
     "heroe": "spiderman", "tema": "investigar"},
    {"texto": "Spider-Man ayuda a una vecina a subir las compras por la escalera.",
     "heroe": "spiderman", "tema": "comunidad"},
    {"texto": "Spider-Man diseña una telaraña más resistente en el laboratorio de la escuela.",
     "heroe": "spiderman", "tema": "ciencia"},
    {"texto": "Los Cuatro Fantásticos comparan pistas para encontrar un mapa perdido.",
     "heroe": "fantasticos", "tema": "investigar"},
    {"texto": "Los Cuatro Fantásticos montan un comedor comunitario en el barrio.",
     "heroe": "fantasticos", "tema": "comunidad"},
    {"texto": "Los Cuatro Fantásticos prueban un motor experimental antes de despegar.",
     "heroe": "fantasticos", "tema": "ciencia"},
    {"texto": "Una detective revisa las cámaras de seguridad para encontrar al culpable.",
     "heroe": "ninguno", "tema": "investigar"},
    {"texto": "El club del barrio reparte mantas durante una noche muy fría.",
     "heroe": "ninguno", "tema": "comunidad"},
    {"texto": "Una ingeniera ajusta un robot que mide la contaminación del aire.",
     "heroe": "ninguno", "tema": "ciencia"},
    {"texto": "La receta lleva tomate, ajo y aceite de oliva.",
     "heroe": "ninguno", "tema": "otro"},
    {"texto": "El partido de fútbol terminó empatado en el último minuto.",
     "heroe": "ninguno", "tema": "otro"},
]

# Consultas ya calculadas: permiten practicar la búsqueda sin clave.
CONSULTAS = [
    "resolver un misterio siguiendo pistas",
    "cuidar a la gente del barrio",
    "inventar un aparato con tecnología",
]


def _pedir_a_openai(textos):
    from langchain_openai import OpenAIEmbeddings

    modelo = OpenAIEmbeddings(model=model_name("embeddings"), max_retries=0)
    return modelo.embed_documents(list(textos))


def leer_cache():
    """Devuelve el cache completo: modelo, fecha, dimensiones y vectores por texto."""
    return json.loads(CACHE.read_text(encoding="utf-8"))


def obtener_embeddings(textos, mode=None):
    """Una lista de vectores, uno por texto y en el mismo orden."""
    if configure(mode) == "live":
        return _pedir_a_openai(textos)
    vectores = leer_cache()["vectores"]
    faltantes = [texto for texto in textos if texto not in vectores]
    if faltantes:
        raise ValueError(
            f"Sin vector offline para {faltantes}. Usa una frase de FRASES o CONSULTAS, "
            "o activa el modo live para calcular frases nuevas."
        )
    return [vectores[texto] for texto in textos]


def regenerar_cache():
    """Calcula con el modelo real todas las frases y consultas y guarda el JSON."""
    textos = [frase["texto"] for frase in FRASES] + CONSULTAS
    vectores = _pedir_a_openai(textos)
    cache = {
        "modelo": model_name("embeddings"),
        "generado_el": date.today().isoformat(),
        "dimensiones": len(vectores[0]),
        # Cinco decimales bastan para la similitud y mantienen el archivo pequeño.
        "vectores": {
            texto: [round(valor, 5) for valor in vector]
            for texto, vector in zip(textos, vectores, strict=True)
        },
    }
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


if __name__ == "__main__":
    configure("live")
    resultado = regenerar_cache()
    print(f"{len(resultado['vectores'])} vectores de {resultado['dimensiones']} dimensiones "
          f"con {resultado['modelo']} → {CACHE}")
