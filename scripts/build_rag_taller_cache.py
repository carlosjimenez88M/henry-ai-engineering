#!/usr/bin/env python3
"""Genera una vez la caché REAL para el taller; requiere autorización explícita --live."""

import argparse
import json
from datetime import datetime, timezone

from henry_agents.rag_taller import (
    CACHE_PATH,
    FRASES_COMPARACION,
    MODELOS_EMBEDDING,
    PREGUNTAS,
    cargar_documentos,
    fragmentar,
    sha_corpus,
    sha_texto,
    solicitar_embeddings,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Autoriza dos batches de OpenAI")
    args = parser.parse_args()
    if not args.live:
        parser.error("Se requiere --live: esta operación llama a la API de embeddings")
    documentos = cargar_documentos()
    textos = list(dict.fromkeys([
        *(f["texto"] for f in fragmentar(documentos)), *PREGUNTAS, *FRASES_COMPARACION,
    ]))
    cache = {
        "version": 1,
        "descripcion": "Embeddings reales de un corpus didáctico original y ficticio",
        "corpus_sha256": sha_corpus(documentos),
        "preguntas": PREGUNTAS,
        "frases_comparacion": FRASES_COMPARACION,
        "modelos": {},
    }
    for modelo, dimensiones in MODELOS_EMBEDDING.items():
        resultado = solicitar_embeddings(textos, modelo)
        cache["modelos"][modelo] = {
            "modelo": modelo,
            "dimensiones": dimensiones,
            "origen": "openai_embeddings_api",
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "usage": resultado["usage"],
            "cantidad_textos": len(textos),
            "entradas": {
                sha_texto(texto): {"texto": texto, "vector": vector}
                for texto, vector in zip(textos, resultado["vectores"], strict=True)
            },
        }
        print(f"{modelo}: {len(textos)} textos, {dimensiones} dimensiones; "
              f"tokens reportados: {resultado['usage']['total_tokens']}")
    # Solo reemplaza la caché cuando ambos batches terminaron y se validaron.
    temporal = CACHE_PATH.with_suffix(".json.tmp")
    temporal.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporal.replace(CACHE_PATH)
    print(f"Caché guardada: {CACHE_PATH.name}")


if __name__ == "__main__":
    main()
