"""Prepara cinco respuestas reales para el mismo contexto usado en la clase.

uv run python scripts/build_rag_taller_respuestas.py --live
No se ejecuta desde el notebook ni desde la verificación offline.
"""

import argparse
import json
from datetime import datetime, timezone
from tempfile import TemporaryDirectory

from henry_agents.config import configure
from henry_agents.rag_taller import (
    CONSULTAS,
    abrir_base,
    buscar,
    cargar_documentos,
    crear_indice,
    fragmentar,
    responder,
    sha_corpus,
    vectores_para,
)
from henry_agents.rag_taller_replay import RESPUESTAS_PATH, firma_contexto


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.parse_args()
    configure("live")
    fragmentos = fragmentar(cargar_documentos())
    modelo = "text-embedding-3-large"
    vectores = vectores_para([f["texto"] for f in fragmentos], modelo, mode="offline")
    datos = {
        "version": 1, "origen": "openai_chat_api",
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": sha_corpus(), "embedding_modelo": modelo,
        "k": 4, "vigentes": True, "entradas": {},
    }
    with TemporaryDirectory(prefix="rag-respuestas-") as carpeta:
        with abrir_base(carpeta) as bd:
            crear_indice(bd, fragmentos, vectores, "centro", modelo)
            for numero, pregunta in enumerate(CONSULTAS, start=1):
                hallazgos = buscar(bd, "centro", pregunta, modelo, k=4, mode="offline")
                resultado = responder(pregunta, hallazgos, mode="live")
                datos["entradas"][firma_contexto(pregunta, hallazgos)] = {
                    "pregunta": pregunta,
                    "contexto": [h.model_dump() for h in hallazgos],
                    "resultado": resultado.model_dump(),
                }
                print(f"Caso {numero}: {resultado.estado}, {len(resultado.citas)} citas, modelo {resultado.modelo_usado}", flush=True)
    temporal = RESPUESTAS_PATH.with_suffix(".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporal.replace(RESPUESTAS_PATH)
    print("Se guardaron cinco respuestas; sus tokens pertenecen a esta preparación.")


if __name__ == "__main__":
    main()
