"""Compara RAG y Agentic RAG con el mismo índice, k y doce preguntas ficticias."""

import argparse
import json
import time
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

from henry_agents.config import configure, model_name
from henry_agents.rag import IndiceLexico, crear_agente_rag, estado_inicial, rag_clasico

ROOT = Path(__file__).resolve().parents[1]


def evaluar(mode="offline", case_id=None):
    mode = configure(mode)
    casos = json.loads(files("henry_agents").joinpath("data/rag_casos.json").read_text("utf-8"))
    if case_id:
        casos = [c for c in casos if c["id"] == case_id]
        if not casos:
            raise ValueError("No existe ese caso en rag_casos.json")
    indice = IndiceLexico()
    app = crear_agente_rag(mode=mode, indice=indice, k=1)
    filas = []
    for caso in casos:
        esperados = set(caso["documentos"])
        fila = {"id": caso["id"], "pregunta": caso["pregunta"]}
        for tipo in ("clasico", "agentico"):
            inicio = time.perf_counter()
            result = (rag_clasico(caso["pregunta"], indice=indice, k=1, mode=mode)
                      if tipo == "clasico" else app.invoke(estado_inicial(caso["pregunta"])))
            retrieved = {f["doc_id"] for f in result["evidencia"]}
            cited = {s.rsplit("-c", 1)[0] if "-c" in s else s for s in result["fuentes"]}
            citas_validas = cited <= retrieved
            citas_completas = (esperados <= cited) if result["estado"] == "respondido" else not cited
            fila[tipo] = {
                "estado": result["estado"], "causa": result["causa"],
                "comportamiento_esperado": result["estado"] == caso[tipo],
                "documentos_recuperados": sorted(retrieved), "documentos_citados": sorted(cited),
                "recall_documentos": len(esperados & retrieved) / len(esperados) if esperados else None,
                "precision_documentos": len(esperados & retrieved) / len(retrieved) if retrieved else None,
                "citas_validas": citas_validas, "citas_completas": citas_completas,
                "busquedas": result["busquedas"], "llamadas_llm": result["llamadas_llm"],
                "latencia_ms": round((time.perf_counter() - inicio) * 1000, 2),
                "respuesta": result["respuesta"],
                "eventos": result.get("eventos", []),
            }
        filas.append(fila)
    metricas = {}
    for tipo in ("clasico", "agentico"):
        positivos = [f[tipo]["recall_documentos"] for f in filas
                     if f[tipo]["recall_documentos"] is not None]
        metricas[tipo] = {
            "comportamientos_correctos": sum(f[tipo]["comportamiento_esperado"] for f in filas),
            "respondidas": sum(f[tipo]["estado"] == "respondido" for f in filas),
            "recall_documentos_promedio": sum(positivos) / len(positivos) if positivos else None,
            "busquedas_totales": sum(f[tipo]["busquedas"] for f in filas),
            "llamadas_llm": sum(f[tipo]["llamadas_llm"] for f in filas),
        }
    passed = all(f[t]["comportamiento_esperado"] and f[t]["citas_validas"]
                 and f[t]["citas_completas"] for f in filas for t in ("clasico", "agentico"))
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "status": "passed" if passed else "failed", "n": len(casos), "k": 1,
        "recuperador": "BM25 local; mismos documentos vigentes en ambos sistemas",
        "modelos_configurados": {r: model_name(r) for r in ("rag", "agent", "embeddings")},
        "metricas": metricas, "resultados": filas,
        "limite": "Criterios y casos didácticos; un juez LLM y las citas no prueban verdad universal.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--case", help="Ejecutar un solo caso, por ejemplo RAG-06")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reporte = evaluar(args.mode, args.case)
    salida = args.output or ROOT / "reports" / f"rag-evaluation-{args.mode}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reporte["metricas"], ensure_ascii=False, indent=2))
    print("Estado:", reporte["status"], "| Reporte:", salida)
    return 0 if reporte["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
