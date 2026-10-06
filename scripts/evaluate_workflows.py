"""Evalúa la referencia o un proyecto de estudiante con diez casos locales."""

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

from henry_agents.workflows import preparar_pedido

ROOT = Path(__file__).resolve().parents[1]


def evaluar(resolver=preparar_pedido):
    casos = json.loads(
        files("henry_agents").joinpath("data/workflows_casos.json").read_text("utf-8")
    )
    resultados = []
    for caso in casos:
        errores = []
        try:
            salida = resolver(caso["pedido"], caso["decision"])
        except ValueError:
            if caso["estado"] == "entrada_invalida":
                salida = {
                    "estado": "entrada_invalida", "cotizacion": {},
                    "mensaje": None, "accion_externa": False,
                }
            else:
                salida = {}
                errores.append("ValueError inesperado")
        except Exception as error:
            salida = {}
            errores.append(f"Excepción inesperada: {type(error).__name__}")
        if not isinstance(salida, dict):
            salida = {}
            errores.append("La solución no devolvió un diccionario")
        if salida.get("estado") != caso["estado"]:
            errores.append("Estado distinto del esperado")
        cotizacion = salida.get("cotizacion")
        cotizacion = cotizacion if isinstance(cotizacion, dict) else {}
        if "total_usd" in caso and cotizacion.get("total_usd") != caso["total_usd"]:
            errores.append("Total distinto del esperado")
        if salida.get("accion_externa") is not False:
            errores.append("El proyecto debe declarar accion_externa=False")
        if caso["estado"] != "aprobado" and salida.get("mensaje") is not None:
            errores.append("No debe entregar mensaje sin aprobación válida")
        if caso["estado"] == "aprobado":
            mensaje = salida.get("mensaje")
            fuentes = {f"CAT-{caso['pedido']['sku'].upper()}", "POL-ENTREGA"}
            if not isinstance(mensaje, str) or not all(f"[{f}]" in mensaje for f in fuentes):
                errores.append("El mensaje aprobado necesita referencias observadas")
            if not isinstance(mensaje, str) or f"USD {caso['total_usd']}" not in mensaje:
                errores.append("El mensaje debe mostrar el total observado")
        resultados.append({"id": caso["id"], "correcto": not errores, "errores": errores})
    return {
        "modo": "reglas_locales",
        "llamadas_api": 0 if resolver is preparar_pedido else None,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(casos),
        "correctos": sum(r["correcto"] for r in resultados),
        "resultados": resultados,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, help="Script propio con función resolver")
    parser.add_argument("--output", type=Path, default=Path("reports/workflows-evaluation.json"))
    args = parser.parse_args()
    resolver = preparar_pedido
    if args.implementation:
        ruta = args.implementation
        ruta = ruta if ruta.is_absolute() else ROOT / ruta
        spec = importlib.util.spec_from_file_location("proyecto_estudiante", ruta)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        resolver = modulo.resolver
    reporte = evaluar(resolver)
    salida = args.output if args.output.is_absolute() else ROOT / args.output
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{reporte['correctos']}/{reporte['n']} casos correctos. Reporte: {salida}")
    return 0 if reporte["correctos"] == reporte["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
