"""Diagnóstico reproducible del taller RAG, con live únicamente al pedir --live.

Desde la raíz: uv run python scripts/check_rag_environment.py
Live:         uv run python scripts/check_rag_environment.py --live
El informe se guarda en reports/rag-intro-environment/verification.json.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
PAQUETES = {
    "qdrant-client": ("1.19", "2"),
    "numpy": ("2", "3"),
    "matplotlib": ("3.9", "4"),
    "ipywidgets": ("8", "9"),
    "ipykernel": ("6.29", "8"),
    "jupyterlab": ("4", "5"),
    "openai": ("1", None),
    "henry-modulo3": ("1", None),
}
MODULOS = {
    "curso": "henry_agents",
    "backend": "henry_agents.rag_taller",
    "replay": "henry_agents.rag_taller_replay",
    "panel": "henry_agents.rag_taller_panel",
}
PREGUNTA_LIVE = "¿Qué materiales debo llevar al taller de dibujo?"


def ruta_informe(root, report_path=None):
    if report_path is None:
        return root / "reports" / "rag-intro-environment" / "verification.json"
    destino = Path(report_path)
    return destino if destino.is_absolute() else root / destino


def _registrar(report, identificador, ok, mensaje, **detalles):
    report["checks"].append({
        "id": identificador, "ok": bool(ok), "mensaje": mensaje, "detalles": detalles,
    })


def _offline_e2e(taller, replay):
    cache = taller.leer_cache()
    respuestas = replay.leer_respuestas()
    modelo = "text-embedding-3-large"
    fragmentos = taller.fragmentar(taller.cargar_documentos())
    if len(fragmentos) != 14:
        raise ValueError("El corpus didáctico debe contener catorce fragmentos")
    vectores = taller.vectores_para([f["texto"] for f in fragmentos], modelo)
    pregunta = taller.CONSULTAS[3]
    with TemporaryDirectory(prefix="henry-rag-check-") as carpeta:
        with taller.abrir_base(Path(carpeta) / "qdrant") as db:
            taller.crear_indice(db, fragmentos, vectores, "verificacion", modelo)
            hallazgos = taller.buscar(db, "verificacion", pregunta, modelo, k=4)
            respuesta = replay.reproducir_respuesta(pregunta, hallazgos)
            cantidad = db.count("verificacion").count
    citas = {c.fragmento_id for c in respuesta.citas}
    if (cantidad != 14 or respuesta.estado != "respondido"
            or not {"CC-01-P1", "CC-01-P2"} <= citas
            or taller.validar_citas(respuesta, hallazgos)):
        raise ValueError("No se recuperaron y citaron plazo y requisitos")
    return {
        "fragmentos_indexados": cantidad,
        "modelo_embeddings": modelo,
        "dimensiones": cache["modelos"][modelo]["dimensiones"],
        "respuestas_preparadas": len(respuestas["entradas"]),
        "cobertura_multifuente": True,
        "solicitudes_api_actuales": 0,
    }


def _live_e2e(taller):
    from qdrant_client import models

    modelo = "text-embedding-3-large"
    fragmentos = taller.fragmentar(taller.cargar_documentos())
    vectores = taller.vectores_para([f["texto"] for f in fragmentos], modelo)
    with TemporaryDirectory(prefix="henry-rag-live-check-") as carpeta:
        with taller.abrir_base(Path(carpeta) / "qdrant") as db:
            taller.crear_indice(db, fragmentos, vectores, "verificacion_live", modelo)
            embedding = taller.solicitar_embeddings([PREGUNTA_LIVE], modelo)
            puntos = db.query_points(
                collection_name="verificacion_live",
                query=embedding["vectores"][0],
                query_filter=models.Filter(must=[models.FieldCondition(
                    key="vigente", match=models.MatchValue(value=True),
                )]),
                limit=4,
                with_payload=True,
            ).points
            hallazgos = [taller.Hallazgo(
                **{campo: p.payload[campo] for campo in (
                    "id", "titulo", "categoria", "vigente", "texto",
                )}, score=p.score,
            ) for p in puntos]
            respuesta = taller.responder(PREGUNTA_LIVE, hallazgos, mode="live")
    if (respuesta.estado != "respondido" or respuesta.llamadas_modelo != 1
            or "CC-04-P2" not in {c.fragmento_id for c in respuesta.citas}
            or taller.validar_citas(respuesta, hallazgos)):
        raise ValueError("La respuesta no se apoya en los materiales del taller")
    return {
        "modelo_embeddings": modelo,
        "tokens_embeddings": embedding["usage"],
        "tokens_generacion": respuesta.uso_tokens,
        "solicitudes_embeddings": 1,
        "solicitudes_generacion": 1,
        "cita_materiales_verificada": True,
    }


def verificar_entorno(root=ROOT, *, live=False, report_path=None):
    """Devuelve y guarda un informe seguro; no imprime excepciones ni valores de .env."""
    root = Path(root).resolve()
    report = {
        "version": 1,
        "modo": "live" if live else "offline",
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "checks": [],
        "versions": {},
        "env": {"archivo_existe": (root / ".env").is_file(), "clave_configurada": None},
        "metadata": {},
    }
    python_ok = (3, 11) <= tuple(sys.version_info[:2]) < (3, 14)
    _registrar(report, "python", python_ok,
               "Python compatible." if python_ok else "Usa Python 3.11, 3.12 o 3.13 y ejecuta uv sync.",
               version=".".join(str(n) for n in sys.version_info[:3]))
    entorno_ok = Path(sys.prefix).resolve() == (root / ".venv").resolve()
    _registrar(report, "venv", entorno_ok,
               "Se utiliza .venv del repositorio." if entorno_ok else
               "Abre la raíz del repositorio y ejecuta uv run python scripts/check_rag_environment.py.")
    for paquete, (minimo, maximo) in PAQUETES.items():
        try:
            version_texto = importlib.metadata.version(paquete)
            version = Version(version_texto)
            # Publicamos únicamente versiones PEP 440, nunca datos arbitrarios del entorno.
            report["versions"][paquete] = str(version)
            compatible = version >= Version(minimo) and (maximo is None or version < Version(maximo))
            _registrar(report, "paquete:" + paquete, compatible,
                       f"{paquete} disponible." if compatible else
                       f"Actualiza {paquete} ejecutando uv sync --locked.")
        except (importlib.metadata.PackageNotFoundError, InvalidVersion):
            _registrar(report, "paquete:" + paquete, False,
                       f"Falta una instalación válida de {paquete}. Ejecuta uv sync --locked.")
        except Exception:
            _registrar(report, "paquete:" + paquete, False,
                       f"No se pudo comprobar {paquete}. Ejecuta uv sync --locked.")
    modulos = {}
    for nombre, ruta in MODULOS.items():
        try:
            modulos[nombre] = importlib.import_module(ruta)
            _registrar(report, "import:" + nombre, True, f"Módulo {nombre} importado.")
        except Exception:
            _registrar(report, "import:" + nombre, False,
                       f"No se pudo importar {nombre}. Restaura el material y ejecuta uv sync --locked.")
    if "backend" in modulos and "replay" in modulos:
        try:
            report["metadata"]["offline"] = _offline_e2e(modulos["backend"], modulos["replay"])
            _registrar(report, "offline_e2e", True,
                       "Cachés reales, Qdrant, recuperación multifuente y citas funcionan sin API.")
        except Exception:
            _registrar(report, "offline_e2e", False,
                       "Falló el recorrido offline. Revisa los corpus/cachés del taller y restaura el material.")
    else:
        _registrar(report, "offline_e2e", False,
                   "No se puede probar RAG hasta restaurar los módulos backend y replay.")
    if live:
        config_ok = False
        try:
            config = importlib.import_module("henry_agents.config")
            config.configure("live")
            report["env"]["clave_configurada"] = bool(os.getenv("OPENAI_API_KEY"))
            nombre_modelo = config.model_name("rag")
            if nombre_modelo not in config.MODELOS:
                raise ValueError("Modelo no permitido")
            report["metadata"]["modelo_rag"] = nombre_modelo
            config_ok = report["env"]["clave_configurada"]
            _registrar(report, "env_live", config_ok,
                       "Configuración cargada y clave disponible; el valor permanece privado." if config_ok else
                       "Configura OPENAI_API_KEY en .env para ejecutar --live.")
        except Exception:
            report["env"]["clave_configurada"] = bool(os.getenv("OPENAI_API_KEY"))
            _registrar(report, "env_live", False,
                       "Revisa .env: OPENAI_API_KEY y OPENAI_MODEL_RAG deben estar configurados correctamente.")
        prerrequisitos_ok = all(check["ok"] for check in report["checks"])
        if config_ok and prerrequisitos_ok and "backend" in modulos:
            try:
                report["metadata"]["live"] = _live_e2e(modulos["backend"])
                _registrar(report, "live_e2e", True,
                           "Una pregunta nueva se embebe, consulta Qdrant y recibe respuesta con cita válida.")
            except Exception:
                _registrar(report, "live_e2e", False,
                           "Falló la comprobación live. Revisa conexión, acceso a modelos y configuración; "
                           "no se sustituyó la llamada por una respuesta offline.")
        else:
            _registrar(report, "live_e2e", False,
                       "La comprobación live requiere configuración válida y todas las comprobaciones offline aprobadas.")
    report["ok"] = all(check["ok"] for check in report["checks"])
    destino = ruta_informe(root, report_path)
    try:
        destino.parent.mkdir(parents=True, exist_ok=True)
        report["informe_guardado"] = True
        destino.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        report["informe_guardado"] = False
        report["ok"] = False
        _registrar(report, "informe", False,
                   "No se pudo guardar el informe. Revisa los permisos de la carpeta de destino.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Autoriza un embedding nuevo y una generación real")
    parser.add_argument("--report", type=Path, help="Ruta alternativa del informe JSON")
    args = parser.parse_args(argv)
    report = verificar_entorno(live=args.live, report_path=args.report)
    for check in report["checks"]:
        print(f"[{'OK' if check['ok'] else 'FALLO'}] {check['mensaje']}")
    if report["informe_guardado"]:
        print("Informe JSON guardado.")
    print("Entorno listo para ejecutar." if report["ok"] else "Corrige los puntos marcados y repite la comprobación.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
