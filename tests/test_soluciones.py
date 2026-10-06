"""Cada solución de la ruta avanzada se ejecuta sola y confirma su resultado."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLUCIONES = sorted((ROOT / "soluciones").glob("*.py"))


def test_hay_soluciones_para_los_ejercicios():
    assert SOLUCIONES, "Falta la carpeta soluciones/ con las soluciones de los ejercicios"


@pytest.mark.parametrize("archivo", SOLUCIONES, ids=lambda p: p.stem)
def test_solucion_se_ejecuta_y_confirma(archivo):
    entorno = {**os.environ, "COURSE_MODE": "offline", "HENRY_GRAPH_PNG": "0"}
    resultado = subprocess.run(
        [sys.executable, str(archivo)], cwd=ROOT, env=entorno, capture_output=True, text=True, timeout=120
    )
    assert resultado.returncode == 0, resultado.stdout[-1500:] + resultado.stderr[-1500:]
    assert "confirmar(" in archivo.read_text(encoding="utf-8"), "Cada solución debe confirmar su resultado"
