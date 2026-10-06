"""Herramientas de práctica para estudiantes: autocorrección amable y soluciones a pedido.

- comprobar(...) muestra ✅ o una pista, en lugar de un AssertionError con traceback.
- ver_solucion("clave") imprime la solución de un ejercicio solo cuando la persona la pide.
"""

import os

from henry_agents.config import ROOT

SOLUCIONES = ROOT / "soluciones"


def _estricto():
    # La verificación automática del curso exige que todo pase; en clase se muestran pistas.
    return os.getenv("HENRY_ESTRICTO") == "1"


def comprobar(condicion, bien, pista):
    """Revisa una condición y responde como lo haría un docente, sin traceback.

    >>> comprobar(2 + 2 == 4, "La suma es correcta.", "Revisá la suma.")
    ✅ La suma es correcta.
    """
    if condicion:
        print(f"✅ {bien}")
        return True
    if _estricto():
        raise AssertionError(pista)
    print(f"🔁 Todavía no. Pista: {pista}")
    return False


def ver_solucion(clave):
    """Imprime la solución guardada en soluciones/<clave>.py. Intentá antes de mirarla."""
    archivo = SOLUCIONES / f"{clave}.py"
    if not archivo.exists():
        disponibles = sorted(p.stem for p in SOLUCIONES.glob("*.py"))
        print(f"No hay solución llamada '{clave}'. Disponibles: {disponibles}")
        return
    print(f"# Solución: {clave}\n")
    print(archivo.read_text(encoding="utf-8"))
