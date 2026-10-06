"""Herramientas de práctica para estudiantes: autocorrección amable y soluciones a pedido.

- comprobar(...) revisa TU ejercicio: muestra ✅ o una pista, nunca un traceback.
- confirmar(...) revisa una demostración: si falla, algo del entorno no está bien.
- ver_solucion("clave") imprime la solución de un ejercicio solo cuando la persona la pide.
"""

from henry_agents.config import ROOT

SOLUCIONES = ROOT / "soluciones"


def comprobar(condicion, bien, pista):
    """Revisa una condición y responde como lo haría un docente, sin traceback.

    >>> comprobar(2 + 2 == 4, "La suma es correcta.", "Revisa la suma.")
    ✅ La suma es correcta.
    """
    if condicion:
        print(f"✅ {bien}")
        return True
    print(f"🔁 Todavía no. Pista: {pista}")
    return False


class ErrorDelCurso(AssertionError):
    """Algo que el material da por cierto no se cumplió: avisar al equipo docente."""


def confirmar(condicion, mensaje):
    """Comprobación de una demostración (no de un ejercicio). Si falla, se detiene con un
    mensaje claro: significa que algo del entorno o del material no está como se esperaba."""
    if not condicion:
        raise ErrorDelCurso(f"❌ {mensaje}. Avisa al equipo docente con esta celda.")


def ver_solucion(clave):
    """Imprime la solución guardada en soluciones/<clave>.py. Inténtalo antes de mirarla."""
    archivo = SOLUCIONES / f"{clave}.py"
    if not archivo.exists():
        disponibles = sorted(p.stem for p in SOLUCIONES.glob("*.py"))
        print(f"No hay solución llamada '{clave}'. Disponibles: {disponibles}")
        return
    print(f"# Solución: {clave}\n")
    print(archivo.read_text(encoding="utf-8"))


def revisar(clave, respuesta):
    """Compara tu respuesta con la clave de soluciones/claves/<clave>.json, sin mostrarla.

    Para diccionarios dice qué ítems repensar; para listas o valores, si coincide o no.
    """
    import json

    esperada = json.loads((SOLUCIONES / "claves" / f"{clave}.json").read_text(encoding="utf-8"))
    if isinstance(esperada, dict):
        respuesta = {str(k): v for k, v in (respuesta or {}).items()}
        pendientes = [k for k, v in esperada.items() if respuesta.get(k) != v]
        if not pendientes:
            print(f"✅ Las {len(esperada)} respuestas coinciden. Lee la solución para comparar razones.")
            return True
        print(f"🔁 Repiensa: {', '.join(pendientes)} ({len(esperada) - len(pendientes)} de {len(esperada)} bien).")
        return False
    if respuesta == esperada:
        print("✅ Coincide con la clave.")
        return True
    print("🔁 Todavía no coincide. Relee la consigna y vuelve a intentar.")
    return False
