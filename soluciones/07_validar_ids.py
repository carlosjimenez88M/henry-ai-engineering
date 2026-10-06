"""Solución · Clase 7 · Tu turno 2: una herramienta que verifica citas."""

import re

from langchain_core.tools import tool

from henry_agents.cultural import load_catalog
from henry_agents.practica import confirmar

ids_catalogo = {ficha["id"] for ficha in load_catalog()}


@tool
def validar_ids(texto: str) -> dict:
    """Revisa que los IDs citados (formato ABC-01) existan en el catálogo del curso."""
    citados = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", texto))
    inventados = citados - ids_catalogo  # los citados que no están en el catálogo
    valido = bool(citados) and not inventados  # hay citas y ninguna es inventada
    return {"citados": sorted(citados), "inventados": sorted(inventados), "valido": valido}


prueba = validar_ids.invoke({"texto": "Según [BAT-01] y [ZZZ-01], el reloj estaba adelantado."})
print(prueba)
print(validar_ids.invoke({"texto": "Según [BAT-01], el reloj estaba adelantado."}))
confirmar(prueba == {"citados": ["BAT-01", "ZZZ-01"], "inventados": ["ZZZ-01"], "valido": False},
          "La herramienta debía marcar ZZZ-01 como inventado")
