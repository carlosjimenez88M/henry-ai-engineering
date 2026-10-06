"""Solución · Clase 0 · El costo de un curso."""

from henry_agents.config import costo_usd
from henry_agents.practica import confirmar

llamadas_totales = 30 * 20 * 5  # personas × consultas × llamadas por consulta

for nombre in ["gpt-6-luna", "gpt-6-astra"]:
    print(f"{nombre:12} ≈ ${llamadas_totales * costo_usd(nombre, 3000, 500):.2f}")

confirmar(llamadas_totales == 3000, "Debían ser 3.000 llamadas")
