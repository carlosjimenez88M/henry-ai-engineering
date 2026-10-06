"""Solución · Clase 4 · Tu turno 2: cada falla con su defensa principal."""

from henry_agents.practica import confirmar

# A contrato estricto · B validar citas · C límite de llamadas · D exigir evidencia de herramienta
mis_defensas = {
    # Respondió sin 🔧: no aceptamos respuestas sin una observación de la herramienta.
    "no_usa_herramienta": "D",
    # Citó [BAT-99]: comparamos las citas con los IDs reales del catálogo.
    "inventa_id": "B",
    # Pidió top_k=50: SearchArgs lo rechazó y el error volvió al modelo para corregirse.
    "argumentos_invalidos": "A",
    # Repetía la búsqueda: LimiteDeLlamadas cortó tras N llamadas al modelo.
    "bucle": "C",
}
for falla, defensa in mis_defensas.items():
    print(f"{falla:22} → {defensa}")
confirmar(sorted(mis_defensas.values()) == ["A", "B", "C", "D"], "Cada defensa se usa una vez")
