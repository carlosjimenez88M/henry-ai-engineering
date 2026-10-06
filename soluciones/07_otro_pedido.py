"""Solución · Clase 7 · Tu turno 1: otro pedido para el equipo profundo."""

import re
from uuid import uuid4

from henry_agents.agentic import crear_equipo_profundo, ejecutar_con_revision
from henry_agents.config import configure
from henry_agents.practica import confirmar

MODE = configure()
mi_pedido = "Prepara una actividad de cooperación con fichas de El Chavo y una canción alegre."
estado, _ = ejecutar_con_revision(
    crear_equipo_profundo(MODE),
    {"messages": [("user", mi_pedido)]},
    {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 80},
    decidir=lambda accion: {"type": "approve"},
)
actividad = estado["files"]["/actividad.md"]["content"]
print(actividad)
ids = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", actividad))
print("Citó:", sorted(ids))
# Las palabras clave son el tema ("cooperación") y la colección ("Chavo").
confirmar({"CHA-01", "MUS-01"} <= ids, "Debía citar CHA-01 y MUS-01")
