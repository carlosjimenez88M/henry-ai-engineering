"""Solución · Clase 7 · Tu turno 1: otro pedido para el equipo profundo."""

import re
from uuid import uuid4

from langchain_core.messages import HumanMessage

from henry_agents.agentic import crear_equipo_profundo, ejecutar_con_revision
from henry_agents.config import configure
from henry_agents.cultural import load_catalog, search_catalog
from henry_agents.practica import confirmar

MODE = configure()
# Antes de ejecutar, averiguamos qué esperar con la herramienta de la clase 1.
fichas = [h.id for h in search_catalog("ciencia", universe="fantasticos", top_k=2).hits]
cancion = [h.id for h in search_catalog("ciencia", universe="canciones", kind="cancion", top_k=1).hits]
mi_prediccion = set(fichas + cancion)
print("Predicción:", sorted(mi_prediccion))

mi_pedido = "Prepara una actividad de ciencia con fichas de los Fantásticos y una canción."
estado, _ = ejecutar_con_revision(
    crear_equipo_profundo(MODE),
    {"messages": [HumanMessage(mi_pedido)]},
    {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 80},
    decidir=lambda accion: {"type": "approve"},
)
actividad = estado.get("files", {}).get("/actividad.md", {}).get("content", "")
print(actividad)
ids = set(re.findall(r"\b[A-Z]{3}-\d{2}\b", actividad))
print("Citó:", sorted(ids))
# Las palabras clave son el tema ("ciencia") y la colección ("Fantásticos").
if MODE == "offline":
    confirmar(ids == mi_prediccion == {"FAN-01", "FAN-02", "MUS-03"}, "Debía citar FAN-01, FAN-02 y MUS-03")
else:
    confirmar(ids <= {f["id"] for f in load_catalog()}, "Toda cita debía existir en el catálogo")
