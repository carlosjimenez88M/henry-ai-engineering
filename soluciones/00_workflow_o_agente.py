"""Solución · Clase 0 · ¿Workflow o agente?

La pregunta clave en cada caso: ¿sé de antemano qué pasos vienen?
"""

from henry_agents.practica import confirmar

respuestas = {
    1: ("workflow", "Pasos fijos: leer → resumir → enviar. Un agente solo agregaría costo."),
    2: ("agente", "No se sabe qué buscar: cada resultado cambia el siguiente paso."),
    3: ("workflow", "Es routing: una clasificación y una regla. Un LLM puede clasificar sin bucle."),
    4: ("agente", "Hay que probar hipótesis y reaccionar a lo que se observa."),
    5: ("workflow", "Traducir y luego revisar con un criterio claro (evaluador, clase 6)."),
}
for numero, (tipo, razon) in respuestas.items():
    print(f"{numero}. {tipo:8} — {razon}")

confirmar([t for t, _ in respuestas.values()].count("agente") == 2, "Solo 2 y 4 son agentes")
