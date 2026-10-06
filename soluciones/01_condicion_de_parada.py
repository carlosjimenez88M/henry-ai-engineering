"""Solución · Clase 1 · La condición de parada del bucle."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from henry_agents.agentic import cerebro, en_espanol, linea_de_tiempo
from henry_agents.config import configure
from henry_agents.cultural import buscar_archivo
from henry_agents.practica import confirmar

REGLAS = "Usa buscar_archivo antes de responder. Cita los IDs entre corchetes."
AVISO_LIMITE = "Límite de pasos alcanzado: me detengo."


def mi_bucle(pedido, modelo, herramientas, max_pasos=4):
    por_nombre = {h.name: h for h in herramientas}
    mensajes = [SystemMessage(REGLAS), HumanMessage(pedido)]
    for paso in range(max_pasos):
        respuesta = modelo.invoke(mensajes)
        mensajes.append(respuesta)
        termino = not respuesta.tool_calls  # lista vacía = no pidió herramientas
        if termino:
            return mensajes
        for llamada in respuesta.tool_calls:
            try:
                resultado = por_nombre[llamada["name"]].invoke(llamada["args"])
                contenido, estado = json.dumps(resultado, ensure_ascii=False), "success"
            except Exception as error:  # el error vuelve al modelo como observación
                contenido, estado = f"Error: {en_espanol(str(error))}", "error"
            mensajes.append(
                ToolMessage(contenido, tool_call_id=llamada["id"], name=llamada["name"], status=estado)
            )
    mensajes.append(AIMessage(content=AVISO_LIMITE))
    return mensajes


MODE = configure()
modelo = cerebro(MODE).bind_tools([buscar_archivo])
mensajes = mi_bucle("Busca fichas de cooperación de El Chavo", modelo, [buscar_archivo])
linea_de_tiempo(mensajes[1:])
confirmar(mensajes[-1].text != AVISO_LIMITE, "El bucle debía terminar solo, sin agotar los pasos")
