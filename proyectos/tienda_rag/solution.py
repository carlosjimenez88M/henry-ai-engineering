"""Comparación controlada: mismos datos/k; el agente puede consultar más veces."""

from henry_agents.rag import IndiceLexico, crear_agente_rag, estado_inicial, rag_clasico


def comparar(pregunta):
    indice = IndiceLexico()
    clasico = rag_clasico(pregunta, indice=indice, k=1)
    agente = crear_agente_rag(indice=indice, k=1, max_busquedas=3, max_decisiones=4)
    agentico = agente.invoke(estado_inicial(pregunta))
    return {"clasico": clasico, "agentico": agentico}


if __name__ == "__main__":
    resultado = comparar("¿Hacen mandados?")
    for nombre, salida in resultado.items():
        print(nombre, salida["estado"], "consultas:", salida["busquedas"])
        print(salida["respuesta"])
