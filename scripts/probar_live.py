"""Prueba de humo del modo live de la ruta avanzada (consume centavos de API).

Comprueba lo que usan las clases: texto, herramientas, salida estructurada, embeddings y
un agente con límites. Uso: uv run python scripts/probar_live.py
"""

from langchain_core.messages import HumanMessage

from henry_agents.agentic import crear_agente, linea_de_tiempo
from henry_agents.config import chat_model, configure, medir_costo, model_name
from henry_agents.cultural import GroundedAnswer, buscar_archivo, compose, search_catalog
from henry_agents.semantica import buscar_por_significado


def paso(nombre, funcion):
    try:
        print(f"✅ {nombre}: {funcion()}")
        return True
    except Exception as error:  # Mostrar el tipo ayuda a diagnosticar sin exponer la clave.
        print(f"❌ {nombre}: {type(error).__name__}: {str(error)[:300]}")
        return False


def main():
    configure("live")
    print("Modelos:", model_name(), "/", model_name("agent"), "/", model_name("embeddings"))
    with medir_costo():
        resultados = [
            paso("texto", lambda: chat_model().invoke("Responde solo: listo").text),
            paso(
                "herramientas",
                lambda: chat_model()
                .bind_tools([buscar_archivo])
                .invoke([HumanMessage("Busca fichas de investigación de Batman")])
                .tool_calls,
            ),
            paso(
                "salida estructurada",
                lambda: chat_model()
                .with_structured_output(GroundedAnswer)
                .invoke("Resume y cita: [BAT-01] Batman ordena pistas."),
            ),
            paso("compose", lambda: compose(search_catalog("investigación", "batman"), "live")),
            paso("embeddings", lambda: buscar_por_significado("un enigma", mode="live")),
            paso(
                "agente",
                lambda: linea_de_tiempo(
                    crear_agente("live").invoke({"messages": [HumanMessage("investigación de Batman")]})[
                        "messages"
                    ]
                ),
            ),
        ]
    print(f"\n{sum(resultados)} de {len(resultados)} pruebas live aprobadas.")
    return 0 if all(resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
