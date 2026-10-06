"""Solución explicada: composición visible de herramientas ya estudiadas.

No hay LLM, envío de mensajes, reserva de stock ni cobros. El plan sigue reglas.
"""

from henry_agents.workflows import (
    cotizar,
    ejecutar_plan,
    planificar,
    redactar_cotizacion,
    revisar_hasta_limite,
    validar_pedido,
)


def construir_plan(pedido):
    # La regla de la clase 06 evita consultar entrega para un retiro.
    return planificar(validar_pedido(pedido))


def resolver(pedido, decision="pendiente"):
    if decision not in {"pendiente", "aprobar", "rechazar"}:
        raise ValueError("Decisión desconocida")
    pedido = validar_pedido(pedido)
    plan = construir_plan(pedido)
    observaciones = ejecutar_plan(pedido, plan)
    cotizacion = cotizar(pedido)
    salida = {
        "estado": cotizacion["estado"],
        "plan": plan,
        "resultados": observaciones,
        "cotizacion": cotizacion,
        "mensaje": None,
        "accion_externa": False,
    }
    # Detenerse antes de crear un borrador si no hay evidencia, stock o cobertura.
    if cotizacion["estado"] != "cotizado":
        return salida

    def generar(feedback):
        # Una plantilla correcta desde el primer intento. En la clase 05 probamos
        # por separado cómo se corrige una mala y cómo se detiene una atascada.
        return {
            "total_usd": cotizacion["total_usd"],
            "fuentes": list(cotizacion["fuentes"]),
            "requiere_aprobacion": True,
        }

    revision = revisar_hasta_limite(generar, cotizacion, max_intentos=3)
    salida["revision"] = revision
    if revision["estado"] != "validado":
        salida["estado"] = revision["estado"]
        return salida
    # La persona decide después de la validación automática, no en lugar de ella.
    salida["estado"] = {"pendiente": "pendiente", "aprobar": "aprobado", "rechazar": "rechazado"}[
        decision
    ]
    if decision == "aprobar":
        salida["mensaje"] = redactar_cotizacion(cotizacion)
    return salida


if __name__ == "__main__":
    pedido = {"sku": "arroz", "cantidad": 2, "barrio": "Centro"}
    for decision in ("pendiente", "aprobar", "rechazar"):
        print(decision, resolver(pedido, decision))
