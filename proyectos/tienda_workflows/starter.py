"""Tu proyecto. Ejecuta desde la raíz: uv run python proyectos/tienda_workflows/starter.py.

Completa las dos fases descritas en README.md. Esta plantilla se puede abrir y
ejecutar, pero declara que el trabajo está pendiente; no es una solución aprobada.
"""


def construir_plan(pedido):
    # TODO fase 1: valida el pedido y devuelve los nombres de workers necesarios.
    return []


def resolver(pedido, decision="pendiente"):
    # TODO fase 2: plan → workers → cotización → revisión → decisión humana.
    # La interfaz debe devolver estado, cotizacion, mensaje y accion_externa.
    return {
        "estado": "pendiente_de_implementar",
        "cotizacion": {},
        "mensaje": None,
        "accion_externa": False,
    }


if __name__ == "__main__":
    pedido = {"sku": "arroz", "cantidad": 2, "barrio": "Centro"}
    print("Plantilla sin completar. Consulta README.md antes de mirar solution.py.")
    print(resolver(pedido))
