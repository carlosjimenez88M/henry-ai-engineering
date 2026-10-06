# %% [markdown]
# # 07 · Proyecto integrador: una propuesta revisable
#
# **Meta:** unir herramientas, plan, revisión y decisión de una persona.
# **Producto:** preparar una cotización y un reporte de casos; no enviar ni cobrar.
# El flujo comienza con un pedido estructurado. La clasificación de mensajes de
# la clase 03 es una etapa previa posible; este proyecto no interpreta lenguaje libre.
#
# Trabaja en `proyectos/tienda_workflows/starter.py`. Las instrucciones, el catálogo,
# la rúbrica y una solución separada están en esa carpeta. Primero construyes
# funciones pequeñas (fase 1); después las conectas (fase 2).

# %%
import json
from importlib.resources import files

from henry_agents.workflows import preparar_pedido

pedido = {"sku": "arroz", "cantidad": 2, "barrio": "Centro"}

# %% [markdown]
# ## 1. Ejecuta hasta una propuesta pendiente
# **Predice:** ¿qué mensaje queda disponible antes de decidir?
# La decisión aquí es una variable; no hay persistencia ni un sistema real de permisos.

# %%
pendiente = preparar_pedido(pedido)
print(pendiente)
assert pendiente["estado"] == "pendiente"
assert pendiente["mensaje"] is None
assert pendiente["accion_externa"] is False

# %% [markdown]
# ## Pausa y revisión humana
# Lee el total, el catálogo y la política. Haz de cuenta que eres quien atiende:
# explica qué aceptarías y qué cambiarías. No apruebes solo porque "suena bien".
#
# ## 2. Compara aprobación y rechazo
# Las dos llamadas recalculan la propuesta. No son reanudaciones de un checkpoint.
# Las clases posteriores de LangGraph muestran interrupciones y reanudaciones reales.

# %%
aprobado = preparar_pedido(pedido, decision="aprobar")
rechazado = preparar_pedido(pedido, decision="rechazar")
assert aprobado["estado"] == "aprobado" and "USD 5.60" in aprobado["mensaje"]
assert rechazado["estado"] == "rechazado" and rechazado["mensaje"] is None
assert not aprobado["accion_externa"] and not rechazado["accion_externa"]
print("Aprobado:", aprobado["mensaje"])
print("Rechazado:", rechazado["estado"])

# %% [markdown]
# ## 3. Error útil: una aprobación no crea stock ni cobertura
# **Predice:** ¿qué devuelve un pedido de cuaderno aunque la decisión sea aprobar?

# %%
sin_stock = preparar_pedido({**pedido, "sku": "cuaderno"}, decision="aprobar")
assert sin_stock["estado"] == "sin_stock"
assert sin_stock["mensaje"] is None
print(sin_stock)

# %% [markdown]
# ## Segunda pausa · Taller de evaluación
# Corre diez casos incluidos. Cuenta aprobados contra expectativas, no contra tu
# impresión. Después añade dos casos tuyos: uno válido y uno de borde.
# **Pista:** una entrada inválida debe contarse como comportamiento esperado,
# sin ocultar otras excepciones como un error de programación.
#
# ## Solución comentada · Reporte en memoria

# %%
casos = json.loads(files("henry_agents").joinpath("data/workflows_casos.json").read_text("utf-8"))
resultados = []
for caso in casos:
    try:
        salida = preparar_pedido(caso["pedido"], caso["decision"])
        estado = salida["estado"]
        total = salida["cotizacion"].get("total_usd")
    except ValueError:
        estado, total = "entrada_invalida", None
    correcto = estado == caso["estado"]
    if "total_usd" in caso:
        correcto = correcto and total == caso["total_usd"]
    resultados.append({"id": caso["id"], "estado": estado, "correcto": correcto})
print(json.dumps(resultados, indent=2, ensure_ascii=False))
assert all(r["correcto"] for r in resultados)
print("Casos correctos:", sum(r["correcto"] for r in resultados), "de", len(resultados))

# %% [markdown]
# ## Entrega y ticket de salida
# Entrega tu script, un diagrama, el reporte y una explicación de una falla.
# Para guardar el reporte de la solución de referencia desde la raíz del repo:
# `uv run python scripts/evaluate_workflows.py`.
#
# Defiende por qué elegiste ese flujo. Reconoce un límite: reglas no comprenden
# lenguaje general; validar campos no prueba un texto; aprobar no equivale a enviar.
# Los diez casos solo cubren estos comportamientos con estos datos ficticios.
#
# Para continuar con agentes reales y grafos, vuelve al README y sigue el recorrido
# existente: `clases/00_mundo_agentico.ipynb` a `clases/05_deep_agents.ipynb`.
