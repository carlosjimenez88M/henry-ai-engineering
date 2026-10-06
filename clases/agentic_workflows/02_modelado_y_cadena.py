# %% [markdown]
# # 02 · Dibujar antes de programar: modelado y cadena
#
# **Meta:** identificar entradas, salidas y condiciones de parada de una secuencia.
# **Producto:** pedido → validar → cotizar → redactar, con abstención.
# Una cadena (*prompt chaining* cuando sus pasos usan prompts) pasa el resultado
# de un paso al siguiente. Nuestra primera versión usa Python y plantillas.
#
# ```mermaid
# flowchart LR
#   P[Pedido] --> V[Validar entrada] --> C[Cotizar con datos]
#   C --> D{Hay evidencia y cobertura?}
#   D -->|Sí| R[Redactar borrador]
#   D -->|No| H[Revisión de una persona]
# ```
#
# El diagrama está en el notebook; el código no descarga imágenes ni usa un
# servicio de dibujo. Si tu editor no renderiza Mermaid, lee las flechas como texto.

# %%
from henry_agents.workflows import cotizar, redactar_cotizacion, validar_pedido

pedido = {"sku": "arroz", "cantidad": 2, "barrio": "Centro"}

# %% [markdown]
# ## 1. Escribe el contrato de cada paso
# **Predice:** ¿qué valor necesita el redactor para escribir el total?
# Esperado: la cotización observada; no una suposición del prompt original.

# %%
entrada_validada = validar_pedido(pedido)
observacion = cotizar(entrada_validada)
borrador = redactar_cotizacion(observacion)
print("Entrada:", entrada_validada)
print("Observación:", observacion)
print("Borrador:", borrador)
assert "USD 5.60" in borrador

# %% [markdown]
# ## Pausa y comprobación
# Señala un nodo y una flecha. Un nodo hace trabajo; una flecha expresa dependencia.
# Si el total cambia, ¿qué dos pasos debes volver a ejecutar? Cotizar y redactar.
#
# ## 2. Construye la cadena visible
# No necesitamos una clase por cada función ni un framework para comprender el patrón.

# %%
def cadena(pedido):
    entrada = validar_pedido(pedido)
    observacion = cotizar(entrada)
    return {"observacion": observacion, "borrador": redactar_cotizacion(observacion)}


salida = cadena(pedido)
assert salida["observacion"]["estado"] == "cotizado"
print(salida)

# %% [markdown]
# ## 3. Error útil: cambiar el barrio
# **Predice:** si el pedido va a Sur, ¿el redactor debería prometer una entrega?

# %%
sin_cobertura = cadena({**pedido, "barrio": "Sur"})
print(sin_cobertura)
assert sin_cobertura["observacion"]["estado"] == "fuera_de_cobertura"
assert "USD" not in sin_cobertura["borrador"]

# %% [markdown]
# ## Segunda pausa · Taller
# Diseña `cadena_con_control(pedido)` que devuelva `estado="pendiente"` y el
# borrador cuando la cotización es válida; si falla, conserva el estado observado
# y devuelve `borrador=None`. Dibuja dónde se toma esa decisión.
# **Pista:** compara `observacion["estado"] == "cotizado"` antes de redactar.
#
# ## Solución comentada

# %%
def cadena_con_control(pedido):
    observacion = cotizar(validar_pedido(pedido))
    if observacion["estado"] != "cotizado":
        return {"estado": observacion["estado"], "borrador": None}
    return {"estado": "pendiente", "borrador": redactar_cotizacion(observacion)}


assert cadena_con_control(pedido)["estado"] == "pendiente"
assert cadena_con_control({**pedido, "sku": "leche"})["borrador"] is None

# %% [markdown]
# ## Ticket de salida
# Explica por qué esta cadena no es un agente autónomo. Propón una entrada que la
# haga detenerse. Si reemplazas el redactor por un LLM, ¿qué verificarías después?
# Mantendrías precio/fuentes y revisarías fidelidad del texto.
# Siguiente: [03 · Routing](03_routing.ipynb).
