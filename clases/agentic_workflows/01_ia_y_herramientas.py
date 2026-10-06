# %% [markdown]
# # 01 · IA, herramientas y límites
#
# **Meta:** distinguir un modelo de lenguaje, una herramienta y un workflow.
# **Necesitas:** la clase 00 o poder leer una función y un diccionario.
# **Producto:** una cotización basada en un catálogo, con error de entrada visible.
#
# - **LLM:** modelo que genera lenguaje a partir del contexto recibido. Puede equivocarse.
# - **Prompt:** instrucción y datos que enviamos al modelo.
# - **Token:** unidad de texto que el modelo procesa; no siempre equivale a una palabra.
# - **Herramienta:** función que nuestro programa ejecuta, con entradas permitidas.
# - **Workflow:** pasos conectados; el código fija su orden y sus condiciones.
# - **Agente con LLM:** el modelo elige la siguiente acción en un ciclo con límites.
#
# Aquí usamos **reglas y plantillas**, sin modelo, sin entrenamiento y sin API.
# No llamaremos "IA" a una función solo porque tenga un nombre como `agente`.
# En la clase 03 hay un experimento opcional con un modelo real.

# %%
from decimal import Decimal

from henry_agents.workflows import consultar_producto, cotizar, validar_pedido

print("Modo de esta clase: reglas locales. Llamadas al modelo: 0.")

# %% [markdown]
# ## 1. La herramienta observa; el modelo no debe inventar el precio
# Imagina que un chat propone "2 arroces cuestan USD 1". La frase es fluida, pero
# el precio se comprueba contra datos. En nuestro ejemplo no hay una base de datos externa.
# **Predice:** ¿cuál es el subtotal de 2 bolsas de arroz a USD 1.80?

# %%
producto = consultar_producto("arroz")
subtotal = Decimal(producto["precio_usd"]) * 2
print("Subtotal:", f"{subtotal:.2f}", "Fuente:", producto["fuente"])
assert subtotal == Decimal("3.60")

# %% [markdown]
# Guardamos dinero como texto y usamos `Decimal` para aritmética decimal.
# No pedimos a un modelo que calcule impuestos ni tipos de cambio.
#
# ## 2. Un contrato limita qué acepta la herramienta
# Nuestro pedido contiene exactamente `sku`, `cantidad` y `barrio`. `sku` es el
# identificador del producto. `cantidad` debe ser un entero entre 1 y 20.
# **Predice:** ¿deberíamos convertir automáticamente `"dos"` en 2?

# %%
pedido = {"sku": "arroz", "cantidad": 2, "barrio": "Centro"}
print(validar_pedido(pedido))
try:
    validar_pedido({**pedido, "cantidad": "dos"})
except ValueError as error:
    print("Entrada rechazada:", error)

# %% [markdown]
# ## Pausa y comprobación
# Explica quién interpreta una frase, quién valida argumentos y quién consulta datos.
# Esperado: LLM si lo usamos; contrato del programa; herramienta del programa.
#
# ## 3. Una cotización tiene evidencia y puede abstenerse
# No conocer un producto es distinto de tenerlo sin stock. No inventaremos un precio.

# %%
cotizacion = cotizar(pedido)
print(cotizacion)
assert cotizacion["total_usd"] == "5.60"  # 3.60 + envío de 2.00.
assert cotizar({**pedido, "sku": "leche"})["estado"] == "sin_producto"
assert cotizar({**pedido, "sku": "cuaderno"})["estado"] == "sin_stock"

# %% [markdown]
# ## 4. Error útil: reglas no equivalen a comprensión
# Un router de palabras clave puede confundir una pregunta negativa o un mensaje
# con dos temas. Un LLM puede manejar más variantes y también introducir errores.
# Conservar la herramienta exacta y verificar ambas alternativas ayuda a compararlas.
#
# ## Segunda pausa · Taller
# Agrega en una celda una lista de tres pedidos: válido, cantidad negativa y
# producto inexistente. Devuelve para cada uno su estado, sin detener el lote.
# **Pista:** captura solo `ValueError`; no conviertas cualquier fallo en "todo bien".
# Esperado: `cotizado`, `entrada_invalida`, `sin_producto`.
#
# ## Solución comentada

# %%
pedidos = [pedido, {**pedido, "cantidad": -1}, {**pedido, "sku": "leche"}]
estados = []
for caso in pedidos:
    try:
        estados.append(cotizar(caso)["estado"])
    except ValueError:
        estados.append("entrada_invalida")
assert estados == ["cotizado", "entrada_invalida", "sin_producto"]
print(estados)

# %% [markdown]
# ## Ticket de salida
# ¿En qué paso aportaría valor un modelo? ¿Cuál mantendrías como código exacto?
# ¿Por qué una referencia como `CAT-ARROZ` ayuda, pero no prueba por sí sola que
# una respuesta libre respete toda la información del catálogo?
# Siguiente: [02 · Modelado y cadena](02_modelado_y_cadena.ipynb).
