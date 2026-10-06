# %% [markdown]
# # 00 · Python para empezar: la tienda La Esquina
#
# **Meta:** leer una lista, un diccionario y una función antes de hablar de agentes.
# No necesitas experiencia con IA. Si ya conoces Python, resuelve el reto del final.
# Una tienda de barrio también puede llamarse almacén, bodega o abarrotería.
# Los datos, precios y barrios de este curso son inventados.
#
# Abre este notebook con el kernel `.venv`. Ejecuta de arriba hacia abajo con
# Shift + Enter. Una celda es un bloque; el kernel guarda variables entre celdas.
# Si algo queda confuso, reinicia el kernel y usa **Run All / Ejecutar todo**.
#
# **Predice:** ¿consultar el precio de un producto requiere IA?

# %%
import sys

from henry_agents.workflows import cargar_tienda

print("Python:", sys.version.split()[0])
print("Entorno:", sys.prefix)
tienda = cargar_tienda()
print(tienda["nombre"])

# %% [markdown]
# ## 1. Un diccionario relaciona nombres con valores
# `producto["nombre"]` pide el valor de esa clave. Las comillas indican texto.
# `cantidad = 2` guarda un entero. `=` asigna; `==` compara.
# USD es una unidad didáctica: no representa la moneda de todos nuestros países.

# %%
producto = tienda["productos"][0]  # [0] es el primer elemento de una lista.
cantidad = 2
print(producto["nombre"], producto["precio_usd"], cantidad)
assert producto["sku"] == "arroz"

# %% [markdown]
# ## 2. Una lista contiene varios elementos
# Un `for` repite el bloque indentado para cada producto. El espacio al inicio
# de una línea tiene significado en Python; no lo borres al copiar.
# **Predice:** ¿qué producto tiene cero unidades disponibles?

# %%
for item in tienda["productos"]:
    print(item["sku"], "→ unidades:", item["stock"])

# %% [markdown]
# ## Pausa y comprobación
# Explica con tus palabras la diferencia entre `tienda["productos"]` y
# `tienda["productos"][0]`. Esperado: lista completa frente a primer diccionario.
# Si no salió la celda anterior, vuelve a ejecutar la que define `tienda`.
#
# ## 3. Una función recibe una entrada y devuelve una salida
# `return` entrega el resultado. `if` decide según una condición explícita.
# **Predice:** el arroz tiene 8 unidades. ¿Podemos atender 9?

# %%
def alcanza_stock(producto, cantidad):
    return cantidad <= producto["stock"]


print(alcanza_stock(producto, 2), alcanza_stock(producto, 9))
assert alcanza_stock(producto, 2) is True
assert alcanza_stock(producto, 9) is False

# %% [markdown]
# ## 4. Error útil: una clave que no existe
# No hace falta adivinar qué pasó: la última línea del error indica su tipo.
# `try/except` permite mostrar esta falla intencional sin detener la clase.

# %%
try:
    print(producto["precio"])
except KeyError:
    print("KeyError: la clave se llama precio_usd. Revisa las claves:", list(producto))

# %% [markdown]
# ## Segunda pausa · Taller
# Escribe una función `productos_disponibles(items)` que devuelva solo los
# identificadores con stock mayor que cero. Usa `for`, `if` y `.append()`.
# **Pista:** empieza con `disponibles = []`; agrega `item["sku"]` dentro del `if`.
# Haz el intento en una celda nueva antes de seguir. Esperado: `['arroz', 'cafe']`.
# Una celda nueva de práctica no se incluye en la verificación hasta sincronizarla.
#
# ## Solución comentada

# %%
def productos_disponibles(items):
    disponibles = []
    for item in items:
        if item["stock"] > 0:
            disponibles.append(item["sku"])
    return disponibles


assert productos_disponibles(tienda["productos"]) == ["arroz", "cafe"]
print(productos_disponibles(tienda["productos"]))

# %% [markdown]
# ## Ticket de salida
# 1. Señala entrada, condición y salida de `alcanza_stock`.
# 2. ¿Por qué esta función es automatización y no un LLM?
# 3. Cambia el stock del cuaderno a 1 en una copia local. ¿Qué debería cambiar?
#
# **Lo aprendido:** Python puede aplicar reglas verificables sin IA. Un LLM será
# útil para lenguaje flexible; no hace falta usarlo para contar unidades.
# Siguiente: [01 · IA, herramientas y límites](01_ia_y_herramientas.ipynb).
