# %% [markdown]
# # 03 · Routing: elegir quién atiende
#
# **Meta:** enviar una consulta a una sola área y reconocer ambigüedad.
# **Producto:** router con tres destinos permitidos: productos, entregas y humano.
# *Routing* significa seleccionar una ruta. Piensa en una recepción que decide
# a qué mostrador te dirige; no necesitas abrir todos los mostradores a la vez.

# %%
from henry_agents.workflows import clasificar_con_modelo, clasificar_por_reglas

mensajes = ["¿Tienen café?", "¿Hacen ENVÍOS?", "Quiero presentar un reclamo"]
for mensaje in mensajes:
    print(mensaje, "→", clasificar_por_reglas(mensaje))
assert [clasificar_por_reglas(m) for m in mensajes] == ["productos", "entregas", "humano"]

# %% [markdown]
# ## 1. Construye primero una regla entendible
# `casefold()` baja mayúsculas; la normalización completa del helper también quita
# tildes. Este ejemplo mínimo permite comparar qué mejora el helper.
# **Predice:** ¿por qué buscar `"cafe"` no reconoce `"CAFÉ"` sin normalización?

# %%
def router_minimo(mensaje):
    if "precio" in mensaje.casefold().split():
        return "productos"
    return "humano"


print(router_minimo("precio del arroz"))
print(router_minimo("¿Precio?"))  # La puntuación muestra un límite de .split().
assert router_minimo("¿Precio?") == "humano"

# %% [markdown]
# ## Pausa y comprobación
# Lee `clasificar_por_reglas` en `src/henry_agents/workflows.py` y encuentra dónde
# se quitan tildes y dónde se buscan palabras completas. No hay embeddings aquí.
#
# ## 2. Error útil: dos temas y una negación
# **Predice:** ¿a dónde va "precio y envío"? ¿Y "No quiero café"?

# %%
assert clasificar_por_reglas("Precio y envío") == "humano"
assert clasificar_por_reglas("No quiero café") == "productos"
assert clasificar_por_reglas("El caféteria está cerrada") == "humano"
print("Dos áreas → humano. Negación → productos: la regla no comprende intención.")

# %% [markdown]
# ## 3. Comparación opcional con un modelo real
# Esta celda está apagada para que todos participen sin cuenta ni saldo.
# Si decides activarla, sigue `docs/INSTALACION.md`, configura el modo live y
# comprueba un modelo disponible en tu cuenta. Consume API y puede fallar.
# La respuesta se valida con un contrato `area` + `motivo`. No se ejecutan nombres
# arbitrarios ni se vuelve silenciosamente al router de reglas ante un error.
# Una clasificación con LLM sigue siendo un paso de workflow; no es un bucle autónomo.

# %%
USAR_MODELO_REAL = False  # Cambia a True solo para el experimento voluntario.
if USAR_MODELO_REAL:
    for mensaje in mensajes:
        print(mensaje, "→", clasificar_con_modelo(mensaje))
else:
    print("Experimento live desactivado. Llamadas a API: 0.")

# %% [markdown]
# ## Segunda pausa · Taller
# Define una tabla de cinco mensajes: producto, envío con tilde, reclamo, dos temas
# y negación. Anota antes la ruta que esperas y explica al menos un desacuerdo.
# **Pista:** la tabla debe tener entradas distintas, no cinco variantes del mismo éxito.
#
# ## Solución comentada

# %%
casos = [
    ("Precio del arroz", "productos"),
    ("Necesito envío", "entregas"),
    ("Reclamo por una compra", "humano"),
    ("Café con entrega", "humano"),
    ("No quiero café", "productos"),  # Describe la regla actual, no la intención ideal.
]
for mensaje, esperado in casos:
    obtenido = clasificar_por_reglas(mensaje)
    assert obtenido == esperado
    print(mensaje, obtenido)

# %% [markdown]
# ## Ticket de salida
# ¿Cuándo conviene una regla? ¿Qué mejora esperarías de un modelo y cómo la medirías?
# ¿Por qué dejamos una ruta humana para consultas ambiguas?
# Siguiente: [04 · Paralelismo](04_paralelismo.ipynb).
