# %% [markdown]
# # 05 · Evaluador y optimizador: corregir con un límite
#
# **Meta:** revisar un borrador contra criterios verificables y detener una falla.
# **Producto:** generador → evaluador → feedback → corrección, con máximo de intentos.
# El generador y el evaluador pueden usar LLMs. Primero construimos una versión
# determinista para observar exactamente qué pasó. No es una evaluación semántica
# de un texto libre: comprobamos tres campos estructurados.

# %%
from henry_agents.workflows import cotizar, evaluar_borrador, revisar_hasta_limite

cotizacion = cotizar({"sku": "arroz", "cantidad": 2, "barrio": "Centro"})

# %% [markdown]
# ## 1. Define el criterio antes de generar
# **Predice:** ¿aprobarías un total correcto con una fuente inventada?
# Reglas: total observado, fuentes exactas y aprobación humana requerida.

# %%
borrador_malo = {"total_usd": "1.00", "fuentes": ["INVENTADA"], "requiere_aprobacion": False}
feedback = evaluar_borrador(borrador_malo, cotizacion)
print(feedback)
assert len(feedback) == 3

# %% [markdown]
# ## 2. Un generador guionado permite inspeccionar una corrección
# La primera salida falla a propósito. Con feedback devuelve los datos de la
# herramienta; no aprende ni razona. En live habría que comprobar también el texto.

# %%
def generar(feedback):
    if not feedback:
        return dict(borrador_malo)
    return {
        "total_usd": cotizacion["total_usd"],
        "fuentes": list(cotizacion["fuentes"]),
        "requiere_aprobacion": True,
    }


revision = revisar_hasta_limite(generar, cotizacion, max_intentos=3)
print(revision)
assert revision["estado"] == "validado"
assert len(revision["historial"]) == 2

# %% [markdown]
# ## Pausa y comprobación
# Señala qué cambió entre intento 1 y 2. ¿Validado significa enviado al cliente?
# No: la validación automática y la decisión de una persona son pasos distintos.
#
# ## 3. Error útil: el generador ignora el feedback
# **Predice:** ¿qué debe devolver el sistema cuando agota su presupuesto de intentos?

# %%
def generador_atascado(feedback):
    return dict(borrador_malo)


fallo = revisar_hasta_limite(generador_atascado, cotizacion, max_intentos=2)
assert fallo["estado"] == "limite_alcanzado"
assert fallo["borrador"] is None
assert len(fallo["historial"]) == 2
print(fallo)

# %% [markdown]
# ## Segunda pausa · Taller
# Escribe un generador que tenga el total correcto pero no las fuentes. Debe
# corregirse en el segundo intento. Luego reduce el límite a 1 y compara estados.
# **Pista:** no busques la palabra "aprobado" en un texto; usa errores estructurados.
#
# ## Solución comentada

# %%
def generar_sin_fuentes(feedback):
    return {
        "total_usd": cotizacion["total_usd"],
        "fuentes": list(cotizacion["fuentes"]) if feedback else [],
        "requiere_aprobacion": True,
    }


assert revisar_hasta_limite(generar_sin_fuentes, cotizacion, 2)["estado"] == "validado"
assert revisar_hasta_limite(generar_sin_fuentes, cotizacion, 1)["estado"] == "limite_alcanzado"

# %% [markdown]
# ## Ticket de salida
# ¿Por qué repetir indefinidamente no garantiza una buena respuesta? ¿Qué límite
# de llamadas/costo agregarías si cada intento consultara dos modelos?
# ¿Qué podría pasar aunque un texto cite IDs válidos? Podría atribuirles algo falso.
# Siguiente: [06 · Orquestador y workers](06_orquestador_workers.ipynb).
