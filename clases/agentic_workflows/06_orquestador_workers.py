# %% [markdown]
# # 06 · Orquestador y workers: un plan según el pedido
#
# **Meta:** diferenciar una lista fija de tareas de un plan que cambia con la entrada.
# **Producto:** plan explícito y ejecución limitada a workers registrados.
# *Orquestar* es organizar qué hace cada componente y cómo se juntan sus resultados.
# Un orquestador con LLM puede proponer un plan; aquí lo propone una regla visible.
# Esto permite estudiar el contrato sin simular que Python es un modelo.

# %%
from henry_agents.workflows import ejecutar_plan, normalizar, planificar, validar_pedido

pedido = {"sku": "arroz", "cantidad": 2, "barrio": "Centro"}

# %% [markdown]
# ## 1. Construye un plan pequeño
# **Predice:** ¿qué consulta adicional necesita entrega frente a retiro?

# %%
def proponer_plan(pedido):
    pedido = validar_pedido(pedido)
    pasos = ["producto"]
    if normalizar(pedido["barrio"]) != "retiro":
        pasos.append("entrega")
    return pasos


assert proponer_plan(pedido) == ["producto", "entrega"]
assert proponer_plan({**pedido, "barrio": "Retiro"}) == ["producto"]
print("Plan para entrega:", proponer_plan(pedido))

# %% [markdown]
# ## 2. Los nombres del plan deben tener una implementación registrada
# A diferencia de la clase 04, el número de tareas ahora cambia según el pedido.
# Puede ejecutarse secuencialmente o en paralelo si son independientes; tener un
# orquestador no implica necesariamente paralelismo ni autonomía del modelo.

# %%
plan = planificar(pedido)
resultados = ejecutar_plan(pedido, plan)
print("Plan:", plan)
print("Resultados:", resultados)
assert set(resultados) == {"producto", "entrega"}

# %% [markdown]
# ## Pausa y comprobación
# Compara el plan de retiro con el de entrega. ¿Qué tarea sobra si retiramos?
# Si pierdes el hilo, vuelve a la definición de `pedido` y lee `plan` en voz alta.
#
# ## 3. Error útil: una tarea que no existe o un plan incompleto
# Nunca ejecutamos código arbitrario cuyo nombre aparece en la salida de un modelo.
# Se valida todo el plan antes del primer worker; un plan de entrega debe incluir entrega.

# %%
for plan_invalido in (["producto", "cobrar_tarjeta"], ["producto"], ["producto", "producto"]):
    try:
        ejecutar_plan(pedido, plan_invalido)
    except ValueError as error:
        print("Plan rechazado:", plan_invalido, "→", error)

# %% [markdown]
# ## Segunda pausa · Taller
# Haz dos pedidos idénticos salvo el barrio: Centro y Retiro. Ejecuta ambos planes
# y comprueba cuántos workers trabajaron. Propón una tarea nueva y escribe primero
# su contrato de entrada/salida, sin incorporarla todavía al registro.
# **Pista:** el largo del diccionario de resultados debe coincidir con el plan.
#
# ## Solución comentada

# %%
for barrio, cantidad_tareas in [("Centro", 2), ("Retiro", 1)]:
    variante = {**pedido, "barrio": barrio}
    plan = planificar(variante)
    resultados = ejecutar_plan(variante, plan)
    assert len(resultados) == cantidad_tareas
    print(barrio, plan, resultados)

# %% [markdown]
# ## Ticket de salida
# ¿En qué difieren routing, paralelismo fijo y un plan variable? ¿Qué validarías
# si un modelo propusiera 50 tareas o inventara un worker? ¿Cuándo usarías solo
# una cadena? Los planes también pueden omitir pasos: hay que comprobar completitud.
# Siguiente: [07 · Proyecto integrador](07_proyecto_integrador.ipynb).
