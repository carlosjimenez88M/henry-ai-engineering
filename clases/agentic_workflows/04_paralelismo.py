# %% [markdown]
# # 04 · Paralelismo: consultar cosas independientes
#
# **Meta:** distinguir dependencia de independencia y recoger resultados de workers.
# **Producto:** consultar producto y cobertura con dos tareas concurrentes.
# Un *worker* es quien realiza una subtarea. Puede ser una función; no necesariamente
# es un agente con un modelo. *Concurrencia* permite solapar tareas; no garantiza
# que este ejemplo pequeño sea más rápido ni que use varios núcleos de CPU.
#
# ```mermaid
# flowchart LR
#   P[Pedido] --> C[Consultar catálogo]
#   P --> E[Consultar entrega]
#   C --> J[Reunir resultados]
#   E --> J
# ```

# %%
from concurrent.futures import ThreadPoolExecutor

from henry_agents.workflows import cargar_tienda, consultar_producto, ejecutar_paralelo

tienda = cargar_tienda()

# %% [markdown]
# ## 1. Comprueba las dos consultas por separado
# **Predice:** ¿la entrega a Centro necesita saber el precio del arroz?
# En estas políticas inventadas no depende del producto. Si el costo dependiera
# del peso total, habría que calcular ese peso antes de consultar el envío.

# %%
def consultar_entrega():
    return dict(tienda["entregas"]["centro"])


secuencial = {"producto": consultar_producto("arroz"), "entrega": consultar_entrega()}
print(secuencial)

# %% [markdown]
# ## 2. Crea futuros y recoge sus resultados
# Un `Future` es un resultado pendiente. `.result()` espera y también propaga
# errores: una tarea que falla no debe desaparecer del reporte.

# %%
with ThreadPoolExecutor(max_workers=2) as equipo:
    futuro_producto = equipo.submit(consultar_producto, "arroz")
    futuro_entrega = equipo.submit(consultar_entrega)
    paralelo = {"producto": futuro_producto.result(), "entrega": futuro_entrega.result()}
assert paralelo == secuencial
print("Mismos datos; las tareas pueden solaparse.")

# %% [markdown]
# ## Pausa y comprobación
# Explica por qué el redactor de la clase 02 no puede ejecutarse en paralelo con
# una cotización que todavía no existe. Nombra una tarea independiente de tu trabajo.
#
# ## 3. Error útil: un worker falla
# No hay datos compartidos en un diccionario global. Cada worker devuelve su salida.

# %%
def worker_con_falla():
    raise RuntimeError("Falla didáctica al consultar cobertura")


try:
    ejecutar_paralelo({"producto": lambda: consultar_producto("arroz"), "entrega": worker_con_falla})
except RuntimeError as error:
    print("El coordinador observa el error:", error)

# %% [markdown]
# ## Segunda pausa · Taller
# Construye tres tareas: arroz, café y cobertura. Usa `ejecutar_paralelo` y verifica
# que el café tiene 3 unidades. Luego sustituye solo una tarea por la falla anterior.
# **Pista:** `lambda: consultar_producto("cafe")` es una función sin argumentos;
# `consultar_producto("cafe")` ya ejecutó la consulta y devuelve un diccionario.
#
# ## Solución comentada

# %%
tareas = {
    "arroz": lambda: consultar_producto("arroz"),
    "cafe": lambda: consultar_producto("cafe"),
    "cobertura": consultar_entrega,
}
resultados = ejecutar_paralelo(tareas)
assert resultados["cafe"]["stock"] == 3
assert list(resultados) == ["arroz", "cafe", "cobertura"]
print(resultados)

# %% [markdown]
# ## Ticket de salida
# ¿El orden de finalización de los hilos está garantizado? No. La colección final
# usa el orden de entrada para poder compararla, no para afirmar cuál terminó primero.
# ¿Cuándo el paralelismo añade complejidad sin aportar valor?
# Siguiente: [05 · Evaluador y optimizador](05_evaluador_optimizador.ipynb).
