# %% [markdown]
# # FastAPI: servir un workflow de IA para cómics
#
# **Bloque 5: una API para el workflow de cómics.**
# Convertiremos una función de Python en un servicio HTTP: leer un pedido, validar
# su cuerpo, ejecutar el workflow y devolver un resultado estructurado. Empezamos
# con una API pequeña antes de utilizar el backend completo del proyecto.
#
# | Orden | Actividad |
# |---|---|
# | 1 | HTTP, rutas, decoradores y primer endpoint |
# | 2 | Parámetros y ejercicio 1 |
# | 3 | POST, contratos y ejercicios 2–3 |
# | 4 | Fallos de modelos y ejercicio 4 |
# | 5 | Entrega de un caso propio y explicación |
#
# Las pruebas usan `TestClient`: envía peticiones a la aplicación dentro del mismo
# proceso. No abre un puerto ni depende de un servidor externo. El modelo real se
# activa después, en la configuración del servidor; este notebook es reproducible
# sin clave y no llama a un LLM al ejecutarlo completo.

# %%
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from henry_agents.comics import (
    PedidoComic,
    ProveedorComicError,
    SalidaComicInvalida,
    cargar_fuentes,
)
from henry_agents.comics_api import crear_api

# %% [markdown]
# ## 1. Una petición, una función, una respuesta
#
# HTTP permite que otro programa llame a nuestro servicio. `GET` consulta datos;
# `POST` envía un cuerpo para procesarlo. La respuesta incluye un código de estado
# y, en estos endpoints, un cuerpo JSON.
#
# Un **decorador** como `@api_intro.get("/health")` registra la función que está
# debajo como manejador de esa ruta. FastAPI convierte su diccionario de retorno
# a JSON. Definir la función no significa ejecutarla: la petición la activa.
#
# | Código | Significado en este proyecto |
# |---|---|
# | 200 | Se pudo procesar la petición; leer también `estado` del guion |
# | 422 | El cuerpo no cumple el contrato de entrada |
# | 502 | La salida del modelo incumple un contrato o control del servicio |
# | 503 | Falta configuración o el proveedor no está disponible |

# %%
api_intro = FastAPI(title="Mi primera API de cómics")


@api_intro.get("/health")
def health_intro():
    return {"status": "ok", "proyecto": "comics"}


@api_intro.post("/pedido", response_model=PedidoComic)
def recibir_pedido(pedido: PedidoComic):
    return pedido


with TestClient(api_intro) as cliente_intro:
    respuesta_health = cliente_intro.get("/health")
print("Código:", respuesta_health.status_code)
print("Cuerpo:", respuesta_health.json())

# %% [markdown]
# El endpoint `/pedido` recibe y devuelve el contrato aprendido; todavía no genera
# un guion. La anotación `pedido: PedidoComic` permite que FastAPI valide el cuerpo.
# `response_model=PedidoComic` define y valida la forma de la respuesta.

# %%
body_base = {
    "tema": "Preparar una feria de ciencias mientras se organiza la ayuda del barrio.",
    "heroes": ["spiderman", "fantasticos"],
    "tono": "aventura",
    "max_vinetas": 4,
}
with TestClient(api_intro) as cliente_intro:
    respuesta_pedido = cliente_intro.post("/pedido", json=body_base)
print("POST:", respuesta_pedido.status_code, respuesta_pedido.json())

# %% [markdown]
# **Ejercicio 1 — un parámetro en la ruta.**
# La ruta `/heroes/{heroe}` recibe un nombre en `heroe`. Completa el bucle para
# contar sus fuentes en `cargar_fuentes()`. Para un ID desconocido devolverá cero.
# Prueba `batman` y un nombre sin fuentes. Vuelve a ejecutar la celda completa
# después de editarla: se crea una aplicación nueva y no se acumulan rutas.

# %%
api_practica = FastAPI(title="Práctica de parámetros")
print("Fuentes para consultar:", len(cargar_fuentes()))


@api_practica.get("/heroes/{heroe}")
def contar_fuentes_practica(heroe: str):
    cantidad = 0
    # Recorre cargar_fuentes() y suma si fuente["heroe"] == heroe.
    return {"heroe": heroe, "cantidad": cantidad}


with TestClient(api_practica) as cliente_practica:
    consulta_batman = cliente_practica.get("/heroes/batman")
    consulta_desconocido = cliente_practica.get("/heroes/personaje-sin-fuentes")
print("Hay fuentes de Batman:", consulta_batman.json()["cantidad"] > 0)
print("El desconocido tiene cero:", consulta_desconocido.json()["cantidad"] == 0)

# %% [markdown]
# <details><summary>Ejercicio 1: contar fuentes por héroe</summary>
#
# ```python
# def contar_fuentes_practica(heroe: str):
#     cantidad = 0
#     for fuente in cargar_fuentes():
#         if fuente["heroe"] == heroe:
#             cantidad += 1
#     return {"heroe": heroe, "cantidad": cantidad}
# assert contar_fuentes_practica("batman")["cantidad"] > 0
# assert contar_fuentes_practica("personaje-sin-fuentes")["cantidad"] == 0
# # Para probar la ruta modificada, edita y reejecuta su celda: el decorador
# # registró la función anterior cuando creaste api_practica.
# ```
#
# </details>

# %% [markdown]
# ## 2. La API completa ejecuta el mismo workflow
#
# [comics_api.py](../../src/henry_agents/comics_api.py) utiliza `PedidoComic` como
# entrada y `ResultadoComic` como respuesta. Su función principal es pequeña:
#
# ```python
# @api.post("/comics", response_model=ResultadoComic)
# def comics(pedido: PedidoComic):
#     return generar_comic(pedido, mode=selected, modelos=modelos)
# ```
#
# El archivo añade manejadores de errores para los fallos que exploraremos abajo.
# `selected` pertenece al servidor. El usuario HTTP envía el pedido, sin elegir
# modelos ni credenciales. Usamos `def` (no `async def`) porque el SDK del
# proyecto es síncrono; FastAPI lo ejecuta en su pool de hilos.

# %%
api_comics = crear_api(mode="offline")
with TestClient(api_comics) as cliente:
    health = cliente.get("/health")
    catalogo = cliente.get("/fuentes")
    respuesta_comic = cliente.post("/comics", json=body_base)

comic = respuesta_comic.json()
print("Health:", health.json())
print("Fuentes disponibles:", len(catalogo.json()["fuentes"]))
print("POST /comics:", respuesta_comic.status_code)
print("Estado narrativo:", comic["estado"])
print("Título:", comic["guion"]["titulo"])
print("Viñetas:", len(comic["guion"]["vinetas"]))
print("Críticas:", [revision["ciclo"] for revision in comic["revisiones"]])
print("Invocaciones reales:", comic["llamadas_modelo"])
for indice, version in enumerate(comic["versiones"]):
    print("Versión", indice, "primera viñeta:", version["vinetas"][0]["descripcion"])

# %% [markdown]
# **Ejercicio 2 — tu pedido por HTTP.**
# Envía una aventura sobre organizar una biblioteca con Batman y tres viñetas.
# Usa un nuevo diccionario en `body_propio`. Conserva las tres comprobaciones;
# revisa también la primera viñeta y describe qué cambió en las reescrituras.
# Compara `versiones[0]` (borrador), `[1]` (ciclo 1) y `[2]` (ciclo 2); cada
# elemento contiene un guion completo. Elige un cambio y explica si mejoró el texto.
# Un 200 puede contener `requiere_revision` en modo real: HTTP exitoso y aprobación
# narrativa representan resultados diferentes.

# %%
body_propio = None
# Construye el cuerpo con tema, heroes, tono y max_vinetas.
if body_propio is not None:
    with TestClient(api_comics) as cliente:
        respuesta_propia = cliente.post("/comics", json=body_propio)
    print("Respuesta 200:", respuesta_propia.status_code == 200)
    if respuesta_propia.status_code == 200:
        datos_propios = respuesta_propia.json()
        print("Tres viñetas:", len(datos_propios["guion"]["vinetas"]) == 3)
        print("Dos ciclos y revisión final:", [r["ciclo"] for r in datos_propios["revisiones"]] == [1, 2, 3])
        print(datos_propios["guion"]["vinetas"][0])
else:
    print("Ejercicio pendiente: prepara tu cuerpo JSON.")

# %% [markdown]
# <details><summary>Ejercicio 2: cuerpo propio y respuesta</summary>
#
# ```python
# body_propio = {
#     "tema": "Organizar una biblioteca barrial antes de una tormenta.",
#     "heroes": ["batman"], "tono": "aventura", "max_vinetas": 3,
# }
# with TestClient(api_comics) as cliente:
#     respuesta_propia = cliente.post("/comics", json=body_propio)
# assert respuesta_propia.status_code == 200
# datos_propios = respuesta_propia.json()
# assert len(datos_propios["guion"]["vinetas"]) == 3
# assert [r["ciclo"] for r in datos_propios["revisiones"]] == [1, 2, 3]
# ```
#
# </details>

# %% [markdown]
# ## 3. Entradas que no llegan al modelo
#
# Rechazamos cantidades fuera de rango, héroes desconocidos, repeticiones y campos
# extra. `True` no es una cantidad válida aunque en Python `bool` derive de `int`:
# el contrato pide un entero estricto. Un 422 indica dónde falla la entrada.
# La API completa omite los valores recibidos al reportar errores de validación.

# %%
cuerpo_invalido = {
    **body_base,
    "max_vinetas": "3",
    "modo": "live",
}
with TestClient(api_comics) as cliente:
    respuesta_invalida = cliente.post("/comics", json=cuerpo_invalido)
    respuesta_bool = cliente.post("/comics", json={**body_base, "max_vinetas": True})
print("Cuerpo inválido:", respuesta_invalida.status_code)
for error in respuesta_invalida.json()["detail"]:
    print("Campo:", error["loc"], "regla:", error["type"])
print("True como cantidad:", respuesta_bool.status_code)

# %% [markdown]
# **Ejercicio 3 — corregir un 422.**
# Copia `cuerpo_invalido`, transforma la cantidad en el entero `3` y elimina `modo`
# con `.pop("modo")`. Envía la versión corregida. Explica por qué corregir sólo
# la cantidad todavía dejaría un error. No modifiques `cuerpo_invalido`.

# %%
cuerpo_corregido = cuerpo_invalido.copy()
# Corrige max_vinetas y elimina modo.
with TestClient(api_comics) as cliente:
    respuesta_corregida = cliente.post("/comics", json=cuerpo_corregido)
print("Corrección completa:", respuesta_corregida.status_code == 200)
print("Entrada original conservada:", cuerpo_invalido["max_vinetas"] == "3" and "modo" in cuerpo_invalido)

# %% [markdown]
# <details><summary>Ejercicio 3: corregir ambos campos</summary>
#
# ```python
# cuerpo_corregido = cuerpo_invalido.copy()
# cuerpo_corregido["max_vinetas"] = 3
# cuerpo_corregido.pop("modo")
# with TestClient(api_comics) as cliente:
#     respuesta_corregida = cliente.post("/comics", json=cuerpo_corregido)
# assert respuesta_corregida.status_code == 200
# assert cuerpo_invalido["max_vinetas"] == "3" and "modo" in cuerpo_invalido
# ```
#
# </details>

# %% [markdown]
# ## 4. Simular fallos sin consumir modelos
#
# Un **doble de prueba** reemplaza temporalmente una dependencia. `patch` sustituye
# la función que la API llama y la restaura al salir del `with`. Aquí provocamos
# los dos errores del servicio sin enviar información a un proveedor.
# Esto comprueba la traducción a HTTP; no reproduce una caída real de OpenAI.
#
# Son situaciones distintas: cuerpo mal formado (422), salida del modelo inválida
# (502) y proveedor o configuración indisponibles (503). Cambiar el pedido no
# siempre resuelve un 503. Las respuestas no incluyen excepciones internas.

# %%
with patch("henry_agents.comics_api.generar_comic", side_effect=ProveedorComicError("Fallo interno simulado")):
    with TestClient(api_comics) as cliente:
        fallo_proveedor = cliente.post("/comics", json=body_base)

with patch("henry_agents.comics_api.generar_comic", side_effect=SalidaComicInvalida("Referencia inexistente simulada")):
    with TestClient(api_comics) as cliente:
        fallo_salida = cliente.post("/comics", json=body_base)

print("Proveedor:", fallo_proveedor.status_code, fallo_proveedor.json())
print("Salida:", fallo_salida.status_code, fallo_salida.json())

# %% [markdown]
# **Ejercicio 4 — decidir el siguiente paso.**
# Completa una función con `if` y `return`: ante 422 devuelve `"corregir entrada"`,
# ante 502 `"revisar salida"`, ante 503 `"revisar servicio"`; para otros códigos,
# `"inspeccionar respuesta"`. Explica un caso en el que repetir la petición
# sin diagnosticar el problema gastaría otras llamadas.

# %%
def siguiente_paso_practica(codigo: int) -> str:
    return "pendiente"


print("422:", siguiente_paso_practica(422) == "corregir entrada")
print("502:", siguiente_paso_practica(502) == "revisar salida")
print("503:", siguiente_paso_practica(503) == "revisar servicio")
print("Otro código:", siguiente_paso_practica(200) == "inspeccionar respuesta")

# %% [markdown]
# <details><summary>Ejercicio 4: acciones según el estado</summary>
#
# ```python
# def siguiente_paso_practica(codigo: int) -> str:
#     if codigo == 422:
#         return "corregir entrada"
#     if codigo == 502:
#         return "revisar salida"
#     if codigo == 503:
#         return "revisar servicio"
#     return "inspeccionar respuesta"
# assert siguiente_paso_practica(422) == "corregir entrada"
# assert siguiente_paso_practica(502) == "revisar salida"
# assert siguiente_paso_practica(503) == "revisar servicio"
# assert siguiente_paso_practica(200) == "inspeccionar respuesta"
# ```
#
# </details>

# %% [markdown]
# ## 5. El contrato publicado: OpenAPI y `/docs`
#
# FastAPI crea un documento OpenAPI a partir de rutas y contratos. `/docs` ofrece
# una interfaz interactiva sobre ese documento. Un cliente puede conocer campos
# de entrada y respuesta antes de enviar un pedido.

# %%
with TestClient(api_comics) as cliente:
    contrato_http = cliente.get("/openapi.json").json()
post_comics = contrato_http["paths"]["/comics"]["post"]
print("Contrato de entrada:", post_comics["requestBody"]["content"]["application/json"]["schema"])
print("Contrato de respuesta:", post_comics["responses"]["200"]["content"]["application/json"]["schema"])

# %% [markdown]
# ## 6. Entrega final
#
# En una copia del notebook, entrega:
#
# 1. Tu pedido y respuesta 200 con tres viñetas, o el fallo real observado.
# 2. Un 422 y su corrección, conservando ambos cuerpos.
# 3. Una consulta a `/health` para diagnosticar el modo del servidor.
# 4. Una explicación de las dos críticas, las dos reescrituras y la revisión final.
# 5. Una limitación del guion que no certifiquen Pydantic ni los controles de IDs.
#
# Muestra tus comprobaciones corregidas y justifica al menos una decisión de código.
#
# ## Ampliación opcional: servidor y modelos reales
#
# Desde la raíz del repo, abre una terminal y ejecuta:
#
# ```bash
# COURSE_MODE=offline uv run uvicorn henry_agents.comics_api:app --host 127.0.0.1 --port 8000
# ```
#
# Abre `http://127.0.0.1:8000/docs`, prueba `GET /health` y luego `POST /comics`.
# Uvicorn mantiene el servidor activo; ciérralo con Ctrl + C.
#
# Para el laboratorio real, configura `OPENAI_API_KEY` y `COURSE_MODE=live` en
# el `.env` privado y reinicia el servidor. No incluyas claves en los cuerpos HTTP.
# La variable explícita del comando anterior tiene prioridad: inicia entonces con
# `uv run uvicorn henry_agents.comics_api:app --host 127.0.0.1 --port 8000`.
# Cada POST válido puede consumir hasta siete invocaciones. `GET /health` informa
# modo y presupuesto, pero no comprueba credenciales ni llama al proveedor.
#
# La API sirve para esta práctica local. Despliegue, autenticación, límites globales
# de concurrencia y procesamiento por colas son temas de una clase posterior.
#
# Referencias: [cuerpo de petición](https://fastapi.tiangolo.com/tutorial/body/),
# [modelo de respuesta](https://fastapi.tiangolo.com/tutorial/response-model/),
# [TestClient](https://fastapi.tiangolo.com/tutorial/testing/),
# [errores HTTP](https://fastapi.tiangolo.com/tutorial/handling-errors/) y
# [funciones síncronas y asíncronas](https://fastapi.tiangolo.com/async/).
