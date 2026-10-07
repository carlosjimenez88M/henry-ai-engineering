# %% [markdown]
# # De Python a un flujo de modelos: cómics con dos ciclos de revisión
#
# **Bloque 4: modelos y revisión.**
# Ya sabemos preparar diccionarios y validar datos. Ahora programaremos las fases
# de un sistema de IA: pedido, evidencia, brief, borrador, crítica y reescritura.
# Al terminar podrás explicar una traza, detectar referencias inválidas y ejecutar
# el mismo flujo con los modelos del `.env`.
#
# | Orden | Actividad |
# |---|---|
# | 0 | Una sola llamada a un modelo |
# | 1 | Pedido, fuentes y configuración |
# | 2 | Contratos y borrador estructurado; ejercicio 1 |
# | 3 | Dos ciclos de crítica y reescritura; ejercicio 2 |
# | 4 | Controles y trazas; ejercicio 3 |
# | 5 | Laboratorio con modelos y consumo; ejercicio 4 |
# | 6 | Explicar el resultado |
#
# Las escenas son originales, ficticias y no representan canon. `offline` ejecuta
# reglas escritas en Python: sirve para estudiar el programa, sin evaluar la
# capacidad de un LLM. `live` ejecuta modelos reales y puede fallar.
#
# > **Límites de este bloque.** Un JSON válido, un ID existente o una crítica
# > favorable no garantizan que el guion sea fiel a sus fuentes ni que tenga
# > calidad: hay que leerlo. Separar instrucciones y datos expresa la intención,
# > pero no protege por sí solo contra instrucciones maliciosas dentro de una
# > fuente. Más ciclos de revisión no aseguran un mejor resultado.

# %%
import json
import os

from pydantic import ValidationError

from henry_agents.comics import (
    INSTRUCCIONES_BASE,
    BriefComic,
    ConfiguracionComicError,
    GuionComic,
    PedidoComic,
    ProveedorComicError,
    SalidaComicInvalida,
    cargar_fuentes,
    crear_borrador,
    criticar_guion,
    extraer_brief,
    generar_comic,
    reescribir_guion,
    seleccionar_fuentes,
    validar_guion,
)
from henry_agents.config import ROOT, chat_model, configure, model_name

configure("offline")
print("Existe .env:", (ROOT / ".env").is_file())
print("Hay clave disponible:", bool(os.getenv("OPENAI_API_KEY")))
print("Extractor y crítico:", model_name("default"))
print("Redactor:", model_name("agent"))

# %% [markdown]
# ## 0. Antes del flujo: una sola llamada
#
# En el notebook 00 armaste una lista de mensajes con `role` y `content`. Un modelo
# de chat recibe exactamente eso y devuelve otro mensaje. Esta es la pieza mínima;
# todo el flujo de este bloque son varias llamadas como esta, encadenadas.
#
# Con la bandera desactivada solo ves los mensajes. Para activarla necesitas
# `OPENAI_API_KEY` en `.env` y reiniciar el kernel; es **una** llamada a
# `model_name("default")`. `respuesta.text` es el texto; `usage_metadata` cuenta
# los tokens que reporta el proveedor.

# %%
EJECUTAR_PRIMERA_LLAMADA = False
mensajes_simples = [
    {"role": "system", "content": "Responde en español, en una sola oración."},
    {"role": "user", "content": "Propón una escena breve donde Batman resuelva un misterio con pistas."},
]

if EJECUTAR_PRIMERA_LLAMADA:
    modelo_simple = chat_model("default", max_retries=0)
    respuesta = modelo_simple.invoke(mensajes_simples)
    print("Respuesta:", respuesta.text)
    print("Tokens:", respuesta.usage_metadata)
else:
    for mensaje in mensajes_simples:
        print(mensaje["role"], "→", mensaje["content"])
    print("Llamada pendiente: bandera desactivada. No se llamó al modelo.")

# %% [markdown]
# Una respuesta libre sirve para conversar, pero un programa necesita datos con
# forma conocida. Por eso el resto del bloque pide **salidas estructuradas** y las
# valida con contratos, como en el notebook 01.
#
# `PedidoComic` es el contrato central del proyecto. El `PedidoEscena` del bloque
# anterior era un modelo para practicar: exigía escribir `max_vinetas` y aún no
# comprobaba duplicados. Aquí omitir la cantidad produce cuatro viñetas y repetir
# un héroe se rechaza. Los mismos nombres de campos no implican idénticas reglas.

# %%
pedido_sin_cantidad = PedidoComic(tema="Investigar las luces de la biblioteca.", heroes=["batman"])
print("Cantidad por defecto:", pedido_sin_cantidad.max_vinetas)
try:
    PedidoComic(tema="Investigar las luces de la biblioteca.", heroes=["batman", "batman"])
except ValidationError as error:
    print("Héroes repetidos rechazados:", error.errors()[0]["loc"])

# %% [markdown]
# ## 1. De un pedido a evidencia que el programa controla
#
# Un brief es un plan breve: objetivo, conflicto y referencias para escribir.
# El pedido contiene datos del usuario; las fuentes aportan escenas de referencia.
# El sistema añade sus instrucciones por separado.
#
# `max_vinetas` se conserva como nombre del campo de la API. En este proyecto el
# guion debe tener **exactamente** esa cantidad; elegimos entre 2 y 6.

# %%
pedido = PedidoComic(
    tema="Organizar una feria de ciencias en el barrio ante un corte de energía.",
    heroes=["spiderman", "fantasticos"],
    tono="aventura",
    max_vinetas=4,
)
fuentes = seleccionar_fuentes(pedido)
print("Fuentes del catálogo:", len(cargar_fuentes()))
for fuente in fuentes:
    print(fuente["id"], fuente["heroe"], fuente["texto"])

# %% [markdown]
# ## 2. Salidas estructuradas: forma, reglas y calidad
#
# En lugar de pedir texto libre y recortarlo con `.split()`, solicitamos un objeto
# con un contrato Pydantic. La integración transforma ese contrato en una petición
# de salida estructurada. Después Python valida restricciones del proyecto.
#
# El adaptador real de [comics.py](../../src/henry_agents/comics.py) utiliza:
#
# ```python
# modelo = chat_model("default", max_retries=0)
# extractor = modelo.with_structured_output(BriefComic, include_raw=True)
# salida = extractor.invoke([("system", instrucciones), ("human", datos_json)])
# brief = salida["parsed"]
# uso_observado = salida["raw"].usage_metadata
# ```
#
# `parsed` contiene el objeto; `raw` conserva información de consumo. Una negativa,
# un corte de respuesta o un error de parsing puede impedir obtener el objeto.
# El adaptador lo reporta como fallo, sin inventar una salida exitosa.
#
# Tres revisiones diferentes: Pydantic valida campos y tipos; Python comprueba
# cantidad, numeración, IDs y citas literales; el crítico revisa claridad narrativa.

# %%
instrucciones = (
    INSTRUCCIONES_BASE
    + " Copia tema, heroes y tono exactamente. Cita texto literal de las fuentes."
)
datos_json = json.dumps({"pedido": pedido.model_dump(), "fuentes": fuentes}, ensure_ascii=False)
mensajes_extractor = [("system", instrucciones), ("human", datos_json)]
print("Instrucciones del sistema:", mensajes_extractor[0][1])
print("Datos del pedido y fuentes:", mensajes_extractor[1][1])
print("Campos de la salida esperada:", list(BriefComic.model_fields))

# %% [markdown]
# Estos son los mensajes que utiliza la fase de extracción. Las instrucciones
# describen la tarea; el JSON contiene el pedido y la evidencia. Editar el JSON
# no cambia el rol del mensaje. En las otras fases, el JSON añade el brief, el guion y el feedback correspondiente.

# %%
brief = extraer_brief(pedido, mode="offline")
borrador = crear_borrador(pedido, brief, mode="offline")
print(brief.model_dump_json(indent=2))
print(borrador.model_dump_json(indent=2))
print("Incumplimientos del borrador:", validar_guion(pedido, brief, borrador))

# %% [markdown]
# **Ejercicio 1 — inspeccionar referencias.**
# Completa una función que devuelva los IDs usados por las viñetas, sin repetirlos
# y conservando el orden de aparición. Usa un bucle y `if id not in lista`.
# Comprueba también un guion sin viñetas, representado por un diccionario.

# %%
def ids_practica(guion_dict: dict) -> list[str]:
    encontrados = []
    # Recorre guion_dict["vinetas"] y las fuentes de cada viñeta.
    return encontrados


caso_ids = {"vinetas": [{"fuentes": ["A", "B"]}, {"fuentes": ["B", "C"]}]}
print("IDs sin repetición:", ids_practica(caso_ids) == ["A", "B", "C"])
print("Guion vacío:", ids_practica({"vinetas": []}) == [])

# %% [markdown]
# <details><summary>Ejercicio 1: IDs sin repetir</summary>
#
# ```python
# def ids_practica(guion_dict: dict) -> list[str]:
#     encontrados = []
#     for vineta in guion_dict["vinetas"]:
#         for fuente_id in vineta["fuentes"]:
#             if fuente_id not in encontrados:
#                 encontrados.append(fuente_id)
#     return encontrados
# assert ids_practica(caso_ids) == ["A", "B", "C"]
# assert ids_practica({"vinetas": []}) == []
# ```
#
# </details>

# %% [markdown]
# ## 3. Dos ciclos de feedback sobre el guion
#
# Un ciclo contiene **crítica y reescritura**. Leer dos críticas sin cambiar el
# guion no forma dos ciclos de mejora. Guardamos cada versión para comparar.
#
# En el simulador, la primera crítica pide causa y consecuencia; la segunda pide
# encuadres. Son reglas didácticas explícitas. El modelo real recibe criterios de
# coherencia, evidencia, diálogo y ritmo; puede discrepar o repetir observaciones.

# %%
critica_1 = criticar_guion(pedido, brief, borrador, ciclo=1, mode="offline")
version_1 = reescribir_guion(pedido, brief, borrador, critica_1, ciclo=1, mode="offline")
critica_2 = criticar_guion(pedido, brief, version_1, ciclo=2, mode="offline")
version_2 = reescribir_guion(pedido, brief, version_1, critica_2, ciclo=2, mode="offline")
revision_final = criticar_guion(pedido, brief, version_2, ciclo=3, mode="offline")

for etiqueta, revision in [
    ("Crítica 1", critica_1), ("Crítica 2", critica_2), ("Revisión final", revision_final),
]:
    print(etiqueta, "aprobada:", revision.aprobada)
    print("Observaciones:", revision.observaciones)

# %%
print("Primera viñeta, borrador:", borrador.vinetas[0].descripcion)
print("Tras el ciclo 1:", version_1.vinetas[0].descripcion)
print("Tras el ciclo 2:", version_2.vinetas[0].descripcion)

# %% [markdown]
# **Ejercicio 2 — comparar versiones.**
# Completa `contar_cambios_practica`: cuenta cuántas descripciones cambiaron entre
# dos versiones de igual longitud. `zip(lista_a, lista_b)` permite recorrer parejas.
# Si sus longitudes difieren, lanza `ValueError`; de lo contrario `zip` ignoraría
# los elementos sobrantes.

# %%
def contar_cambios_practica(antes: list[str], despues: list[str]) -> int:
    # Valida las longitudes y suma las parejas que difieren.
    return 0


print("Dos cambios:", contar_cambios_practica(["A", "B", "C"], ["A", "b", "c"]) == 2)
print("Sin cambios:", contar_cambios_practica(["A"], ["A"]) == 0)
print("Listas vacías:", contar_cambios_practica([], []) == 0)
rechaza_longitudes = False
try:
    contar_cambios_practica(["A"], [])
except ValueError:
    rechaza_longitudes = True
print("Rechaza longitudes diferentes:", rechaza_longitudes)

textos_borrador = []
textos_version_1 = []
textos_version_2 = []
for vineta in borrador.vinetas:
    textos_borrador.append(vineta.descripcion)
for vineta in version_1.vinetas:
    textos_version_1.append(vineta.descripcion)
for vineta in version_2.vinetas:
    textos_version_2.append(vineta.descripcion)
print("Cambios reales del ciclo 1:", contar_cambios_practica(textos_borrador, textos_version_1))
print("Cambios reales del ciclo 2:", contar_cambios_practica(textos_version_1, textos_version_2))

# %% [markdown]
# <details><summary>Ejercicio 2: cambios observables</summary>
#
# ```python
# def contar_cambios_practica(antes: list[str], despues: list[str]) -> int:
#     if len(antes) != len(despues):
#         raise ValueError("Las versiones deben tener la misma longitud")
#     cambios = 0
#     for texto_antes, texto_despues in zip(antes, despues):
#         if texto_antes != texto_despues:
#             cambios += 1
#     return cambios
# assert contar_cambios_practica(["A", "B", "C"], ["A", "b", "c"]) == 2
# assert contar_cambios_practica(["A"], ["A"]) == 0
# assert contar_cambios_practica([], []) == 0
# assert contar_cambios_practica(textos_borrador, textos_version_1) == pedido.max_vinetas
# assert contar_cambios_practica(textos_version_1, textos_version_2) == pedido.max_vinetas
# try:
#     contar_cambios_practica(["A"], [])
# except ValueError:
#     pass
# else:
#     raise AssertionError("Falta rechazar longitudes diferentes")
# ```
#
# </details>

# %% [markdown]
# ## 4. Contratos cumplidos y referencias inventadas
#
# Un ID inexistente puede tener el tipo correcto, `str`, y pasar Pydantic.
# Necesitamos una comprobación adicional contra el conjunto de fuentes permitido.
# Conservamos el original y alteramos una copia serializada para investigar.

# %%
alterado_dict = version_2.model_dump()
alterado_dict["vinetas"][0]["fuentes"] = ["FUENTE-INVENTADA"]
alterado = GuionComic.model_validate(alterado_dict)
errores_referencia = validar_guion(pedido, brief, alterado)
print("El contrato acepta la forma:", isinstance(alterado, GuionComic))
print("El control del proyecto detecta:", errores_referencia)
print("El original sigue válido:", validar_guion(pedido, brief, version_2) == [])

# %% [markdown]
# **Ejercicio 3 — reparar la referencia.**
# Copia `alterado_dict` mediante `GuionComic.model_validate(...).model_dump()` para
# obtener listas independientes. Reemplaza el ID inventado por las fuentes de la
# primera viñeta original. Valida de nuevo y verifica que el original no cambió.
# Explica por qué Pydantic aceptó el ID inventado (es un `str` válido) y qué
# comprobación lo detectó.

# %%
reparado_dict = GuionComic.model_validate(alterado_dict).model_dump()
# Reemplaza reparado_dict["vinetas"][0]["fuentes"] con una lista de IDs permitidos.
reparado = GuionComic.model_validate(reparado_dict)
print("Referencias reparadas:", validar_guion(pedido, brief, reparado) == [])
print("Original conservado:", version_2.vinetas[0].fuentes != ["FUENTE-INVENTADA"])

# %% [markdown]
# <details><summary>Ejercicio 3: reparar datos sin alterar el original</summary>
#
# ```python
# reparado_dict = GuionComic.model_validate(alterado_dict).model_dump()
# reparado_dict["vinetas"][0]["fuentes"] = list(version_2.vinetas[0].fuentes)
# reparado = GuionComic.model_validate(reparado_dict)
# assert validar_guion(pedido, brief, reparado) == []
# assert version_2.vinetas[0].fuentes != ["FUENTE-INVENTADA"]
# ```
#
# </details>

# %% [markdown]
# ## 5. Un flujo completo y una traza observable
#
# `generar_comic` reúne las fases anteriores. Es un workflow con rutas y dos ciclos
# acotados; el crítico produce datos, pero no elige herramientas ni un número libre
# de pasos. Las fuentes se eligen por héroe; en el bloque 02 viste cómo podrían
# elegirse por significado con embeddings.
#
# | Fase | Rol configurado | Invocaciones live |
# |---|---|---|
# | Brief | default | 1 |
# | Borrador | agent | 1 |
# | Crítica y reescritura, ciclo 1 | default + agent | 2 |
# | Crítica y reescritura, ciclo 2 | default + agent | 2 |
# | Revisión final | default | 1 |
#
# El máximo son siete invocaciones lógicas. Si una fase falla, el flujo se detiene
# y el adaptador no reintenta.

# %%
resultado = generar_comic(pedido, mode="offline")
for evento in resultado.eventos:
    print(evento.etapa, "ciclo", evento.ciclo, "origen", evento.origen, "rol", evento.rol)
print("Estado:", resultado.estado)
print("Invocaciones reales:", resultado.llamadas_modelo)
print("Tokens:", resultado.uso_tokens.model_dump())

# %% [markdown]
# ## 6. Ejecutar con los modelos del `.env`
#
# **Preparación docente:** ejecuta el laboratorio real en otra copia del notebook
# y conserva pedido, fecha, versiones y consumo. Puedes mostrar esa evidencia o
# realizar una demostración en otro kernel mientras el grupo trabaja con datos
# offline etiquetados. Revisa también los fallos observados. No cambies
# silenciosamente una ejecución live fallida por una simulación.
#
# Cambia la bandera únicamente para ejecutar el laboratorio real. Completa la
# clave en `.env`, reinicia el kernel y ejecuta esta celda una vez. El modo real
# carga los nombres por rol desde la configuración; no pegues claves aquí.
#
# Esta celda hace hasta siete llamadas. Las fases anteriores se ejecutaron offline.
# Si el proveedor falla, el error se propaga: identifica la fase antes de reintentar.
# `requiere_revision` es un resultado útil, no un permiso para publicar el guion.
# El consumo de tokens viene del proveedor; `None` indica un dato no disponible.

# %%
EJECUTAR_MODELOS_REALES = False
resultado_real = None

if EJECUTAR_MODELOS_REALES:
    try:
        resultado_real = generar_comic(pedido, mode="live")
    except ConfiguracionComicError:
        print("Falta configuración: revisa .env y reinicia el kernel.")
        raise
    except (ProveedorComicError, SalidaComicInvalida) as error:
        print("Fase que falló:", error.etapa, "ciclo:", error.ciclo)
        raise
    print("Estado real:", resultado_real.estado)
    print(resultado_real.guion.model_dump_json(indent=2))
    print("Invocaciones:", resultado_real.llamadas_modelo)
    print("Consumo observado:", resultado_real.uso_tokens.model_dump())
    for indice, version in enumerate(resultado_real.versiones):
        print("Versión", indice, "primera viñeta:", version.vinetas[0].descripcion)
    for evento in resultado_real.eventos:
        print(evento.etapa, evento.modelo, evento.uso)
else:
    print("Laboratorio real pendiente: bandera desactivada. No se llamó a un LLM.")

# %% [markdown]
# **Ejercicio 4 — evaluar un pedido nuevo.**
# Construye un `PedidoComic` sobre una biblioteca barrial con Batman, tono misterio
# y tres viñetas. Ejecuta `generar_comic(..., mode="offline")`. Comprueba longitud,
# referencias y traza. Lee el guion y escribe una limitación que no detecten esas
# comprobaciones. Si habilitas live para este caso, son otras siete llamadas.

# %%
pedido_nuevo = None
resultado_nuevo = None
# Reemplaza ambos None con un pedido propio y su resultado offline.
if resultado_nuevo is not None and pedido_nuevo is not None:
    print("Tres viñetas:", len(resultado_nuevo.guion.vinetas) == 3)
    print("Referencias válidas:", validar_guion(pedido_nuevo, resultado_nuevo.brief, resultado_nuevo.guion) == [])
    print("Dos ciclos y revisión final:", [r.ciclo for r in resultado_nuevo.revisiones] == [1, 2, 3])
else:
    print("Ejercicio pendiente: crea y ejecuta tu pedido.")

# %% [markdown]
# <details><summary>Ejercicio 4: transferencia a otro pedido</summary>
#
# ```python
# pedido_nuevo = PedidoComic(
#     tema="Investigar por qué desapareció el catálogo de una biblioteca barrial.",
#     heroes=["batman"], tono="misterio", max_vinetas=3,
# )
# resultado_nuevo = generar_comic(pedido_nuevo, mode="offline")
# assert len(resultado_nuevo.guion.vinetas) == 3
# assert validar_guion(pedido_nuevo, resultado_nuevo.brief, resultado_nuevo.guion) == []
# assert [r.ciclo for r in resultado_nuevo.revisiones] == [1, 2, 3]
# # Limitación: los controles no detectan si el misterio resulta interesante.
# ```
#
# </details>

# %% [markdown]
# ## 7. Salida del bloque
#
# En parejas, expliquen: ¿qué contiene el brief?, ¿qué cambió en cada ciclo?, ¿qué
# detecta Python que no detecta Pydantic?, ¿qué evidencia necesitarían para decir
# que un guion mejoró? Guarden sus respuestas junto al pedido y el resultado.
# Continúen con [04 · FastAPI](04_fastapi_comics.ipynb).
#
# ## Ampliación opcional: comparar críticos
#
# Pide a un crítico Astra evaluar **el mismo** guion final con los mismos criterios.
# Es una invocación adicional independiente. No basta elegir el modelo más capaz:
# compara observaciones concretas, falsos positivos, latencia y consumo.
# La función de una fase devuelve la revisión; el total de uso de `resultado_real`
# pertenece al workflow completo y no incluye esta comparación.

# %%
COMPARAR_CRITICO_ASTRA = False
if COMPARAR_CRITICO_ASTRA:
    if resultado_real is None:
        raise ValueError("Ejecuta primero el workflow real para comparar el mismo guion.")
    critico_astra = chat_model("default", model="gpt-6-astra", max_retries=0)
    revision_astra = criticar_guion(
        pedido, resultado_real.brief, resultado_real.guion, ciclo=3, mode="live",
        modelos={"default": critico_astra},
    )
    print("Crítico alternativo:", critico_astra.model_name)
    print(revision_astra.model_dump_json(indent=2))
else:
    print("Comparación opcional desactivada.")

# %% [markdown]
# Referencia: [salidas estructuradas de OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs).
# Lee cómo reportar negativas y respuestas incompletas; el esquema no elimina
# esas situaciones. Los modelos y sus roles se centralizan en
# [la configuración del repo](../../docs/MODELOS.md).
