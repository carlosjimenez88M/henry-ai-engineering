# %% [markdown]
# # Datos y contratos para una aplicación de cómics con IA
#
# **Nivel:** principiante. Continúa después de los fundamentos de Python.
#
# En este bloque prepararemos la entrada que después recibirá una API y un
# modelo de IA. Trabajaremos con pedidos de escenas originales que mencionan
# a Batman, Spider-Man o los Cuatro Fantásticos. Son ejercicios ficticios;
# no describen hechos del canon ni reproducen diálogos publicados.
#
# **Producto del bloque:** un `pedido_validado` y su representación JSON,
# guardada y recuperada de un archivo temporal. No hay llamadas a modelos.
#
# Al terminar podrás comprobar cuatro resultados observables:
#
# 1. Filtrar y limpiar una lista de diccionarios sin modificar los originales.
# 2. Distinguir un diccionario de una cadena JSON y recuperar sus datos.
# 3. Manejar un error esperado de conversión y otro de JSON con excepciones específicas.
# 4. Aceptar un pedido válido y rechazar un héroe, un campo o una cantidad fuera del contrato.
#
# | Orden | Trabajo, con práctica incluida |
# |---|---|
# | 1 | Imports y lectura del caso |
# | 2 | Listas, diccionarios y funciones: ejercicio 1 |
# | 3 | JSON y archivos: ejercicio 2 |
# | 4 | Errores esperados: ejercicio 3 |
# | 5 | Tipos y contrato Pydantic: ejercicio 4 |
# | 6 | Integración y comprobación de salida |
#
# Los ejercicios muestran comprobaciones `True` o `False`. Un `False` indica
# una tarea pendiente y no detiene **Run All**. Intenta resolver antes de abrir
# cada solución. Los ejemplos posteriores usan datos propios para que ejecutar
# todo siga siendo posible mientras practicas.
#
# > **Límites de este bloque.** Un contrato valida la *forma* de un dato: campos,
# > tipos y rangos. No comprueba que el contenido sea verdadero, original o de
# > buena calidad, ni que un modelo respete después lo pedido. Esas comprobaciones
# > pertenecen a otras etapas.

# %% [markdown]
# ## 1. Imports y módulos
#
# Un **módulo** es una unidad de código que podemos reutilizar mediante `import`.
# `json` viene con Python: convierte entre datos de Python y texto JSON.
# `pathlib` y `tempfile` también vienen con Python. Pydantic es una dependencia
# del proyecto, instalada con su entorno.
#
# `import json` conserva el nombre del módulo: llamaremos a `json.dumps(...)`.
# `from pathlib import Path` importa un nombre concreto: usaremos `Path(...)`.

# %%
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

print("Módulos preparados: json, pathlib, tempfile y pydantic.")

# %% [markdown]
# ## 2. Una función que prepara datos
#
# Recibimos tres borradores. Cada diccionario representa una entrada distinta.
# Queremos preparar solamente los borradores activos con un tema útil. Los IDs
# nos permiten reconocer qué entrada cambió durante la preparación.
#
# **Antes de ejecutar:** ¿por qué quitar espacios de un tema no convierte una
# entrada desactivada en una entrada válida para este lote?
#
# En los textos usamos nombres legibles. Los datos que enviará la aplicación
# usan IDs cortos y estables; así una etiqueta de pantalla no cambia el contrato.
#
# | ID en los datos | Nombre legible |
# |---|---|
# | `batman` | Batman |
# | `spiderman` | Spider-Man |
# | `fantasticos` | Cuatro Fantásticos |

# %%
borradores = [
    {"id": "ESC-01", "tema": "  Batman organiza una biblioteca comunitaria.  ", "activo": True},
    {"id": "ESC-02", "tema": "   ", "activo": True},
    {"id": "ESC-03", "tema": "Spider-Man prepara una feria de ciencias.", "activo": False},
]


def preparar_borradores(entradas: list[dict]) -> list[dict]:
    resultado = []
    for entrada in entradas:
        tema_limpio = entrada["tema"].strip()
        if entrada["activo"] and tema_limpio:
            resultado.append({"id": entrada["id"], "tema": tema_limpio})
    return resultado


preparados = preparar_borradores(borradores)
print(preparados)
print("Origen conservado:", borradores[0]["tema"].startswith("  "))

# %% [markdown]
# `list[dict]` y `-> list[dict]` son **anotaciones de tipos**: describen lo que
# esperamos recibir y devolver. Ayudan a una persona o a un editor a entender
# la función. Por sí solas no inspeccionan ni rechazan los datos al ejecutarla:
# más adelante pondremos una validación explícita en la entrada.
#
# Creamos diccionarios nuevos dentro de `resultado`. Así limpiamos los temas
# sin reemplazar el texto original. La condición exige dos cosas a la vez:
# `activo` verdadero y texto no vacío después de limpiar.
#
# ### Ejercicio 1 · Seleccionar y limpiar temas
#
# Completa `seleccionar_temas_practica`: conserva solo las entradas activas,
# elimina espacios de los extremos y devuelve una lista de temas. Debe servir
# para el lote siguiente, no depender de los IDs del ejemplo anterior.
#
# **Pista:** usa un ciclo, `.strip()`, una condición con `and` y `.append(...)`.

# %%
lote_practica = [
    {"id": "NUEVA-01", "tema": "  Los Cuatro Fantásticos reparan un planetario.  ", "activo": True},
    {"id": "NUEVA-02", "tema": "Batman prepara una exposición.", "activo": False},
    {"id": "NUEVA-03", "tema": "  ", "activo": True},
]


def seleccionar_temas_practica(entradas: list[dict]) -> list[str]:
    temas = []
    # Completa aquí el ciclo y la condición.
    return temas


temas_practica = seleccionar_temas_practica(lote_practica)
print("Selección correcta:", temas_practica == ["Los Cuatro Fantásticos reparan un planetario."])
print("Lote vacío correcto:", seleccionar_temas_practica([]) == [])
print("Original conservado:", lote_practica[0]["tema"].startswith("  "))

# %% [markdown]
# <details>
# <summary>Solución del ejercicio 1</summary>
#
# ```python
# def seleccionar_temas_practica(entradas: list[dict]) -> list[str]:
#     temas = []
#     for entrada in entradas:
#         tema = entrada["tema"].strip()
#         if entrada["activo"] and tema:
#             temas.append(tema)
#     return temas
#
# print(seleccionar_temas_practica(lote_practica))
# ```
#
# Comprueba también el lote vacío y que los originales conservaron sus espacios.
# </details>

# %% [markdown]
# ## 3. Un diccionario no es una cadena JSON
#
# Un diccionario es un valor que Python puede consultar mediante claves.
# JSON es un formato de texto usado para intercambiar datos, por ejemplo en el
# cuerpo de una petición HTTP. `json.dumps(...)` serializa datos a una cadena;
# `json.loads(...)` interpreta una cadena JSON y devuelve datos de Python.
#
# `ensure_ascii=False` conserva los acentos legibles. `indent=2` añade sangría
# para leer el texto. Cambian la representación, no el significado del pedido.

# %%
pedido_dict = {
    "tema": "Preparar una feria de ciencias en el barrio.",
    "heroes": ["spiderman", "fantasticos"],
    "max_vinetas": 4,
}

pedido_json = json.dumps(pedido_dict, ensure_ascii=False, indent=2)
pedido_recuperado = json.loads(pedido_json)

print("Antes:", type(pedido_dict).__name__)
print("Durante el transporte:", type(pedido_json).__name__)
print("Después:", type(pedido_recuperado).__name__)
print(pedido_json)
print("Mismos datos:", pedido_recuperado == pedido_dict)

# %% [markdown]
# En Python escribimos `True`, `False` y `None`; en JSON aparecen como `true`,
# `false` y `null`. JSON exige comillas dobles para claves y cadenas.
# Evita preparar JSON a mano pegando comillas: deja la serialización a `json`.
#
# `Path` representa una ruta. `write_text` escribe una cadena y `read_text`
# la recupera. `encoding="utf-8"` permite guardar y leer caracteres como `á`.
# `TemporaryDirectory` crea una carpeta temporal; al salir de `with`, se limpia.
# Esto permite practicar sin depender de la carpeta donde abriste Jupyter.

# %%
with TemporaryDirectory() as carpeta_temporal:
    ruta_pedido = Path(carpeta_temporal) / "pedido.json"
    ruta_pedido.write_text(pedido_json, encoding="utf-8")
    texto_archivo = ruta_pedido.read_text(encoding="utf-8")
    pedido_desde_archivo = json.loads(texto_archivo)
    print("Archivo creado:", ruta_pedido.is_file())
    print("Datos recuperados:", pedido_desde_archivo == pedido_dict)

print("Archivo temporal eliminado:", not ruta_pedido.exists())

# %% [markdown]
# `with` delimita el tiempo de uso de este recurso. La indentación indica qué
# instrucciones se ejecutan mientras la carpeta existe. Una ruta puede seguir
# guardada en una variable aunque el archivo ya haya sido eliminado.
#
# ### Ejercicio 2 · Serializar y recuperar un pedido
#
# Serializa `pedido_practica` a JSON y recupera el diccionario. Conserva héroes,
# tema con acentos y cantidad. Corrige ambas asignaciones de la celda siguiente.
#
# **Pista:** `dumps` produce texto; `loads` lee ese texto. `str(diccionario)`
# no garantiza JSON válido.

# %%
pedido_practica = {
    "tema": "Organizar una exposición de astronomía.",
    "heroes": ["batman"],
    "max_vinetas": 3,
}
texto_practica = "{}"  # Sustituye con la serialización del pedido.
recuperado_practica = {}  # Sustituye con la lectura del texto JSON.

print("Es texto:", isinstance(texto_practica, str))
print("Es un diccionario:", isinstance(recuperado_practica, dict))
print("Roundtrip completo:", recuperado_practica == pedido_practica)
print("Acento legible:", "astronomía" in texto_practica)

# %% [markdown]
# <details>
# <summary>Solución del ejercicio 2</summary>
#
# ```python
# texto_practica = json.dumps(pedido_practica, ensure_ascii=False, indent=2)
# recuperado_practica = json.loads(texto_practica)
# print(recuperado_practica == pedido_practica)
# print("astronomía" in texto_practica)
# ```
#
# El resultado recuperado es un diccionario, no una cadena con comillas.
# </details>

# %% [markdown]
# ## 4. Errores que esperamos y podemos explicar
#
# Una entrada externa puede tener un número inválido o JSON incompleto.
# `try` delimita la operación que puede fallar. `except` recibe un tipo de
# excepción esperado y define qué devolver o explicar. No cubras toda la clase
# con una captura general: perderías información sobre errores de programación.
#
# En el ejemplo de conversión, `int("cuatro")` produce `ValueError`.
# Una cadena JSON con una coma final produce `json.JSONDecodeError`.
# Vamos a manejar ambos errores cerca de su operación, con mensajes distintos.

# %%
numero_recibido = "cuatro"
try:
    cantidad_convertida = int(numero_recibido)
    print("Cantidad convertida:", cantidad_convertida)
except ValueError:
    print("Cantidad inválida: escribe un número entero, por ejemplo 4.")

json_incompleto = '{"heroes": ["batman"],}'
try:
    datos_decodificados = json.loads(json_incompleto)
    print(datos_decodificados)
except json.JSONDecodeError as error_json:
    print("JSON inválido, línea:", error_json.lineno, "columna:", error_json.colno)

# %% [markdown]
# Poder convertir `"4"` a `4` es una decisión explícita del programa.
# Más adelante el contrato de nuestra API exigirá un entero y rechazará `"4"`:
# así diferenciamos ayuda de conversión de una regla de entrada estricta.
#
# Leer JSON sin error tampoco garantiza obtener un diccionario: `json.loads`
# puede devolver una lista, una cadena o un número. Usaremos una condición para
# distinguir **sintaxis válida** de **forma esperada**.
#
# ### Ejercicio 3 · Leer un objeto JSON
#
# Completa `leer_objeto_practica`. Debe devolver un diccionario con `ok` y `datos`.
# Si el JSON tiene sintaxis incorrecta o su resultado no es un diccionario,
# devuelve `{"ok": False, "datos": None}`. Conserva los datos cuando sí es un objeto.
#
# **Pista:** dentro de `try`, usa `json.loads`; después revisa `isinstance(datos, dict)`.
# Captura únicamente `json.JSONDecodeError`. Las entradas de este ejercicio son cadenas.

# %%
def leer_objeto_practica(texto: str) -> dict:
    # Sustituye este retorno por la lectura y sus comprobaciones.
    return {"ok": False, "datos": None}


objeto_practica = leer_objeto_practica('{"heroes": ["spiderman"], "max_vinetas": 4}')
json_malo_practica = leer_objeto_practica('{"heroes": ["spiderman"],}')
lista_practica = leer_objeto_practica('["batman"]')

print("Objeto aceptado:", objeto_practica == {"ok": True, "datos": {"heroes": ["spiderman"], "max_vinetas": 4}})
print("Sintaxis inválida manejada:", json_malo_practica == {"ok": False, "datos": None})
print("Lista rechazada:", lista_practica == {"ok": False, "datos": None})

# %% [markdown]
# <details>
# <summary>Solución del ejercicio 3</summary>
#
# ```python
# def leer_objeto_practica(texto: str) -> dict:
#     try:
#         datos = json.loads(texto)
#     except json.JSONDecodeError:
#         return {"ok": False, "datos": None}
#     if not isinstance(datos, dict):
#         return {"ok": False, "datos": None}
#     return {"ok": True, "datos": datos}
#
# print(leer_objeto_practica('{"heroes": ["spiderman"]}'))
# print(leer_objeto_practica('{"heroes": ["spiderman"],}'))
# print(leer_objeto_practica('["batman"]'))
# ```
#
# El primer resultado es `ok=True`; los otros dos son `ok=False` por causas diferentes.
# </details>

# %% [markdown]
# ## 5. Anotar tipos y validar entradas son operaciones diferentes
#
# Esta función declara que recibe un entero. Observa qué ocurre al pasarle
# texto: Python ejecuta la multiplicación que ese valor permite. La anotación
# no añadió una comprobación. Esta diferencia importa cuando una entrada llega
# de otro programa o de un modelo de IA.

# %%
def duplicar_cantidad(cantidad: int) -> int:
    return cantidad * 2


print("Entero:", duplicar_cantidad(3))
print("Texto sin validación:", duplicar_cantidad("3"))

# %% [markdown]
# Pydantic sí ejecuta validación cuando construimos un modelo con sus datos.
# Una clase agrupa una descripción de campos. `BaseModel` proporciona la
# validación; `Field` agrega restricciones. Por ahora, concéntrate en leerlas
# y probar entradas: no necesitas conocer toda la programación con clases.
#
# Nuestro contrato local tiene tres campos obligatorios y un campo con valor
# predeterminado. Usaremos estos mismos IDs y campos en la API del taller:
#
# | Campo | Entrada permitida | Motivo |
# |---|---|---|
# | `tema` | Texto de 5–500 caracteres, después de quitar espacios extremos | Limitar el tamaño de la instrucción |
# | `heroes` | Lista de 1–3 IDs de la selección | Mantener un vocabulario conocido |
# | `max_vinetas` | Entero de 2–6; no cadena, decimal ni booleano | Limitar la cantidad solicitada |
# | `tono` | `aventura`, `misterio` o `humor`; por defecto `aventura` | Elegir una orientación para la escena |
#
# `Literal` enumera opciones permitidas. `strict=True` impide convertir `"4"`
# o `True` en un entero. `extra="forbid"` rechaza campos desconocidos y ayuda
# a detectar claves mal escritas. `str_strip_whitespace=True` quita espacios
# extremos de los textos antes de comprobar sus restricciones.

# %%
class PedidoEscena(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tema: str = Field(min_length=5, max_length=500)
    heroes: list[Literal["batman", "spiderman", "fantasticos"]] = Field(min_length=1, max_length=3)
    max_vinetas: int = Field(strict=True, ge=2, le=6)
    tono: Literal["aventura", "misterio", "humor"] = "aventura"


pedido_validado = PedidoEscena.model_validate(pedido_dict)
print("Tema validado:", pedido_validado.tema)
print("Máximo de viñetas:", pedido_validado.max_vinetas)
print("Diccionario exportado:", pedido_validado.model_dump())

# %% [markdown]
# `.model_validate(...)` recibe datos y construye un objeto `PedidoEscena` si
# cumplen el contrato. Consultamos sus campos con un punto: `pedido_validado.tema`.
# `.model_dump()` vuelve a un diccionario de Python para guardarlo o transportarlo.
# El campo `tono` aparece con `"aventura"` aunque el diccionario de entrada
# no lo incluyera: le dimos ese valor predeterminado en la clase.
#
# Esta clase local permite estudiar el contrato sin entrar todavía al backend.
# La API usará el contrato del módulo del proyecto como su fuente central. En
# una aplicación mantendríamos esa definición común en vez de editar dos copias.
#
# ¿Qué rechazará esta variante? Cambiamos solo la cantidad para poder identificar
# el error. No imprimimos el pedido completo dentro del manejo del error.

# %%
pedido_cantidad_texto = {**pedido_dict, "max_vinetas": "4"}
try:
    PedidoEscena.model_validate(pedido_cantidad_texto)
except ValidationError as error_validacion:
    for detalle in error_validacion.errors():
        print("Campo:", detalle["loc"], "tipo de error:", detalle["type"])

# %% [markdown]
# `{**pedido_dict, "max_vinetas": "4"}` crea un diccionario nuevo con las mismas
# claves y reemplaza una de ellas. El original conserva el entero `4`.
#
# Recuerda el recuadro del inicio: un tema puede estar bien formado y pedir una
# escena incoherente.
#
# ### Ejercicio 4 · Corregir un pedido inválido
#
# Corrige `pedido_corregido`: elige un héroe permitido, reduce la cantidad al
# rango y elimina el campo desconocido. Usa `pop("clave")` para quitar un campo.
# Mantén un tema de al menos cinco caracteres.
#
# **Pista:** cambia tres cosas. Luego explica por qué cambiar solo `"9"` a `9`
# todavía dejaría errores. La comprobación mostrará `False` mientras sea inválido.

# %%
pedido_por_corregir = {
    "tema": "Diseñar una actividad de ciencia para un museo.",
    "heroes": ["superman"],
    "max_vinetas": "9",
    "modo": "live",
}
pedido_corregido = pedido_por_corregir.copy()
# Corrige aquí los valores y elimina el campo que no pertenece al contrato.

try:
    validado_practica = PedidoEscena.model_validate(pedido_corregido)
    print("Contrato satisfecho:", True)
    print("Original conservado:", pedido_por_corregir["max_vinetas"] == "9")
except ValidationError as error_practica:
    print("Contrato satisfecho:", False)
    print("Campos pendientes:", [detalle["loc"] for detalle in error_practica.errors()])

# %% [markdown]
# <details>
# <summary>Solución del ejercicio 4</summary>
#
# ```python
# pedido_corregido = pedido_por_corregir.copy()
# pedido_corregido["heroes"] = ["batman"]
# pedido_corregido["max_vinetas"] = 4
# pedido_corregido.pop("modo")
# validado_practica = PedidoEscena.model_validate(pedido_corregido)
# print(validado_practica.model_dump())
# ```
#
# `9` sigue fuera del máximo de seis; `superman` sigue fuera de esta selección;
# `modo` sigue siendo un campo desconocido: el servidor decidirá cómo ejecutar
# su trabajo, no quien envía este pedido. `copy()` crea una copia
# superficial: al reemplazar campos no cambiamos el original. Si modificaras
# directamente la lista interior, ambas copias compartirían esa misma lista.
# </details>

# %% [markdown]
# Comprobemos ahora casos nuevos. Cada error se maneja y la ejecución puede
# continuar hacia el siguiente caso.

# %%
variantes_contrato = [
    ("válido", {**pedido_dict, "max_vinetas": 2}, True),
    ("sin héroes", {**pedido_dict, "heroes": []}, False),
    ("booleano como entero", {**pedido_dict, "max_vinetas": True}, False),
    ("cantidad excesiva", {**pedido_dict, "max_vinetas": 7}, False),
    ("tema vacío", {**pedido_dict, "tema": "   "}, False),
    ("campo inesperado", {**pedido_dict, "modelo": "cualquier-modelo"}, False),
]

for nombre_caso, datos_caso, debe_aceptarse in variantes_contrato:
    try:
        PedidoEscena.model_validate(datos_caso)
        aceptado = True
    except ValidationError:
        aceptado = False
    print(nombre_caso, "comportamiento esperado:", aceptado == debe_aceptarse)

# %% [markdown]
# ## 6. Integración y siguiente paso
#
# Recuperemos el producto del bloque en un archivo temporal. La entrada vuelve
# a pasar por el contrato después de leerla: un archivo también puede haber
# sido modificado por otro programa. Las variables exportadas quedan disponibles
# para inspeccionarlas aunque el archivo de práctica se limpie al final.

# %%
pedido_para_archivo = pedido_validado.model_dump()
with TemporaryDirectory() as carpeta_integracion:
    archivo_integracion = Path(carpeta_integracion) / "pedido_validado.json"
    archivo_integracion.write_text(
        json.dumps(pedido_para_archivo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    datos_archivo_integracion = json.loads(archivo_integracion.read_text(encoding="utf-8"))
    pedido_recuperado_validado = PedidoEscena.model_validate(datos_archivo_integracion)

print("Contrato conservado:", pedido_recuperado_validado == pedido_validado)
print("Contenido listo:", pedido_recuperado_validado.model_dump())
print("Archivo temporal eliminado:", not archivo_integracion.exists())

# %% [markdown]
# Esta tabla prepara la clase de FastAPI. Todavía no hemos creado un servidor
# ni enviado una petición HTTP. Allí verificaremos los estados con peticiones
# reales en el proceso de prueba.
#
# | Situación | Resultado local de este bloque | Estado previsto en FastAPI después |
# |---|---|---|
# | Pedido que cumple el contrato | `PedidoEscena` válido | `200` si el endpoint procesa correctamente |
# | Cantidad `"4"`, héroe no permitido o campo extra | `ValidationError` | `422` al validar el cuerpo de entrada |
# | Texto JSON con sintaxis incorrecta | `JSONDecodeError` al leerlo con `json.loads` | `422` en la validación de la petición |
# | Pedido válido y contenido generado incoherente | Este contrato de entrada no lo detecta | Requiere validar la salida y el comportamiento del sistema |
#
# **Comprobación de salida: explicación y un cambio.**
#
# 1. Explica qué tipo tiene `pedido_json` y por qué no consultamos
#    `pedido_json["heroes"]` como si fuera un diccionario.
# 2. Cambia un héroe por `"fantasticos"` y prueba `max_vinetas=6` y
#    `max_vinetas="6"`. Predice cuál satisface el contrato antes de ejecutarlo.
# 3. Explica por qué un contrato válido no demuestra que una escena sea verdadera
#    o que un modelo haya seguido las instrucciones.
#
# Estás listo para continuar cuando tus cuatro ejercicios muestran sus
# comprobaciones correctas y puedes justificar las dos cantidades del punto 2.
# Sigue con [02 · Embeddings](02_embeddings_visual.ipynb): verás cómo un modelo
# convierte texto en números que se pueden comparar y graficar.
#
# Referencias de consulta: [JSON en Python 3.13](https://docs.python.org/3.13/library/json.html),
# [pathlib](https://docs.python.org/3.13/library/pathlib.html),
# [restricciones de campos Pydantic](https://docs.pydantic.dev/latest/concepts/fields/)
# y [configuración de modelos](https://docs.pydantic.dev/latest/api/config/).
