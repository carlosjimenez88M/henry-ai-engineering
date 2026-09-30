# %% [markdown]
# # Clase 1 · El archivo de la Baticueva: construir una herramienta confiable
#
# **Duración total: 120 minutos, incluidas dos pausas.**
# Al terminar vas a poder diseñar el contrato de una herramienta, validar una consulta,
# buscar evidencia con filtros y explicar quién decide y quién ejecuta una llamada.
#
# **Pregunta de ingeniería:** “Buscá dos fichas de investigación de Batman y devolvé
# sus fuentes”. ¿Qué tendría que devolver el programa para que otro sistema pueda
# usarlo sin adivinar qué significa su respuesta?
#
# Una herramienta potente no es una función que hace cualquier cosa. Es una capacidad
# bien delimitada, reutilizable y fácil de comprobar. Hoy construimos esa capacidad.
#
# | Minutos | Trabajo y producto observable |
# |---|---|
# | 0–8 | Distinguir respuesta inventada de evidencia; escribir una predicción |
# | 8–18 | Ver la herramienta terminada; identificar entrada, salida y límites |
# | 18–30 | Construir una búsqueda mínima; explicar un bucle y una coincidencia |
# | 30–40 | Modificar el filtro y detectar un falso positivo |
# | 40–45 | Pausa |
# | 45–55 | Definir un contrato con tipos y validación |
# | 55–70 | Recorrer filtros, ranking y resultados; comprobar invariantes |
# | 70–80 | Exponer la función como herramienta y observar un tool call |
# | 80–85 | Pausa |
# | 85–100 | Reto: una recomendación musical con evidencia |
# | 100–112 | Comparar solución y probar tres fallas |
# | 112–120 | Ticket de salida y decisión de diseño |
#
# **Cómo trabajar:** anticipá una salida, ejecutá una celda, observá y explicá.
# La persona que ejecuta y la que revisa intercambian roles durante el reto.
# No hay carrera: el criterio es explicar el resultado. Si te perdés, usá el punto
# de reenganche más cercano. Las soluciones están después de los retos para poder
# volver a ejecutar todo sin dejar celdas rotas.
#
# **Contexto:** las fichas de Batman, los Cuatro Fantásticos y El Chavo son escenarios
# inventados para aprender ingeniería, no resúmenes de cómics o episodios reales.
# Las canciones son inventadas y solo tienen metadatos: no contienen letras ni audio.
# No necesitás conocer estos personajes para resolver las actividades.
#
# **Dos modos:** offline usa búsqueda real y grafos reales, pero sustituye al LLM por
# reglas/extractos explícitos. Live usa OpenAI y consume API. El estudiante puede
# hacer toda la práctica offline; el docente demuestra live. No compartas el .env.

# %% [markdown]
# ## 0–8 · Problema antes de código
# Una persona pide “la mejor historia de Batman sobre cooperación”. El modelo puede
# escribir algo convincente, pero nuestro archivo tiene solo doce fichas. La respuesta
# debe decir qué encontró allí, no fingir haber leído toda una colección comercial.
#
# **Escribí antes de ejecutar:** ¿qué debería pasar si pido una ficha que no existe?
# A) Inventarla. B) Devolver una lista vacía con un estado claro. C) Repetir para siempre.
# Explicá por qué la opción B permite que otro programa tome una decisión.

# %%
from henry_agents.config import chat_model, configure
from henry_agents.cultural import load_catalog, search_catalog

MODE = configure()
catalogo = load_catalog()
print("Modo:", MODE, "| fichas disponibles:", len(catalogo))

# %% [markdown]
# ## 8–18 · Primero vemos el contrato en acción
# Mirá solo tres campos: query es la pregunta; hits son resultados; status informa
# si hubo evidencia. Cada hit tiene ID, colección, texto y score. Un ID nos permite
# citar una fuente sin copiar todo el documento.
#
# **Predicción:** con universe="batman" no debería aparecer ningún resultado MUS.
# Ejecutá y comprobalo leyendo los IDs. No hace falta memorizar los nombres de campos.

# %%
ejemplo = search_catalog("investigación", universe="batman", top_k=2)
print("Estado:", ejemplo.status)
for ficha in ejemplo.hits:
    print(ficha.id, ficha.title)
print("Fichas examinadas después del filtro:", ejemplo.inspected)

# %% [markdown]
# **Discutimos:** ¿por qué usar top_k? Limita lo que vuelve al modelo y hace más
# predecible el tamaño del contexto. top_k no hace que una respuesta sea correcta.
# Un score lexical tampoco es “porcentaje de verdad”.
#
# ## 18–30 · Una función mínima que sí entendemos
# Primero trabajamos con diccionarios. Cada ficha es una fila con campos con nombre.
# `for` recorre filas; `if` decide si conservar una; `append` agrega a una lista.
# Todavía no usamos modelos ni grafos. Python alcanza para buscar coincidencias.

# %%
primera = catalogo[0]
print("ID:", primera["id"])
print("Colección:", primera["universe"])
print("Temas:", primera["tags"])
print("Texto:", primera["text"])


# %%
def buscar_minimo(palabra, fichas):
    encontradas = []
    for ficha in fichas:
        texto = " ".join(ficha["tags"])
        if palabra in texto:
            encontradas.append(ficha["id"])
    return encontradas


print(buscar_minimo("investigacion", catalogo))

# %% [markdown]
# **Explicá con una fila:** ¿qué valor tiene palabra? ¿Qué texto se compara?
# ¿Por qué aparece una canción al buscar investigación en todo el archivo?
# La búsqueda encuentra una coincidencia, no sabe qué colección queríamos.
#
# ## 30–40 · Modificación y un error que conviene descubrir
# Cambiá solo palabra por "equipo". Anotá si esperás Batman, Fantásticos, Chavo o
# canciones. Después probá "investigación" con tilde: nuestra primera función no
# normaliza texto. El comportamiento incorrecto nos indica qué mejorar.

# %%
palabra = "equipo"  # Después probá investigación, con tilde.
print(buscar_minimo(palabra, catalogo))
print("Sin tilde:", buscar_minimo("investigacion", catalogo))
print("Con tilde:", buscar_minimo("investigación", catalogo))

# %% [markdown]
# **Microejercicio:** en papel o en una celda nueva, agregá una condición para conservar
# solo la colección batman. Pista: ficha["universe"] == "batman".
# No cambies la herramienta terminada todavía: compará primero las dos estrategias.
#
# **Punto de reenganche 1:** podés seguir una fila desde entrada hasta resultado y
# nombrar dos límites de buscar_minimo: no filtra colección ni maneja tildes.
#
# ## 40–45 · Pausa
#
# ## 45–55 · Diseñar el contrato antes de agregar más poder
# La entrada de una herramienta también es una interfaz. Debe aceptar entradas útiles
# y rechazar las que no puede manejar. Pydantic nos ayuda a escribir esas reglas.
#
# | Campo | Regla | Problema que evita |
# |---|---|---|
# | query | Texto, 3–240 caracteres, sin espacios vacíos | Consultas vacías o enormes |
# | universe | Una colección de una lista cerrada | Nombres arbitrarios o rutas de archivos |
# | kind | todos, ficha o cancion | Mezclar formatos sin intención |
# | top_k | Entero de 1 a 5 | Pedir resultados ilimitados |
#
# La herramienta no recibe SQL, rutas ni código para ejecutar. No usa eval.
# Las restricciones existen en Python aunque el modelo proponga otra cosa.
# Leé primero una línea: `top_k` debe ser un entero entre 1 y 5. `Field` añade
# esas reglas al tipo. `Literal` enumera las únicas opciones aceptadas. Vamos a
# definir el contrato aquí y usarlo después en nuestra propia herramienta.

# %%
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=3, max_length=240)
    universe: Literal["todos", "batman", "fantasticos", "chavo", "canciones"] = "todos"
    kind: Literal["todos", "ficha", "cancion"] = "todos"
    top_k: int = Field(default=3, ge=1, le=5, strict=True)


# %% [markdown]
# `extra="forbid"` rechaza campos que no declaramos. La segunda opción quita
# espacios al principio y al final. Construir `SearchArgs(...)` ejecuta estas
# reglas: no es solamente un comentario para quien lee el programa.

# %%

entrada = SearchArgs(query="  investigación  ", universe="batman", top_k=2)
print(entrada.model_dump())
try:
    SearchArgs(query="investigación", top_k=50)
except ValidationError:
    print("Rechazado: top_k debe estar entre 1 y 5.")

# %% [markdown]
# **Pregunta:** ¿schema válido significa respuesta correcta? No: "receta" puede ser
# una consulta válida y no tener resultados. Diferenciamos error de entrada de falta
# de evidencia; no devolvemos el mismo mensaje para todo.
#
# ## 55–70 · La tubería de una herramienta potente
# Vamos a recorrer su implementación sin memorizarla:
#
# ```text
# entrada → validar → normalizar → filtrar → puntuar → ordenar → limitar → devolver IDs
# ```
#
# Primero filtramos por colección y tipo. Luego contamos coincidencias con la consulta.
# Ordenamos por score y usamos el ID para desempatar siempre igual. Finalmente top_k
# limita cuántas fichas regresan. Cada fragmento tiene como máximo 600 caracteres.
# No es búsqueda semántica aprendida: los sinónimos no registrados pueden fallar.

# %%
from henry_agents.cultural import query_terms

print("Normalización:", query_terms("Buscá investigación"))
print("Alias explícito:", query_terms("música"))
filtrado = search_catalog("equipo", universe="fantasticos", top_k=1)
print(filtrado.model_dump())

# %% [markdown]
# **Invariante** es una condición que debe mantenerse. Escribimos tres: número de
# resultados acotado, colección correcta e IDs presentes. Son más útiles que comprobar
# una frase exacta que el modelo puede redactar de otra manera.

# %%
assert len(filtrado.hits) <= 1
assert all(f.universe == "fantasticos" for f in filtrado.hits)
assert all(f.id for f in filtrado.hits)
sin_resultados = search_catalog("vacuna marciana")
assert sin_resultados.status == "no_results"
assert sin_resultados.hits == []
print("Filtros, límites y ausencia de evidencia comprobados.")

# %% [markdown]
# **Parada de discusión (3 minutos):** ¿qué perderíamos si devolvemos solo texto?
# Perdemos IDs, criterios del ranking y estado estructurado. ¿Qué riesgo hay si
# permitimos un top_k gigante? Contexto innecesario, más costo y resultados confusos.
#
# ## 70–80 · De función Python a herramienta del modelo
# El decorador @tool aporta nombre, descripción y esquema. La descripción le dice
# al modelo cuándo usarla; la validación impone los límites. La función no se vuelve
# inteligente por tener el decorador.
#
# Construimos el wrapper aquí. La búsqueda reutiliza search_catalog; el contrato
# que escribimos arriba valida los argumentos antes de ejecutarla. El docstring,
# entre comillas triples, explica al modelo para qué sirve la herramienta.

# %%
from langchain_core.tools import tool


@tool(args_schema=SearchArgs)
def buscar_archivo(query: str, universe="todos", kind="todos", top_k=3) -> dict:
    """Busca fichas ficticias con IDs por tema, colección y tipo.

    Usar para recuperar evidencia del catálogo; no busca en Internet.
    Si no encuentra evidencia, devuelve no_results y una lista vacía.
    """
    return search_catalog(query, universe=universe, kind=kind, top_k=top_k).model_dump()


# %%
print("Nombre:", buscar_archivo.name)
print("Campos:", list(buscar_archivo.args_schema.model_fields))
resultado_tool = buscar_archivo.invoke({"query": "investigación", "universe": "batman", "top_k": 2})
print("IDs:", [f["id"] for f in resultado_tool["hits"]])

# %% [markdown]
# El modelo propone un nombre y argumentos; la aplicación ejecuta. Esta demo fuerza
# la herramienta para observar su contrato. **Todavía no demuestra autonomía**:
# la decisión de usarla fue nuestra. En la clase 3 habrá un bucle que decide cuándo terminar.

# %%
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

if MODE == "live":
    modelo = chat_model().bind_tools([buscar_archivo], tool_choice="buscar_archivo")
    propuesta = modelo.invoke(
        [HumanMessage(content="Busca dos fichas de investigación de Batman.")]
    )
else:
    propuesta = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "buscar_archivo",
                "args": {"query": "investigación", "universe": "batman", "top_k": 2},
                "id": "demo-1",
            }
        ],
    )
llamada = propuesta.tool_calls[0]
assert llamada["name"] == buscar_archivo.name
observacion = buscar_archivo.invoke(llamada["args"])
mensaje_tool = ToolMessage(content=json.dumps(observacion), tool_call_id=llamada["id"])
print("Acción:", llamada["name"], llamada["args"])
print("ID que vincula acción y observación:", mensaje_tool.tool_call_id)

# %% [markdown]
# **Punto de reenganche 2:** podés señalar la propuesta, la ejecución Python y la
# observación. Los IDs vinculan mensajes; no son razonamiento privado del modelo.
#
# ## 80–85 · Pausa
#
# ## 85–100 · Taller: DJ de una historia de detectives
# La persona que escribe configura la herramienta; la otra revisa los resultados.
# A mitad del bloque cambien roles.
#
# **Pedido:** encontrar una canción para una actividad de investigación. No queremos
# fichas de Batman en esa lista. Entregar título, ID y criterio usado para elegir.
#
# 1. Escribí la entrada antes de llamar a la herramienta.
# 2. Usá universe y kind para excluir fichas.
# 3. Pedí como máximo dos resultados.
# 4. Probá también una consulta que no tenga evidencia.
#
# **Pista 1:** hay una canción con la etiqueta investigacion.
# **Pista 2:** universe="canciones", kind="cancion".
# **Extra si terminás:** explicá si necesitamos un LLM para este subproblema.
# La selección por filtros ya funciona sin uno.

# %%
entrada_del_reto = {
    "query": "investigación",
    "universe": "canciones",
    "kind": "cancion",
    "top_k": 2,
}
mi_busqueda = buscar_archivo.invoke(entrada_del_reto)
for ficha in mi_busqueda["hits"]:
    print(ficha["id"], ficha["title"])

# %% [markdown]
# ## 100–112 · Solución y pruebas de borde
# La solución no elige por gusto personal: muestra por qué esa ficha aparece en el
# archivo. Si faltan pruebas de error, todavía no sabemos si el contrato se respeta.
# Los errores siguientes son deliberados y se capturan solo en esta demostración.

# %%
solucion = search_catalog("investigación", universe="canciones", kind="cancion", top_k=2)
assert [h.id for h in solucion.hits] == ["MUS-02"]
casos_invalidos = [
    {"query": "  "},
    {"query": "equipo", "top_k": 0},
    {"query": "equipo", "universe": "internet"},
]
for caso in casos_invalidos:
    try:
        SearchArgs(**caso)
        raise AssertionError("Esta entrada debía rechazarse")
    except ValidationError:
        print("Entrada rechazada correctamente:", caso)
print("Resultado respaldado:", solucion.hits[0].text)

# %% [markdown]
# ## 112–120 · Ticket de salida
# Entregá una entrada válida, una inválida y el ID de una fuente. Explicá:
# - ¿Qué valida el código y qué decide el modelo?
# - ¿Qué diferencia hay entre no_results y un error de validación?
# - ¿Qué mejorarías para buscar sinónimos sin romper el contrato?
#
# **Criterio de logro:** podés modificar un filtro y demostrar que se cumple.
# La próxima clase usa esta misma herramienta como pieza de un RAG y un grafo.
# La implementación reutilizable está en src/henry_agents/cultural.py; no es necesario
# leer todo ese módulo para entender esta clase.
