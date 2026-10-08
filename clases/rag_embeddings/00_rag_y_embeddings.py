# %% [markdown]
# # RAG y embeddings: de una pregunta a una respuesta con fuentes
#
# **Para empezar:** no necesitas conocer embeddings, bases vectoriales ni RAG.
# Usaremos un centro cultural ficticio del barrio y un poco de Python preparado.
# Si puedes leer una lista, un diccionario y una llamada a función, puedes seguir
# la práctica. El objetivo es explicar y modificar el flujo, no memorizar librerías.
#
# Al terminar podrás:
# - Explicar qué representa un embedding y comparar textos con dos modelos.
# - Guardar fragmentos en una base vectorial y recuperar los más cercanos.
# - Separar búsqueda, contexto y generación; revisar una respuesta y sus citas.
# - Detectar una fuente archivada y una pregunta que los documentos no contestan.
#
# **Caso:** Camila quiere llevarse un libro de ilustración del centro cultural
# para practicar dibujo en casa. Pregunta: **«¿Qué necesito para llevar un libro y
# cuándo debo devolverlo?»**. Necesita dos datos: qué presentar y cuál es el plazo.
# Están en párrafos distintos; hay además una norma vieja que podría confundirla.
# La información vive en documentos. Buscaremos evidencia para construir la respuesta.
#
# | Necesidad de Camila | Fuente que debe llegar al contexto |
# |---|---|
# | Qué presentar para retirar el libro | CC-01-P2: credencial y documento de identidad |
# | Cuándo devolverlo | CC-01-P1: siete días calendario |
#
# ![Dos recorridos: indexar documentos y consultar para responder](assets/01_flujo_rag.png)
#
# **RAG** significa *Retrieval-Augmented Generation*: generación apoyada por
# información recuperada. Aquí lo llamaremos **buscar para responder**.
# **Embedding** es una lista de números que representa un texto para compararlo.
# **Base vectorial** guarda esos números junto con el texto y sus metadatos.
# El modelo de embeddings y el modelo que redacta respuestas cumplen tareas distintas.
# Este RAG no entrena sus modelos: añade documentos al contexto de cada pregunta.

# %% [markdown]
# ## 1. Abrir los documentos antes de buscar
#
# Abre la clase con `uv run python scripts/start_rag_class.py` desde la raíz.
# Ejecuta con el kernel **Henry AI Engineering (.venv)**, de arriba hacia abajo.
# Esta celda prepara las
# herramientas del laboratorio. No necesitas aprender todos sus imports.
# `True` y `False` son valores lógicos; `texto[0]` selecciona el primer elemento;
# `for` recorre una lista; `funcion(...)` ejecuta una operación.
#
# El notebook usa vectores **calculados por modelos reales** y respuestas reales
# guardadas por el docente. Ejecutar todo con las banderas iniciales no llama a la
# API. En cada respuesta distinguiremos esa reproducción de una generación nueva.

# %%
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from IPython.display import display
from qdrant_client import models

from henry_agents.config import ROOT
from henry_agents.rag_taller import (
    CONSULTAS,
    FRASES,
    MODELOS_EMBEDDING,
    abrir_base,
    buscar,
    cargar_documentos,
    crear_indice,
    fragmentar,
    leer_cache,
    palabras_compartidas,
    responder,
    validar_citas,
    vectores_para,
)
from henry_agents.rag_taller_experimentos import (
    dividir_por_caracteres,
    evaluar_recuperacion,
    fusionar_rankings,
)
from henry_agents.rag_taller_panel import crear_explorador
from henry_agents.rag_taller_replay import leer_respuestas, reproducir_respuesta
from henry_agents.rag_taller_visuales import (
    dibujar_coberturas,
    dibujar_fragmentacion,
    dibujar_mapa_manual,
    dibujar_metricas,
    dibujar_modelos,
    dibujar_proyeccion,
    dibujar_ranking,
    mostrar_figura,
)

documentos = cargar_documentos()
print("Raíz del proyecto:", ROOT)
print("Documentos:", len(documentos))
print("Título:", documentos[0]["titulo"])
print(documentos[0]["texto"])
print("¿Está vigente?:", documentos[0]["vigente"])

# %% [markdown]
# ## 2. Un documento puede contener varias respuestas
#
# **Fragmentar** (*chunking*) significa dividir el contenido en piezas que podamos
# recuperar. Cortar sólo por cantidad de caracteres puede separar una regla de
# su condición. En este corpus pequeño usamos un párrafo completo por fragmento.
#
# El primer párrafo informa el plazo; el segundo, los requisitos. Recuperar sólo
# uno no basta para contestar una pregunta que pide ambas cosas.
# Cada fragmento conserva un ID, título, texto, categoría y vigencia.

# %%
parrafos_del_primero = documentos[0]["texto"].split("\n\n")
print("Párrafos del primer documento:", len(parrafos_del_primero))
fragmentos = fragmentar(documentos)
print("Fragmentos del corpus:", len(fragmentos))
print("ID:", fragmentos[0]["id"])
print("Texto:", fragmentos[0]["texto"])
print("Otra pieza del mismo documento:", fragmentos[1]["id"])

# %% [markdown]
# **Predice:** si la pregunta pide plazo y requisitos, ¿qué dos fragmentos del
# primer documento querrías recuperar? Localiza en el texto las frases de apoyo.
# La división por párrafos es una decisión de esta clase; en documentos extensos
# debemos probar otros tamaños y conservar títulos, tablas y relaciones relevantes.
#
# ## 3. Entender un vector con un mapa inventado
#
# Estos puntos tienen dos coordenadas asignadas a mano. **No son embeddings de un
# modelo**. Sirven para ver qué significa comparar la dirección de dos flechas.
# El modelo real produce muchas coordenadas y sus ejes no tienen nombres humanos
# como «libros» o «cómics».

# %%
mostrar_figura(dibujar_mapa_manual())

# %% [markdown]
# La **similitud coseno** compara direcciones: 1 indica la misma dirección, 0
# direcciones perpendiculares y -1 direcciones opuestas. Es una medida geométrica.
# En embeddings aprendidos la cercanía puede resultar útil para recuperar textos;
# no demuestra que una afirmación sea verdadera ni que el resultado responda bien.
#
# Esta función preparada multiplica coordenadas por parejas, suma y divide por
# los largos de ambos vectores. No necesitas memorizar la fórmula.

# %%
def coseno(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


pregunta_manual = [0.93, 0.72]
print("Con cómics:", round(coseno(pregunta_manual, [0.90, 0.45]), 3))
print("Con libros:", round(coseno(pregunta_manual, [0.10, 1.00]), 3))
print("Misma dirección:", math.isclose(coseno([1, 0], [3, 0]), 1))

# %% [markdown]
# ## 4. Ver embeddings aprendidos, sin inventar sus números
#
# Dos modelos transformarán los **mismos textos**: `text-embedding-3-small` y
# `text-embedding-3-large`. La caché guarda modelo, dimensión, fecha, textos y tokens
# reportados durante su preparación. Los números provienen de la API, no del mapa.
#
# No se comparan directamente un vector Small y uno Large: pertenecen a espacios
# distintos. Incluso dos modelos con la misma dimensión pueden ser incompatibles.
# La pregunta y los documentos de una búsqueda usan el mismo modelo y configuración.

# %%
cache = leer_cache()
SMALL = "text-embedding-3-small"
LARGE = "text-embedding-3-large"
for nombre in MODELOS_EMBEDDING:
    espacio = cache["modelos"][nombre]
    print(nombre, "dimensiones:", espacio["dimensiones"], "origen:", espacio["origen"])
    print("Calculado:", espacio["generado_utc"], "tokens de preparación:", espacio["usage"])

vectores_frases = vectores_para(FRASES, SMALL, mode="offline")
print("Texto:", FRASES[0])
print("Primeros seis números:", vectores_frases[0][:6])
print("Longitud completa:", len(vectores_frases[0]))

# %% [markdown]
# La matriz compara cada frase con las demás usando **todos** sus números.
# La diagonal compara una frase consigo misma. Lee la frase 1 frente a la 2:
# expresan una intención parecida con palabras diferentes.

# %%
etiquetas_frases = ["1: libro a casa", "2: préstamo", "3: dibujo", "4: computadora"]
vectores_por_modelo = {
    "Small": vectores_frases,
    "Large": vectores_para(FRASES, LARGE, mode="offline"),
}
mostrar_figura(dibujar_modelos(etiquetas_frases, vectores_por_modelo))
for numero, frase in enumerate(FRASES, start=1):
    print(numero, frase)

# %% [markdown]
# **Actividad visual:** predice qué pareja quedará más cercana antes de ejecutar.
# Después explica un resultado usando sus textos. ¿Un score de 0.7 significa un
# 70 % de probabilidad de una respuesta correcta? No: sigue siendo un coseno.
#
# ## 5. Guardar y consultar vectores: una base real, un ejemplo pequeño
#
# Qdrant guarda cada **punto** con ID, vector y **payload**: sus datos asociados.
# Una **colección** agrupa puntos compatibles. Empezamos con tres vectores manuales
# para leer la programación sin mezclarla todavía con llamadas al modelo.
# Usamos almacenamiento en disco y volvemos a abrirlo para comprobar persistencia.

# %%
carpeta_temporal = TemporaryDirectory(prefix="henry-rag-clase-")
ruta_db = Path(carpeta_temporal.name) / "qdrant"

with abrir_base(ruta_db) as bd:
    bd.create_collection(
        collection_name="mapa_manual",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    bd.upsert("mapa_manual", points=[
        models.PointStruct(id=1, vector=[0.90, 0.45], payload={"texto": "Taller de cómics"}),
        models.PointStruct(id=2, vector=[0.10, 1.00], payload={"texto": "Préstamo de libros"}),
        models.PointStruct(id=3, vector=[-0.80, 0.28], payload={"texto": "Uso de computadoras"}),
    ])

# %%
with abrir_base(ruta_db) as bd:
    resultado_manual = bd.query_points(
        collection_name="mapa_manual", query=pregunta_manual, limit=2, with_payload=True,
    ).points
    print("Puntos conservados tras cerrar y abrir:", bd.count("mapa_manual").count)
for punto in resultado_manual:
    print(punto.payload["texto"], "coseno:", round(punto.score, 3))

# %% [markdown]
# **Ejercicio 1 — cambiar cuántos resultados recibimos.**
# `k` es la cantidad máxima de resultados. Corrige `k_practica` para obtener dos
# puntos y ejecuta la celda. Observa que aumentar `k` incorpora resultados menos
# cercanos; no añade documentos nuevos ni mejora automáticamente la respuesta.

# %%
k_practica = 1  # Cambia este valor para obtener dos resultados.
with abrir_base(ruta_db) as bd:
    puntos_practica = bd.query_points("mapa_manual", query=pregunta_manual, limit=k_practica).points
print("Esperado: 2; obtenido:", len(puntos_practica))
print("Criterio logrado:", len(puntos_practica) == 2)

# %% [markdown]
# ## 6. Indexar los documentos y buscar una pregunta
#
# Ahora sustituimos las coordenadas manuales por embeddings reales. Cada modelo
# recibe su propia colección. `crear_indice` crea la colección y hace `upsert` de
# los fragmentos con IDs estables; repetirlo no duplica registros.
# `buscar` transforma la pregunta con el mismo modelo y llama a `query_points`.
# Los helpers están en [rag_taller.py](../../src/henry_agents/rag_taller.py).
#
# La base está en una carpeta temporal de esta sesión: persiste entre aperturas
# y se conserva mientras repetimos ejercicios. Qdrant local sirve para este pequeño laboratorio;
# no estamos midiendo índices distribuidos ni rendimiento de una instalación servidor.

# %%
colecciones = {SMALL: "centro_small", LARGE: "centro_large"}
textos_fragmentos = [f["texto"] for f in fragmentos]
with abrir_base(ruta_db) as bd:
    for modelo, coleccion in colecciones.items():
        vectores = vectores_para(textos_fragmentos, modelo, mode="offline")
        crear_indice(bd, fragmentos, vectores, coleccion, modelo)
        print(coleccion, "puntos:", bd.count(coleccion).count)

pregunta = CONSULTAS[3]  # La pregunta de Camila: requisitos Y plazo.
with abrir_base(ruta_db) as bd:
    hallazgos = buscar(bd, colecciones[LARGE], pregunta, LARGE, k=4, mode="offline")
print("Pregunta:", pregunta)
for hallazgo in hallazgos:
    print(hallazgo.id, "coseno:", round(hallazgo.score, 4), "texto:", hallazgo.texto)

# %%
mostrar_figura(dibujar_ranking(
    [h.model_dump() for h in hallazgos], titulo="Embeddings reales: resultados del centro cultural",
))

# %% [markdown]
# ## 7. Comparar búsqueda literal y dos modelos de embeddings
#
# La búsqueda literal de esta celda cuenta palabras exactas compartidas. Es un
# baseline simple, sin stemming ni BM25: no representa todos los buscadores léxicos.
# Un código preciso puede beneficiarse de coincidencia literal; una paráfrasis
# puede beneficiarse de embeddings. Una búsqueda híbrida combina ambas señales.
#
# Comparamos las mismas preguntas, documentos vigentes y `k=4`. **No comparamos
# scores Small contra Large como una nota de calidad**: inspeccionamos qué textos
# recuperan y si contienen las piezas necesarias para responder.

# %%
comparacion = []
fuentes_esperadas = [
    {"CC-01-P1", "CC-01-P2"},  # Posibilidad, plazo y requisitos.
    {"CC-01-P1"},              # Plazo inicial.
    {"CC-05-P1"},              # Código de la feria.
    {"CC-01-P1", "CC-01-P2"},  # Plazo y requisitos, pedidos explícitamente.
]
with abrir_base(ruta_db) as bd:
    for caso, (consulta, esperadas) in enumerate(zip(CONSULTAS[:4], fuentes_esperadas, strict=True), 1):
        print("Caso", caso, "—", consulta, "Fuentes esperadas:", sorted(esperadas))
        literales = sorted(
            [f for f in fragmentos if f["vigente"]],
            key=lambda f: palabras_compartidas(consulta, f["texto"]), reverse=True,
        )[:4]
        ids_literal = {f["id"] for f in literales}
        comparacion.append({
            "Caso": caso, "Método": "Literal", "IDs": ", ".join(f["id"] for f in literales),
            "Cobertura@4": len(esperadas & ids_literal) / len(esperadas),
        })
        for modelo, nombre in [(SMALL, "Small"), (LARGE, "Large")]:
            encontrados = buscar(bd, colecciones[modelo], consulta, modelo, k=4, mode="offline")
            comparacion.append({
                "Caso": caso, "Método": nombre, "IDs": ", ".join(h.id for h in encontrados),
                "Cobertura@4": len(esperadas & {h.id for h in encontrados}) / len(esperadas),
            })
print(pd.DataFrame(comparacion).to_string(index=False))

# %%
mostrar_figura(dibujar_coberturas(comparacion))

# %% [markdown]
# **Cobertura de fuentes:** cuenta cuántas piezas esperadas aparecen entre los
# cuatro resultados. Si necesitamos dos y recuperamos una, la cobertura es 0.5.
# Acordamos los IDs esperados leyendo los documentos, antes de medir el buscador.
# Una cobertura de 1 no acredita fidelidad de la generación ni que los otros
# resultados sean relevantes. Estos cuatro casos enseñan a comparar; no permiten
# afirmar que un modelo es mejor en general. Los empates literales conservan
# el orden del corpus; ese buscador elemental también tiene limitaciones.
#
# **Lee los textos antes de elegir un ganador.** En este corpus el primer resultado
# de la pregunta sobre plazo puede hablar de renovar un préstamo. La respuesta
# necesita diferenciar plazo inicial y renovación. `top-1` no siempre basta.
# Más dimensiones tampoco garantizan mejor recuperación para cualquier tarea.
#
# ## Laboratorio visual: qué cambia cuando mueves una decisión
#
# El panel consulta **Qdrant real** cada vez que cambias un control. Las tarjetas
# muestran el texto que llegaría al generador. No se ejecutan llamadas a modelos.
# Usa los controles para estos tres experimentos, cambiando una decisión a la vez:
#
# 1. **Camila:** mantén Large y vigentes. Baja `k` a 1; observa qué necesidades
#    quedan sin fuente. Vuelve a 4 y localiza plazo y requisitos.
# 2. **Norma vieja:** elige la pregunta sobre plazo (caso 2), Large y `k=4`.
#    Desmarca vigencia: aparecerá CC-07-P1. Lee la tarjeta antes de volver a filtrar.
# 3. **Pregunta ajena:** elige Mongolia (caso 5), Large, `k=4` y vigentes.
#    Hay candidatos, pero la generación preparada se abstiene.
#
# Cambiar contexto puede quitar la respuesta guardada. El panel lo indica:
# necesitas una respuesta nueva del estudiante o una generación live voluntaria.
# Al leer el notebook en GitHub los controles no funcionan; ábrelo en JupyterLab.

# %%
explorador = crear_explorador(ruta_db, colecciones)
display(explorador)

# %% [markdown]
# Para interpretar un ID, imprime su texto. Cambia el caso (1 a 4) y el método
# (`"Literal"`, `"Small"` o `"Large"`) y lee los cuatro resultados.

# %%
caso_a_inspeccionar = 4
metodo_a_inspeccionar = "Small"
textos_por_id = {f["id"]: f["texto"] for f in fragmentos}
fila_elegida = next(fila for fila in comparacion
                   if fila["Caso"] == caso_a_inspeccionar and fila["Método"] == metodo_a_inspeccionar)
for id_fragmento in fila_elegida["IDs"].split(", "):
    print(id_fragmento, textos_por_id[id_fragmento])

# %% [markdown]
# **Ejercicio 2 — impedir que una norma archivada llegue al contexto.**
# La base conserva dos fragmentos de una norma anterior. Completa el filtro de
# `vigentes_practica` para excluirlos. Compara IDs y explica qué metadato lo permite.
# El filtro no depende de que el modelo entienda las palabras «ya no vigente».

# %%
vigentes_practica = False  # Cámbialo para consultar sólo documentos vigentes.
with abrir_base(ruta_db) as bd:
    sin_filtro = buscar(bd, colecciones[LARGE], CONSULTAS[1], LARGE, k=10, vigentes=False)
    filtrados_practica = buscar(
        bd, colecciones[LARGE], CONSULTAS[1], LARGE, k=10, vigentes=vigentes_practica,
    )
print("Archivados sin filtro:", [h.id for h in sin_filtro if not h.vigente])
print("Esperado: ningún archivado; obtenido:", [h.id for h in filtrados_practica if not h.vigente])
print("Criterio logrado:", all(h.vigente for h in filtrados_practica))

# %% [markdown]
# ## 8. Del contexto a una respuesta RAG completa
#
# Recuperar párrafos todavía no genera una respuesta. El **contexto** son los
# fragmentos que enviamos junto con la pregunta al modelo que redacta.
# Le pedimos responder con esa información, incluir citas y declarar falta de
# evidencia cuando no alcanza. Sus instrucciones no garantizan que siempre cumpla.
#
# Para repetir sin API, mostraremos una respuesta **real guardada** del mismo
# conjunto de fragmentos. Conserva procedencia, modelo y consumo de la preparación.
# Si cambias pregunta o contexto, esa reproducción no se aplica: debes inspeccionar
# los documentos o activar una generación nueva. No se sustituye silenciosamente.

# %%
print("PREGUNTA:", pregunta)
print("CONTEXTO QUE RECIBE EL GENERADOR:")
for hallazgo in hallazgos:
    print(f"[{hallazgo.id}] {hallazgo.texto}")

respuesta_preparada = reproducir_respuesta(pregunta, hallazgos)
preparacion = leer_respuestas()
print("Preparación real:", preparacion["generado_utc"])
print("REPRODUCCIÓN de una generación anterior:", respuesta_preparada.respuesta)
for cita in respuesta_preparada.citas:
    print("Fuente:", cita.fragmento_id, "frase literal:", cita.cita_literal)
print("Modelo de la preparación:", respuesta_preparada.modelo_usado)
print("Tokens de la preparación:", respuesta_preparada.uso_tokens)
print("Llamadas nuevas al mostrar esta respuesta: 0")
print("Problemas de citas:", validar_citas(respuesta_preparada, hallazgos))

# %% [markdown]
# **Ejercicio 3 — recuperar todas las piezas de una pregunta.**
# Usa la pregunta preparada sobre requisitos y plazo. Cambia `k_cobertura` hasta
# incluir CC-01-P1 y CC-01-P2. Lee ambas fuentes: una contiene el plazo y otra la
# credencial y el documento de identidad. El código comprueba presencia de fuentes;
# tú debes comprobar si sus frases apoyan la respuesta.

# %%
k_cobertura = 1  # Amplía la recuperación para incluir plazo Y requisitos.
with abrir_base(ruta_db) as bd:
    cobertura_practica = buscar(bd, colecciones[LARGE], CONSULTAS[3], LARGE, k=k_cobertura)
ids_cobertura = [h.id for h in cobertura_practica]
print("Esperado: CC-01-P1 y CC-01-P2; obtenido:", ids_cobertura)
print("Criterio logrado:", {"CC-01-P1", "CC-01-P2"}.issubset(ids_cobertura))

# %% [markdown]
# ## 9. Cuando el buscador devuelve algo que no responde
#
# Un buscador de vecinos cercanos puede devolver los mejores puntos disponibles
# aunque ninguno conteste la pregunta. Veamos qué pasa con una pregunta cuya
# respuesta no está en los documentos del centro cultural.
# No fijamos un umbral universal de coseno: debe calibrarse con casos de la aplicación.

# %%
pregunta_sin_datos = CONSULTAS[4]
with abrir_base(ruta_db) as bd:
    irrelevantes = buscar(bd, colecciones[LARGE], pregunta_sin_datos, LARGE, k=4)
print("Pregunta:", pregunta_sin_datos)
print("El buscador devuelve IDs:", [h.id for h in irrelevantes])
sin_evidencia = reproducir_respuesta(pregunta_sin_datos, irrelevantes)
print("Generación real preparada, estado:", sin_evidencia.estado)
print(sin_evidencia.respuesta)
print("¿Incluye citas?:", bool(sin_evidencia.citas))

# %% [markdown]
# Una fuente citada puede existir y aun así no sostener toda la respuesta.
# Los controles de Python revisan ID y cita literal; la fidelidad exige comparar
# afirmaciones con texto. No evaluamos sólo «¿ejecutó sin error?».
#
# **Ejercicio 4 — una cita válida puede acompañar una afirmación falsa.**
# En esta respuesta alterada conservamos una cita real y cambiamos la afirmación.
# Predice si el control de citas la detectará. Localiza luego la frase que demuestra
# el error. Copia la frase del plazo vigente y escribe la corrección en tus palabras.
# Explica por qué siete días contradice treinta. Este caso separa contrato y fidelidad.

# %%
alterada = respuesta_preparada.model_copy(deep=True)
alterada.respuesta = "La norma vigente permite conservar los libros durante treinta días."
print("Control de IDs y citas:", validar_citas(alterada, hallazgos))
print("Afirmación alterada:", alterada.respuesta)
print("Fuente que debemos leer:", fragmentos[0]["texto"])
correccion_practica = ""  # Escribe el plazo vigente respaldado por CC-01-P1.
cita_de_apoyo = ""  # Copia la frase completa que establece el plazo vigente.
plazo_en_la_fuente = "El plazo vigente de préstamo es de siete días calendario."
print("Cita copiada coincide con la norma vigente:", cita_de_apoyo == plazo_en_la_fuente)
print("Corrección escrita:", correccion_practica or "Pendiente")
print("Revisión humana pendiente: ¿tu corrección dice siete días y coincide con esa cita?")

# %% [markdown]
# ## 10. Programar una consulta nueva con modelos reales
#
# La preparación y la reproducción permiten inspeccionar datos → embeddings →
# Qdrant → contexto → generación con fuentes. Ahora puedes ejecutar ese flujo con una pregunta
# propia. Modifica `mi_pregunta` y activa la bandera cuando quieras hacer llamadas.
# `buscar(..., mode="live")` calcula su embedding; `responder(..., mode="live")`
# llama al generador configurado en `.env`. No pegues la clave en el notebook.
#
# Son dos llamadas nominales: una de embeddings y una de generación. Un error
# se propaga para diagnosticarlo. Cambiar de modelo de embeddings exige indexar
# otra colección compatible; cambiar el generador no cambia los vectores guardados.

# %%
EJECUTAR_LIVE = False
mi_pregunta = "¿Qué materiales debo llevar al taller de dibujo?"
if EJECUTAR_LIVE:
    with abrir_base(ruta_db) as bd:
        mis_fuentes = buscar(bd, colecciones[LARGE], mi_pregunta, LARGE, k=4, mode="live")
    print("Fuentes enviadas al generador:", [h.id for h in mis_fuentes])
    mi_respuesta = responder(mi_pregunta, mis_fuentes, mode="live")
    print("GENERACIÓN NUEVA:", mi_respuesta.respuesta)
    print("Estado:", mi_respuesta.estado, "modelo:", mi_respuesta.modelo_usado)
    print("Tokens de generación:", mi_respuesta.uso_tokens)
    for cita in mi_respuesta.citas:
        print("Fuente:", cita.fragmento_id, "frase literal:", cita.cita_literal)
    print("Controles:", validar_citas(mi_respuesta, mis_fuentes))
else:
    print("Laboratorio live desactivado. No se hicieron llamadas nuevas.")

# %% [markdown]
# ## 11. Entrega: explica un fallo y modifica el flujo
#
# Elige una de las preguntas preparadas, predice sus fuentes, cambia `k` o el filtro
# y registra los IDs y textos obtenidos. Entrega pregunta, contexto, respuesta,
# cita literal y una explicación de qué decisión cambió el resultado.
# Si usas una pregunta propia, distingue una generación nueva de una reproducción.
#
# **Ruta sin clave:** compara los resultados originales con los modificados.
# Si cambia el contexto, redacta tú la respuesta y rotúlala **«respuesta del
# estudiante»**; si faltan datos, indica qué parte no puedes responder. Copia una
# cita de los fragmentos modificados. Esa respuesta no es una generación del modelo
# ni una reproducción válida de la anterior. La ruta live es voluntaria.
#
# Para un resultado incorrecto, identifica la etapa antes de corregir:
#
# | Lo que observas | Qué revisar |
# |---|---|
# | Faltan requisitos aunque existe el documento | Fragmentación y cobertura de recuperación |
# | Aparece una regla anterior | Metadatos y filtro de vigencia |
# | Hay texto cercano, pero ajeno a la pregunta | Relevancia y suficiencia del contexto |
# | Una afirmación contradice la cita | Fidelidad de la generación |
# | Se consulta otra dimensión o modelo | Compatibilidad del espacio vectorial |
#
# **Comprobación de salida:** explica qué guardamos en Qdrant, qué recibe el
# generador, por qué un score no es una probabilidad y cuándo corresponde abstenerse.

# %% [markdown]
# ## Ampliación visual opcional: una sombra de los embeddings
#
# PCA reduce muchas coordenadas a dos para dibujar. La proyección pierde información:
# cercanía en el dibujo puede diferir del ranking. El gráfico identifica IDs;
# buscamos y calculamos cosenos con los vectores completos, nunca con estas posiciones.

# %%
vectores_para_mapa = vectores_para(textos_fragmentos, SMALL, mode="offline")
mostrar_figura(dibujar_proyeccion(
    [f["id"] for f in fragmentos], vectores_para_mapa,
    grupos=["vigente" if f["vigente"] else "archivado" for f in fragmentos],
))

# %% [markdown]
# ## Ampliaciones para continuar la práctica
#
# El recorrido principal termina con la entrega de Camila. Las secciones 12–15
# amplían el laboratorio y pueden trabajarse en otra sesión. Mantienen la misma
# base y añaden decisiones que aparecen al construir proyectos de RAG.
#
# ## 12. Fragmentación y solapamiento: qué se pierde al cortar
#
# Dividir por caracteres es fácil, pero puede separar una condición de su regla.
# **Solapamiento** significa repetir parte del texto entre fragmentos vecinos.
# Ayuda a conservar contexto cerca del corte, a cambio de repetir información.
# Los tamaños siguientes cuentan caracteres Python, no tokens del proveedor.
#
# Compara el documento de préstamo dividido por párrafos con cortes pequeños.
# Modifica `tamano_fragmento` y `solapamiento`. Mira qué pasa con la frase del plazo.
# Este experimento inspecciona los cortes; no calcula embeddings de esos textos nuevos.

# %%
tamano_fragmento = 100
solapamiento = 25
partes_por_caracteres = dividir_por_caracteres(
    documentos[0]["texto"], tamano=tamano_fragmento, solapamiento=solapamiento,
)
mostrar_figura(dibujar_fragmentacion(documentos[0]["texto"], partes_por_caracteres))
for parte in partes_por_caracteres:
    print(parte["id"], parte["inicio"], parte["fin"], repr(parte["texto"]))

# %% [markdown]
# **Comprueba:** ¿algún corte deja incompleta la frase «siete días calendario»?
# ¿El solapamiento recupera esa frase completa en otro fragmento? Repetir texto no
# garantiza recuperar todas las condiciones. El tamaño se evalúa con preguntas reales.
# Si indexaras estos nuevos fragmentos, necesitarías nuevos embeddings y sus IDs.
#
# ## 13. Búsqueda híbrida: combinar palabras y significado
#
# La búsqueda literal encuentra palabras exactas; los embeddings pueden recuperar
# una intención expresada de otra forma. Aquí combinamos sus **posiciones** mediante
# *Reciprocal Rank Fusion* (RRF): cada lista aporta `1 / (60 + posición)` por candidato.
# Un fragmento que aparece en ambas recibe dos aportes. No sumamos cosenos con
# conteos de palabras, porque son medidas de escalas distintas.
#
# En este ejemplo léxico sencillo, un empate conserva el orden del corpus. La
# búsqueda híbrida tampoco garantiza mejorar; mide el resultado con las mismas fuentes.
# Recuperamos seis candidatos de cada lista y fusionamos sus posiciones para
# seleccionar cuatro; las métricas compararán esos cuatro resultados finales.

# %%
consulta_hibrida = CONSULTAS[3]
literal_ordenado = sorted(
    [f for f in fragmentos if f["vigente"]],
    key=lambda f: palabras_compartidas(consulta_hibrida, f["texto"]), reverse=True,
)
with abrir_base(ruta_db) as bd:
    semanticos = buscar(bd, colecciones[SMALL], consulta_hibrida, SMALL, k=6)
ids_literal = [f["id"] for f in literal_ordenado[:6]]
ids_semanticos = [h.id for h in semanticos]
hibridos = fusionar_rankings([ids_literal, ids_semanticos], k=4)
display(pd.DataFrame(hibridos))
for resultado in hibridos:
    print(resultado["id"], textos_por_id[resultado["id"]])
print("Fuentes esperadas:", sorted({"CC-01-P1", "CC-01-P2"}))

# %% [markdown]
# `rrf_score` sirve para ordenar esta fusión. No es coseno, confianza ni probabilidad.
# **Actividad:** compara los cuatro primeros de literal, Small e híbrido.
# Identifica qué fuente se añadió o salió; no elijas por el score más alto.
#
# ## 14. Evaluar recuperación: cobertura y ruido
#
# **Recall**: fuentes esperadas recuperadas / fuentes esperadas.
# **Precision**: fuentes esperadas recuperadas / resultados devueltos.
# La cobertura del recorrido principal era recall. Ahora vemos también qué
# fracción del contexto responde las necesidades que anotamos previamente.
#
# Estas etiquetas incluyen sólo plazo y requisitos; una renovación puede ser útil
# como información adicional aunque no cuente como esperada aquí. La calidad de las
# etiquetas condiciona la métrica. No evaluamos la redacción del generador con ellas.

# %%
metricas = []
with abrir_base(ruta_db) as bd:
    for k_evaluacion in [1, 2, 3, 4, 6]:
        recuperados = buscar(bd, colecciones[LARGE], CONSULTAS[3], LARGE, k=k_evaluacion)
        evaluacion = evaluar_recuperacion([h.id for h in recuperados], {"CC-01-P1", "CC-01-P2"})
        metricas.append({"k": k_evaluacion, "IDs": ", ".join(h.id for h in recuperados), **evaluacion})
display(pd.DataFrame(metricas))
mostrar_figura(dibujar_metricas(metricas))

# %% [markdown]
# Esta gráfica cambia `k` usando sólo Large. Para evaluar la fusión anterior,
# comparamos en otra tabla Literal, Small y Literal+Small con cuatro resultados,
# el mismo corpus, filtro y pregunta. No mezclamos las dos comparaciones.

# %%
comparacion_hibrida = []
for metodo, ids in [
    ("Literal", ids_literal[:4]),
    ("Small", ids_semanticos[:4]),
    ("Literal + Small (RRF)", [h["id"] for h in hibridos]),
]:
    comparacion_hibrida.append({
        "Método": metodo, "IDs": ", ".join(ids),
        **evaluar_recuperacion(ids, {"CC-01-P1", "CC-01-P2"}),
    })
display(pd.DataFrame(comparacion_hibrida))

# %% [markdown]
# **Actividad:** encuentra el menor `k` que recupera ambas fuentes. ¿Aumentarlo
# después añade fuentes necesarias o ruido respecto de estas etiquetas?
# Mantén fijo el corpus, filtro y modelo mientras comparas. Amplía luego el conjunto
# de preguntas antes de recomendar una configuración para otro proyecto.
#
# ## 15. Actualizar información: texto, metadatos y embeddings
#
# Una base vectorial necesita mantenerse. Si cambia el texto, se calcula un nuevo
# embedding y se actualiza el punto. Si sólo cambia su vigencia, se puede modificar
# el payload sin recalcular el vector. Retirar una norma puede dejar preguntas sin
# respuesta: hay que comprobar que exista su reemplazo vigente.
#
# Practicaremos retirar el documento de préstamo en una colección de laboratorio.
# Su texto y vectores siguen iguales, pero el filtro dejará de seleccionarlo.

# %%
with abrir_base(ruta_db) as bd:
    crear_indice(bd, fragmentos, vectores_para(textos_fragmentos, SMALL), "laboratorio_versiones", SMALL)
    antes_retiro = buscar(bd, "laboratorio_versiones", CONSULTAS[3], SMALL, k=4)
    cantidad_antes = bd.count("laboratorio_versiones").count
    registros, _ = bd.scroll("laboratorio_versiones", limit=100, with_payload=True)
    ids_a_retirar = [p.id for p in registros if p.payload["documento_id"] == "CC-01"]
    bd.set_payload("laboratorio_versiones", {"vigente": False}, points=ids_a_retirar)
    tras_retiro = buscar(bd, "laboratorio_versiones", CONSULTAS[3], SMALL, k=4)
    cantidad_despues = bd.count("laboratorio_versiones").count
print("Fragmentos retirados:", len(ids_a_retirar))
print("Puntos guardados antes y después:", cantidad_antes, cantidad_despues)
comparacion_retiro = []
for estado, fuentes in [("Antes", antes_retiro), ("Después", tras_retiro)]:
    ids = [h.id for h in fuentes]
    comparacion_retiro.append({"Estado": estado, "IDs": ", ".join(ids),
                              **evaluar_recuperacion(ids, {"CC-01-P1", "CC-01-P2"})})
display(pd.DataFrame(comparacion_retiro))

# %% [markdown]
# **Cierre de las ampliaciones:** relaciona un fallo con una acción: ajustar cortes,
# combinar búsquedas, ampliar el conjunto de evaluación o actualizar documentos.
# Para cambiar el corpus oficial de la práctica, el docente regenera los embeddings
# y las respuestas preparadas; el replay rechaza datos de un corpus anterior.
# En el retiro cambiamos metadatos: los catorce puntos siguen guardados. El filtro
# excluye la norma, pero no la borra físicamente ni agrega una norma de reemplazo.

# %% [markdown]
# ## Soluciones de referencia
#
# <details><summary>Ejercicio 1: ampliar k</summary>
#
# ```python
# k_practica = 2
# with abrir_base(ruta_db) as bd:
#     puntos_practica = bd.query_points("mapa_manual", query=pregunta_manual, limit=k_practica).points
# assert len(puntos_practica) == 2
# ```
#
# </details>
#
# <details><summary>Ejercicio 2: filtrar por metadatos</summary>
#
# ```python
# vigentes_practica = True
# with abrir_base(ruta_db) as bd:
#     filtrados_practica = buscar(bd, colecciones[LARGE], CONSULTAS[1], LARGE, k=10, vigentes=vigentes_practica)
# assert all(h.vigente for h in filtrados_practica)
# ```
#
# El helper traduce esta decisión a un filtro de Qdrant con `key="vigente"`
# y `MatchValue(value=True)`. Se ejecuta durante la consulta, antes del generador.
# </details>
#
# <details><summary>Ejercicio 3: reunir plazo y requisitos</summary>
#
# ```python
# k_cobertura = 4
# with abrir_base(ruta_db) as bd:
#     cobertura_practica = buscar(bd, colecciones[LARGE], CONSULTAS[3], LARGE, k=k_cobertura)
# assert {"CC-01-P1", "CC-01-P2"}.issubset({h.id for h in cobertura_practica})
# ```
#
# El generador todavía debe distinguir el préstamo inicial de una renovación.
# </details>
#
# <details><summary>Ejercicio 4: corregir una afirmación sin cambiar la fuente</summary>
#
# ```python
# correccion_practica = "El plazo vigente de préstamo es de siete días calendario."
# cita_de_apoyo = "El plazo vigente de préstamo es de siete días calendario."
# assert cita_de_apoyo == plazo_en_la_fuente
# assert correccion_practica in fragmentos[0]["texto"]
# assert validar_citas(alterada, hallazgos) == []
# # El control pasa: evalúa citas existentes, no la fidelidad de esta afirmación.
# ```
#
# </details>
#
# ## Cerrar la práctica
#
# La base queda disponible para volver a ejecutar los ejercicios después de
# «Run All». La persistencia se comprobó al cerrar y reabrir Qdrant durante la sesión.
# Activa la última bandera sólo cuando hayas terminado: borra la carpeta temporal.
# Si reinicias el kernel, la próxima ejecución crea una práctica nueva.
# Para conservar el proyecto entre
# sesiones, usa una carpeta propia bajo `.local/` y cierra el cliente antes de
# abrirlo desde otro kernel. Reinicia y ejecuta todo para reproducir la clase.

# %%
LIMPIAR_AL_TERMINAR = False
if LIMPIAR_AL_TERMINAR:
    carpeta_temporal.cleanup()
    print("Carpeta de práctica eliminada; ejecuta de nuevo desde la creación de la base.")
else:
    print("Base conservada durante la sesión: puedes repetir los ejercicios.")

# %% [markdown]
# Referencias primarias: [embeddings de OpenAI](https://developers.openai.com/api/docs/guides/embeddings),
# [búsqueda semántica](https://developers.openai.com/api/docs/guides/retrieval),
# [Qdrant local](https://github.com/qdrant/qdrant-client#local-mode) y
# [colecciones](https://qdrant.tech/documentation/manage-data/collections/).
# La [guía del taller](README.md) reúne preparación, evaluación y ampliaciones.
