# %% [markdown]
# # 08 · RAG desde cero: recuperar antes de responder
#
# **Meta:** construir un sistema que consulte un manual y muestre evidencia para
# responder. **Necesitas:** funciones/diccionarios y los contratos de las clases 01–02.
# **Producto:** índice, consultas, respuesta con citas y una evaluación de recuperación.
#
# RAG significa *Retrieval-Augmented Generation*: recuperar información, agregarla
# al contexto y generar una respuesta. No reentrena el modelo ni garantiza verdad.
# La recuperación y la redacción pueden fallar por separado; por eso las medimos aparte.
#
# En offline hacemos búsqueda real y **respuesta extractiva**: copiamos evidencia
# en lugar de consultar un LLM. En live se puede generar y revisar una respuesta.
# No confundas el baseline lexical con embeddings semánticos aprendidos.
#
# ```mermaid
# flowchart LR
#   D[Documentos] --> F[Fragmentos y metadatos] --> I[Índice]
#   P[Pregunta] --> B[Buscar en índice] --> E[Evaluar cobertura]
#   I --> B
#   E -->|Suficiente| G[Generar y validar citas]
#   E -->|Insuficiente| A[Abstenerse]
# ```

# %%
from henry_agents.config import MODELOS_VERIFICADOS_EL, model_name
from henry_agents.rag import (
    Afirmacion,
    IndiceLexico,
    IndiceVectorial,
    Necesidad,
    RespuestaRAG,
    documentos,
    evaluar_evidencia,
    fragmentar,
    rag_clasico,
    revisar_fidelidad,
    validar_respuesta,
)

print("Modelos verificados:", MODELOS_VERIFICADOS_EL)
print("Generación:", model_name("rag"), "| Embeddings:", model_name("embeddings"))

# %% [markdown]
# ## 1. Preparación del conocimiento: documentos y versiones
# La tienda tiene un manual inventado de entregas, retiro, cambios, horarios y pagos.
# El catálogo de stock es otra fuente; un manual no garantiza disponibilidad actual.
# **Predice:** si una política archivada dice "gratis" y la vigente dice USD 2.00,
# ¿qué debe entrar al contexto? La vigencia viene de metadatos, no del parecido textual.

# %%
docs = documentos()
for doc in docs:
    print(doc["id"], doc["categoria"], "vigente:", doc["vigente"])
assert len(docs) == 6
assert sum(d["vigente"] for d in docs) == 5

# %% [markdown]
# ## 2. Fragmentación: tamaño, solapamiento y origen
# Un *chunk* es un fragmento recuperable. Muy pequeño puede separar condición y
# monto; muy grande añade ruido. El solapamiento repite palabras cerca de un corte.
# Estas ventanas cuentan **palabras**, no tokens del modelo.
#
# Construye una ventana visible antes de usar el helper con metadatos.

# %%
palabras = docs[0]["texto"].split()
TAMANO = 12
SOLAPAMIENTO = 4
ventanas = []
for inicio in range(0, len(palabras), TAMANO - SOLAPAMIENTO):
    ventanas.append(" ".join(palabras[inicio:inicio + TAMANO]))
    if inicio + TAMANO >= len(palabras):
        break
print("Primer fragmento:", ventanas[0])
print("Segundo fragmento:", ventanas[1])
print("Palabras repetidas:", palabras[TAMANO - SOLAPAMIENTO:TAMANO])

# %%
pequenos = fragmentar(palabras=8, solapamiento=0)
completos = fragmentar(palabras=60, solapamiento=10)
print("Fragmentos pequeños:", len(pequenos), "| fragmentos amplios:", len(completos))
print(completos[0])
assert completos[0]["id"] == "ENVIO-2026-c01"

# %% [markdown]
# ## Primera pausa · Reenganche
# Ubica documento, fragmento y `doc_id`. Repetir palabras no agrega conocimiento.
# Para recuperar tras modificar el tamaño, debes reconstruir el índice, no solo el prompt.
#
# ## 3. Índice y ranking: BM25 como baseline gratuito
# El índice calcula coincidencias de palabras y considera su frecuencia en los
# fragmentos. BM25 es búsqueda lexical; no entiende sinónimos generales ni intención.
# `top_k` limita cuántos candidatos pasan al contexto. Un score mayor ordena resultados;
# no es una probabilidad de que la respuesta sea verdadera.

# %%
indice = IndiceLexico(completos)
pregunta = "¿Cuánto cuesta el envío a Centro?"
hits = indice.buscar(pregunta, categoria="entregas", k=2)
for hit in hits:
    print(hit["id"], "score:", hit["score"], "→", hit["texto"])
assert [h["doc_id"] for h in hits] == ["ENVIO-2026"]
assert all(h["vigente"] for h in hits)

# %% [markdown]
# ## 4. Encontrar algo no implica encontrar lo necesario
# La actividad exige categoría entregas y los términos Centro + USD 2.00 en un
# mismo fragmento. Es un criterio explícito para este caso; no un juez semántico universal.
# **Predice:** ¿un corte de ocho palabras puede perder el vínculo entre zona y costo?

# %%
necesidades = [Necesidad(consulta="envío Centro", categoria="entregas",
                        terminos_requeridos=["Centro", "USD 2.00"])]
grade = evaluar_evidencia(necesidades, hits)
print(grade)
assert grade["suficiente"]
hits_pequenos = IndiceLexico(pequenos).buscar(pregunta, categoria="entregas", k=2)
print("Ventanas de 8 palabras:", [(h["id"], h["texto"]) for h in hits_pequenos])
print("Cobertura con corte pequeño:", evaluar_evidencia(necesidades, hits_pequenos)["suficiente"])

# %% [markdown]
# ## 5. Generar, comprobar citas y revisar fidelidad son tareas distintas
# La versión gratuita entrega extractos exactos. La versión live redacta, valida
# ID/cita literal y consulta además un juez de fidelidad. Ese juez puede equivocarse;
# la lectura de fuentes sigue siendo necesaria.

# %%
salida = rag_clasico(pregunta, necesidades, indice=indice, k=2)
assert salida["estado"] == "respondido"
assert salida["llamadas_llm"] == 0
print(salida["respuesta"])

# %% [markdown]
# **Error útil:** una fuente real puede acompañar una afirmación falsa.
# La propuesta siguiente copia la cita correcta y la contradice en su texto.

# %%
propuesta_falsa = RespuestaRAG(abstencion=False, afirmaciones=[Afirmacion(
    texto="El envío a Centro es gratuito.", fuente=hits[0]["id"], cita_literal=hits[0]["texto"],
)])
assert validar_respuesta(propuesta_falsa, necesidades, hits)  # Forma, cita y cobertura pasan.
assert not revisar_fidelidad(propuesta_falsa, hits).respaldada
print("ID válido ≠ afirmación fiel. En offline se rechaza una paráfrasis no extractiva.")

# %% [markdown]
# ## Segunda pausa · Taller: evalúa recuperación antes de retocar el prompt
# Usa dos preguntas y sus documentos esperados. Calcula `recall@k`: proporción de
# documentos necesarios recuperados. Cuenta por documento, no por cada chunk repetido.
# **Pista:** compara conjuntos de `doc_id`; una métrica perfecta aquí solo describe
# estos casos y este corpus. Después añade una pregunta sin respuesta.
#
# ## Solución visible

# %%
casos = [
    ("envío a Centro", {"ENVIO-2026"}),
    ("retiro gratuito y cambios cuadernos", {"RETIRO-2026", "CAMBIOS-2026"}),
]
for consulta, esperados in casos:
    recuperados = {h["doc_id"] for h in indice.buscar(consulta, k=3)}
    recall = len(esperados & recuperados) / len(esperados)
    print(consulta, "recall@3:", recall, "recuperados:", sorted(recuperados))
    assert recall == 1.0
assert rag_clasico("¿Qué garantía tiene un televisor?")["estado"] == "abstencion"

# %% [markdown]
# ## Experimentos opcionales con modelos vigentes
# Mantén ambas variables en False para ejecutar gratis. La primera usa embeddings
# aprendidos y un vector store en memoria; indexar envía el manual a la API y consume saldo.
# La segunda genera y revisa con el modelo de chat. Son experimentos separados.
# Reinicia el kernel tras cambiar `.env`. No hace falta instalar otra librería.

# %%
USAR_EMBEDDINGS_REALES = False
USAR_GENERACION_REAL = False
if USAR_EMBEDDINGS_REALES:
    vectorial = IndiceVectorial(completos)
    print("Búsqueda semántica:", vectorial.buscar("¿Me acercan las compras a mi casa?", k=2))
if USAR_GENERACION_REAL:
    real = rag_clasico(pregunta, necesidades, indice=indice, mode="live")
    print(real["estado"], real["respuesta"], "llamadas de chat:", real["llamadas_llm"])

# %% [markdown]
# ## Ticket de salida
# Señala una falla de ingesta, una de recuperación y una de generación. ¿Por qué
# un embedding no es una base de datos de hechos? ¿Qué cambia si `k=1` pierde
# una de dos fuentes necesarias? No la inventes: abstente o recupera de nuevo.
#
# [Guía RAG y Agentic RAG](../../docs/RAG_Y_AGENTIC_RAG.md) ·
# Siguiente: [09 · Agentic RAG](09_agentic_rag.ipynb).
