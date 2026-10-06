# Proyecto · RAG que sabe cuándo volver a buscar

La tienda recibe preguntas sobre envíos, retiro, cambios, horarios, pagos y stock.
Construye una comparación entre una búsqueda única y un ciclo que puede reunir
evidencia adicional. La entrega debe explicar qué respaldo tiene cada respuesta,
cuándo se abstiene y cuánto trabajo adicional realiza.

Prerrequisito: [clase 08](../../clases/agentic_workflows/08_rag_desde_cero.ipynb) y
[clase 09](../../clases/agentic_workflows/09_agentic_rag.ipynb). Usa Python, LangGraph
y los datos locales del repo. No necesitas clave ni una base vectorial externa.
Consulta la [guía RAG](../../docs/RAG_Y_AGENTIC_RAG.md) si un concepto no está claro.

## Fase 1 · Preparar y probar la recuperación

1. Lee el manual y explica por qué la política archivada se excluye. Conserva
   documento de origen, ID, categoría y vigencia al fragmentar.
2. Compara tamaños pequeños y grandes. Muestra una condición que se pierde al
   dividir demasiado y el efecto de repetir información por solapamiento.
3. Construye un único `IndiceLexico` y reutilízalo en ambas alternativas con `k=1`.
   No mejores solo el recuperador del agente y atribuyas la mejora al ciclo.
4. Prueba pregunta directa, pregunta compuesta y consulta sin evidencia. Registra
   recall por documento y explica por qué un score alto no prueba fidelidad.

## Fase 2 · Recuperar, observar y decidir

Completa `comparar(pregunta)` en [starter.py](starter.py). Devuelve un diccionario
con las claves `clasico` y `agentico`, cada una con estado, respuesta, fuentes,
evidencia y cantidad de búsquedas. El agente también debe conservar sus eventos.
Puedes reutilizar los nodos del curso: la explicación del estado y de las rutas
debe ser tuya. No se evalúa escribir una segunda biblioteca de LangGraph.

Permite buscar, consultar el catálogo, responder o abstenerse. Exige evidencia
para cada necesidad, IDs válidos, citas existentes y fidelidad de las afirmaciones.
Demuestra límite de una búsqueda, bloqueo de consulta repetida, rechazo de una
respuesta prematura y abstención ante una pregunta fuera del manual.

Incluye dos pruebas distintas de generación: una cita inexistente y una afirmación
que contradice una cita real. La primera es un problema de referencias; la segunda
exige revisar contenido. No basta con comprobar que la lista de IDs no esté vacía.

## Entrega y defensa

- Tu `comparar` ejecutable, un diagrama y la explicación de su estado.
- Resultados de los doce [casos de referencia](../../src/henry_agents/data/rag_casos.json)
  y dos casos propios, uno de ellos sin respuesta suficiente.
- Por sistema: estados, documentos recuperados/citados, recall, búsquedas y
  llamadas de chat. Explica cada consulta adicional y si compensó su costo.
- Una falla intencional con evidencia del bloqueo. Puedes presentar la defensa
  oralmente, en texto o con un diagrama acompañado del código ejecutado.

La referencia y su evaluación se ejecutan desde la raíz:

```bash
uv run python proyectos/tienda_rag/starter.py
uv run python proyectos/tienda_rag/solution.py
uv run python scripts/evaluate_rag.py --mode offline
uv run pytest -q tests/test_rag.py
```

`starter.py` muestra un resultado pendiente hasta que lo completes.
El evaluador ejecuta **la implementación de referencia**, no importa automáticamente
tu starter. Usa sus casos para llamar tu función y contrastar estados y documentos.
Consulta [solution.py](solution.py) después de intentar el proyecto.

## Rúbrica

| Criterio | Puntos | Evidencia esperada |
|---|---|---|
| Ingesta y recuperación | 2 | Fragmentos rastreables, vigencia y comparación con el mismo índice |
| Cobertura y abstención | 2 | Dos necesidades cubiertas y ausencia de evidencia reconocida |
| Citas y fidelidad | 2 | Rechazo de ID/cita falsa y de afirmación contradictoria |
| Ciclo y límites | 2 | Observaciones cambian la búsqueda; presupuesto y repetición se bloquean |
| Comparación y explicación | 2 | Casos propios, métricas interpretadas y costo de buscar más |

En offline la decisión es una simulación con reglas explícitas. La extensión live
usa un LLM coordinador y otro rol para respuesta/revisión; es voluntaria y consume
API. Una ejecución correcta offline no demuestra calidad live. Justifica cuándo
una búsqueda única alcanza y cuándo el ciclo aporta algo; más consultas por sí
solas no justifican usar un agente.
