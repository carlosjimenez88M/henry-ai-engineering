# Estudio del origen y decisiones para la ruta inicial

Origen estudiado: [udacity/cd14525-agentic-workflows-classroom](https://github.com/udacity/cd14525-agentic-workflows-classroom/tree/e6e3fa1c28dd91f0b86624312e2f9d5d31df9886),
commit `e6e3fa1c28dd91f0b86624312e2f9d5d31df9886`, copia local examinada el
6 de octubre de 2026. Se leyeron demos, ejercicios/soluciones, diagramas y el proyecto
con sus dos fases. No se ejecutaron ejemplos que requieren la plataforma de Udacity.

## Qué enseña

Diez lecciones: introducción, decisiones de workflows, modelado, implementación,
cadena de prompts, routing, paralelismo, evaluador–optimizador, orquestador–workers
y repaso. El proyecto construye una librería de agentes y luego conecta planificación,
conocimiento, routing y evaluación para producir un plan de desarrollo de producto.

## Correspondencia conceptual

| Lecciones del origen | Ruta nueva en Henry | Cambio para quien empieza |
|---|---|---|
| 1–2 · Introducción y decisiones | 00 Python + 01 IA y herramientas | Puente de Python; modelo, regla y herramienta diferenciados |
| 3 · Modelado | 02 Modelado y cadena | Diagrama del mismo pedido que luego se ejecuta |
| 4 · Implementación | 01–02 + fase 1 del proyecto | Funciones pequeñas y contratos antes de clases y herencia |
| 5 · Prompt chaining | 02 Cadena | Cotizar → redactar con dependencia visible y abstención |
| 6 · Routing | 03 Routing | Productos/entregas/humano; baseline con tildes y ambigüedad |
| 7 · Paralelismo | 04 Paralelismo | Workers devuelven resultados; la falla se propaga al coordinador |
| 8 · Evaluador–optimizador | 05 Evaluador–optimizador | Criterios estructurados, historial y límite explícito |
| 9 · Orquestador–workers | 06 Orquestador–workers | Plan variable con lista permitida y completitud validada |
| 10 · Repaso | 07 Proyecto y defensa | Decisiones y límites demostrados con casos |
| Proyecto fase 1 | Proyecto Henry fase 1 | Biblioteca de funciones/contratos, reutilización permitida |
| Proyecto fase 2 | Proyecto Henry fase 2 | Preparar pedido, revisar y decidir; contexto único y sencillo |
| Conocimiento añadido y RAG del proyecto | Evidencia local en 01–02; RAG en recorrido ampliado | Recuperación por identificador no se presenta como embeddings ni RAG completo |

La cobertura es de conceptos: no se afirma que Henry replique los siete tipos de
agentes del proyecto original. El recorrido ampliado preexistente complementa con
LangChain, LangGraph, agentes con herramientas, revisión humana y Deep Agents.

## Hallazgos concretos de reproducción

| Hallazgo observado en el origen | Decisión en Henry |
|---|---|
| Demos 1, 2 y 5–9 usan `openai.vocareum.com/v1` y clave de API | Ruta local predeterminada; experimento de LLM explícito y opcional |
| Los ejemplos mezclan nombres de modelos y acceso a APIs | Se reutiliza la configuración central de Henry; no se publican nuevos precios supuestos |
| Demo 2 transforma errores de API en una estrategia de prioridad | La comparación live propaga el fallo; no pasa a simulación sin decirlo |
| Demo 4 simula una verificación con `accuracy="high"` y `verified_claims=3` | Criterios se comprueban contra datos; no se inventa una certeza de verificación |
| Demo 5 pregunta por novedades sin recuperar fuentes externas | Caso estático con catálogo y referencias locales |
| Demo 7 usa hilos y un diccionario global de resultados | Futuros por ejecución, sin estado global, y error de worker observable |
| Evaluadores reconocen aprobación por prefijo de texto | Errores estructurados; agotamiento de intentos es un estado de falla |
| Demo 9 parsea planes XML línea a línea y admite worker genérico | Lista de nombres registrados, número limitado y validación de todos los pasos |
| RAG del proyecto convierte embeddings CSV con `eval(x)` | JSON local y aritmética explícita; no se evalúan expresiones de datasets |
| Algunas plantillas son incompletas o diagramas, no programas terminados | Demos/notebooks ejecutables; la plantilla del estudiante se identifica como pendiente |
| Ejercicios incluyen regulación, inversión o interpretación clínica | Casos originales de una tienda ficticia, sin conocimiento especializado requerido |

Son observaciones del código del commit estudiado, no resultados de una ejecución
de la plataforma original. Tampoco un workflow con reglas deja de ser útil por no usar IA:
se compara qué parte necesita flexibilidad y qué parte conviene mantener verificable.

## Adaptación pedagógica

Un solo escenario reduce la cantidad de contexto nuevo en cada clase. El español
es claro, se explican los anglicismos y se reconoce vocabulario regional sin asumir
un país ni moneda compartidos. Cada notebook incluye objetivo, prerrequisito,
predicción, implementación visible, dos pausas, falla intencional, taller, pista,
solución y ticket de salida. La clase 00 ofrece el Python mínimo.

Los notebooks y sus scripts se sincronizan. La verificación recorre subcarpetas,
conserva reportes separados y ejecuta cada notebook desde un kernel nuevo. Así
una celda no puede depender silenciosamente de variables de una sesión pasada.

## Atribución y contenido original

El [LICENSE.md del origen](https://github.com/udacity/cd14525-agentic-workflows-classroom/blob/e6e3fa1c28dd91f0b86624312e2f9d5d31df9886/LICENSE.md)
declara CC BY-NC-ND 4.0 con condiciones adicionales. Se conserva aquí la referencia
para no confundir estudiar conceptos con una licencia abierta de traducción o copia.
Las consignas, textos, catálogo, proyecto y código incorporados a Henry se escribieron
desde cero; no se trasladaron archivos, traducciones ni extractos del curso de Udacity.
No se agregó a Henry una licencia que otorgue derechos sobre el material de origen.

## Documentación primaria de apoyo

- [Workflows y agentes de LangChain](https://docs.langchain.com/oss/python/langgraph/workflows-agents):
  comparación de patrones y distinción entre workflows y agentes.
- [concurrent.futures de Python](https://docs.python.org/3.13/library/concurrent.futures.html):
  futuros, executor y propagación de errores. Se usa `ThreadPoolExecutor`, compatible con Python 3.13.
- [Salida estructurada de LangChain](https://docs.langchain.com/oss/python/langchain/structured-output):
  contratos para la comparación live. La implementación concreta se contrastó también
  con `langchain-openai` instalado y se probó con un doble sin red.

Consultadas el 6 de octubre de 2026. La ejecución live, otras plataformas y la
calidad pedagógica en una cohorte requieren evidencia propia; ver `VALIDACION.md`.
