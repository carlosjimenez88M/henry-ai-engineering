# Fuentes, decisiones y alcance de la revisión cultural

## Ruta inicial de workflows · octubre de 2026

Se estudió `udacity/cd14525-agentic-workflows-classroom` en el commit
`e6e3fa1c28dd91f0b86624312e2f9d5d31df9886`. El
[análisis del origen](ANALISIS_AGENTIC_WORKFLOWS.md) registra hallazgos, correspondencia
de conceptos y atribución. Se añadieron ocho clases originales, un puente de Python,
un catálogo ficticio de tienda, glosario y un proyecto en dos fases con solución separada.

El recorrido cultural preexistente se conserva como ampliación. La ruta inicial
usa reglas y guiones explicados; el router admite un experimento live opcional.
Las verificaciones, sincronización y diagnóstico ahora incluyen subcarpetas de clases,
con reportes separados y selección por recorrido. No se copiaron archivos de Udacity.

## Revisión cultural preexistente

Se conserva la progresión de los PPTX aportados: herramientas, orquestación/RAG,
sistemas multiagente y pruebas. El repositorio anterior de agentes y sus ejemplos
de soporte/ventas fueron referencia técnica y permanecen disponibles, pero el
recorrido actual está completo en los cuatro notebooks culturales.

Las fechas, códigos de actividades y consignas operativas de cohortes anteriores
no se interpretan como instrucciones nuevas ni se trasladan a esta entrega.

## Cambio pedagógico

La revisión previa simplificaba el arranque, pero dejaba demasiado trabajo dentro
de helpers y ofrecía pocas decisiones de diseño. La versión actual conserva
instrucciones cortas y pausas, y añade construcción progresiva, contraste de
arquitecturas, fallas controladas, criterios de evaluación y soluciones explicadas.

Cada notebook desarrolla un recorrido completo de aprendizaje. La guía recomienda
observar una primera cohorte y ajustar ejercicios con feedback, sin eliminar pausas
ni cierre. Ejecutar el código verifica funcionamiento, no demuestra comprensión.

## Corpus y ejemplos

Los doce textos de cultural_catalog.json son escenarios didácticos originales.
Usan personajes conocidos como referencia contextual, pero no describen cómics,
episodios ni canciones publicados. Las canciones y sus metadatos son inventados;
no se reproducen letras, audio, imágenes ni páginas de obras. No se requiere
conocimiento previo de ficción para comprender el problema técnico.

La búsqueda es lexical con normalización y alias explícitos. Sus scores no representan
probabilidades ni similitud semántica aprendida. Los filtros y resultados tienen
contratos comprobables, que constituyen el objetivo de la primera clase.

## Correspondencia con las clases originales

| Origen | Desarrollo actual |
|---|---|
| Introducción a agentes y herramientas | Función mínima, contrato, validación, tool call y observación |
| Componentes LangChain y RAG | Prompt, respuesta estructurada, fuentes, estado y dos topologías |
| Sistemas multiagente | Comparación de patrones, paralelo, Send, reducers y agente con herramientas |
| Construcción y pruebas | Corrección acotada, revisión humana, tabla de casos y reportes locales |

No se reinstala ni usa un servicio externo de observabilidad. El `.env` se conserva
privado; se desactivan flags heredados de trazado remoto en la configuración del curso.

## Revisión de octubre de 2026

- **Instalación en VS Code:** se agregó `.vscode/` (intérprete `.venv`, raíz de notebooks,
  extensiones recomendadas), `docs/INSTALACION.md` y un `doctor.py` que detecta la falta de
  la extensión Jupyter, un kernel ajeno al `.venv` y notebooks desincronizados.
- **Causa de las fallas observadas:** VS Code sin extensión Jupyter; notebooks editados
  directamente (con `input()` y gráficos que requieren Internet) que ya no coincidían con
  sus scripts, por lo que la verificación se detenía antes de ejecutar.
- **Modelos:** de `gpt-4.1-mini` a la familia GPT-6 (`gpt-6-luna`, `gpt-6.1-sol`), con
  Responses API, `reasoning_effort` configurable y límite de salida que contempla los
  tokens de razonamiento.
- **Nuevas clases:** 0 (mundo agéntico) y 5 (Deep Agents). La clase 3 suma `create_agent`
  con middleware y un supervisor real con `Command`.
- **Dependencias:** `deepagents` y `grandalf` (dibujo de grafos sin Internet); LangGraph 1.2.14.

## Documentación primaria consultada

- [Workflows y agentes](https://docs.langchain.com/oss/python/langgraph/workflows-agents):
  patrones de secuencia, reparto, agentes y revisión iterativa.
- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api):
  estado, actualizaciones, aristas, reducers y Send.
- [Interrupciones](https://docs.langchain.com/oss/python/langgraph/interrupts):
  interrupt, Command, reanudación y comportamiento del nodo al reiniciar.
- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview):
  `create_deep_agent`, subagentes, sistema de archivos y `interrupt_on`.
- [Agentes de LangChain](https://docs.langchain.com/oss/python/langchain/agents):
  `create_agent` y middleware.
- [Modelos de OpenAI](https://developers.openai.com/api/docs/models): IDs, precios,
  esfuerzo de razonamiento y endpoints de la familia GPT-6 (consultado el 6/10/2026).

Las actividades, datos y ejemplos de código del curso fueron diseñados para este
repositorio. uv.lock fija las versiones usadas para comprobarlos.
