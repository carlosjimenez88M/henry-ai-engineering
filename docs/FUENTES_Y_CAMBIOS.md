# Fuentes, decisiones y alcance de la revisión cultural

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

## Documentación primaria consultada

- [Workflows y agentes](https://docs.langchain.com/oss/python/langgraph/workflows-agents):
  patrones de secuencia, reparto, agentes y revisión iterativa.
- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api):
  estado, actualizaciones, aristas, reducers y Send.
- [Interrupciones](https://docs.langchain.com/oss/python/langgraph/interrupts):
  interrupt, Command, reanudación y comportamiento del nodo al reiniciar.

Las actividades, datos y ejemplos de código del curso fueron diseñados para este
repositorio. uv.lock fija las versiones usadas para comprobarlos.
