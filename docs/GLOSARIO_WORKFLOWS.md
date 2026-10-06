# Glosario para empezar

| Término | En palabras simples | En nuestro caso |
|---|---|---|
| IA | Familia de técnicas para tareas como reconocer patrones o generar contenido | No toda automatización del curso usa IA |
| LLM | Modelo de lenguaje que genera una respuesta a partir del contexto recibido | Clasificación opcional de la clase 03 |
| Prompt | Instrucción junto con los datos que recibe el modelo | Clasificar un mensaje en tres áreas |
| Token | Unidad de texto usada por el modelo, a menudo parte de una palabra | Entrada y salida afectan consumo de API |
| API | Interfaz para pedir un servicio desde otro programa | Consultar un modelo remoto |
| Clave de API | Credencial privada para acceder al servicio | Se guarda en `.env`, no en un notebook |
| Herramienta / tool | Función que el programa ejecuta y cuyo contrato controla | `cotizar` consulta datos y calcula dinero |
| Tool calling | El modelo propone nombre y argumentos; el programa valida y ejecuta | Se estudia en el recorrido ampliado |
| Workflow | Pasos conectados con dependencias y condiciones | Validar → cotizar → redactar |
| Cadena / chaining | El resultado de un paso alimenta al siguiente | La redacción usa la cotización |
| Routing | Seleccionar la ruta que atenderá una entrada | Productos, entregas o humano |
| Worker | Componente que realiza una subtarea | Consultar cobertura; puede ser una función |
| Orquestador | Componente que organiza tareas y reúne resultados | Decide consultar entrega cuando no hay retiro |
| Concurrencia | Solapar tareas en un mismo período | Consultar catálogo y cobertura con hilos |
| Future | Objeto que representa un resultado pendiente | `.result()` entrega datos o propaga un error |
| Evaluador | Revisa contra criterios explícitos | Total y referencias observadas |
| Optimizador | Modifica una propuesta según feedback | Guion que corrige el borrador |
| Agente con LLM | Ciclo donde el modelo decide la próxima acción dentro de límites | Modelo → herramienta → observación → modelo |
| Estado | Información que describe el avance actual | `pendiente`, `sin_stock`, `aprobado` |
| Abstención | Reconocer que no hay datos suficientes para responder | Producto fuera del catálogo |
| RAG | Recuperar evidencia y usarla para generar una respuesta | Se construye después, en la ruta ampliada |
| Embedding | Representación numérica aprendida para comparar significado aproximado | No se necesita en la ruta inicial |
| Kernel | Proceso de Python que guarda y ejecuta las celdas de un notebook | Reiniciar borra variables de la sesión |
| Entorno virtual | Python y librerías aislados para este proyecto | Carpeta `.venv` |
| Lockfile | Lista exacta de versiones y sus dependencias | `uv.lock`, usado con `uv sync --locked` |
| Caso de prueba | Entrada y resultado esperado para comprobar un comportamiento | Sin stock debe detenerse aunque se apruebe |

Los anglicismos se mantienen cuando ayudan a reconocerlos en documentación técnica.
Puedes usar su equivalente en español al explicar tu trabajo. Una tienda de barrio
puede llamarse almacén, bodega o abarrotería; los ejercicios no dependen de una variante
regional. Los montos en USD son inventados y no describen precios locales ni conversiones.
