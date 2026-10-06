# Proyecto · Preparar un pedido para revisión

La tienda ficticia La Esquina recibe pedidos estructurados. Necesita consultar
precio, stock y cobertura y preparar una cotización para que una persona decida.
Trabajas con un solo producto por pedido. Los barrios y montos son inventados.
La entrega es una propuesta; no se conecta con WhatsApp, no reserva stock ni cobra.

## Archivos

- [`starter.py`](starter.py): tu punto de partida. Se ejecuta, pero aún no resuelve el proyecto.
- [`solution.py`](solution.py): solución de referencia para revisar después del intento.
- [`tienda.json`](../../src/henry_agents/data/tienda.json): catálogo y política inventados.
- [`workflows_casos.json`](../../src/henry_agents/data/workflows_casos.json): diez casos con expectativas.
- [`workflows.py`](../../src/henry_agents/workflows.py): funciones reutilizables estudiadas en clase.

## Fase 1 · Herramientas y contrato

1. Lee el catálogo y señala el identificador, precio, stock y fuente de cada producto.
2. Prueba `validar_pedido`: tres campos exactos, cantidad entera de 1 a 20 y textos no vacíos.
3. Prueba `cotizar`: devuelve un total con `Decimal` o un estado que explica por qué se detiene.
4. Implementa `construir_plan(pedido)` en la plantilla: producto siempre, entrega solo
   cuando el barrio no es Retiro. Puedes reutilizar funciones del curso si explicas su contrato.
5. Define qué comprobará el evaluador: total observado, fuentes exactas y aprobación requerida.

Ejemplo de entrada y salida esperada antes de decidir:

```python
pedido = {"sku": "arroz", "cantidad": 2, "barrio": "Centro"}
# Subtotal 3.60, envío 2.00, total 5.60. Estado: pendiente; mensaje: None.
```

No necesitas usar un LLM para esta fase. Si escribes un redactor con modelo como
extensión, separa sus pruebas de estas verificaciones locales y registra llamadas reales.

## Fase 2 · Conectar el workflow

Implementa `resolver(pedido, decision="pendiente")`:

1. Valida toda la entrada y el plan antes de ejecutar workers.
2. Ejecuta los workers permitidos y conserva sus observaciones.
3. Cotiza con el mismo catálogo. Si falta producto, stock o cobertura, conserva ese
   estado y detente sin mensaje, incluso si `decision="aprobar"`.
4. Revisa un borrador estructurado con máximo de tres intentos. Si no cumple,
   devuelve `limite_alcanzado` sin entregar un borrador como si estuviera aprobado.
5. Aplica la decisión: `pendiente`, `aprobar` o `rechazar`. Un valor distinto debe
   producir `ValueError`. Solo una aprobación válida permite un mensaje con total y fuentes.
6. Devuelve al menos `estado`, `cotizacion`, `mensaje` y `accion_externa=False`.

En esta simulación cada llamada recalcula la propuesta; no hay reanudación durable.
Puedes elegir cadena o plan variable, pero justifica la elección. El router de
mensajes es un ejercicio previo opcional: este proyecto recibe datos estructurados.

## Ejecutar y comprobar

Desde la raíz del repo:

```bash
uv run python proyectos/tienda_workflows/starter.py
uv run python scripts/evaluate_workflows.py --implementation proyectos/tienda_workflows/starter.py --output reports/mi-proyecto.json
```

Al principio el segundo comando devuelve **0/10 y código de salida 1**: la plantilla
está pendiente, no es un error de instalación. Conforme implementes, subirá el resultado.
El reporte muestra por caso qué contrato no se cumplió. No captura ni mide llamadas
a API de implementaciones propias; `llamadas_api=null` significa no medidas.

Para comparar después del intento:

```bash
uv run python scripts/evaluate_workflows.py --implementation proyectos/tienda_workflows/solution.py --output reports/solucion-proyecto.json
```

Esperado: **10/10**. Añade dos casos tuyos y conserva su evidencia por separado
en tu entrega; la tabla de referencia es pequeña y no demuestra calidad universal.

## Qué entregar

- Tu script completo con la función `resolver`.
- Diagrama con entrada, decisiones, dependencias y condiciones de parada.
- Reporte de los diez casos y evidencia de dos adicionales.
- Un ejemplo de aprobación, uno de rechazo y uno sin stock o cobertura.
- Una explicación de tu arquitectura y un límite que tu solución todavía tenga.

## Rúbrica

| Dimensión | Puntos | Evidencia |
|---|---:|---|
| Herramientas y contratos | 25 | Entrada inválida rechazada, precio decimal, stock y fuentes |
| Composición del flujo | 25 | Diagrama corresponde al código; plan completo y workers permitidos |
| Revisión y límites | 20 | Feedback observable, falla acotada y ningún borrador inválido aprobado |
| Decisión humana | 15 | Pendiente y rechazo sin mensaje; sin acciones externas |
| Evaluación y explicación | 15 | Diez casos, dos propios y una limitación argumentada |

Puedes explicar oralmente, por escrito o con un diagrama acompañado del código.
No se premia terminar primero ni usar más agentes. La revisión de campos no prueba
que un texto generado sea fiel a la evidencia; esa comprobación sería una extensión.
