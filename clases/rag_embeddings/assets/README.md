# Figuras de la clase

Cada figura tiene una versión PNG para notebooks y una versión SVG para ampliar
sin perder nitidez. Todos los textos están en español y la información también
aparece con etiquetas o números; el color no es la única señal.

| Archivo | Qué permite explicar |
|---|---|
| `01_flujo_rag` | Diferencia entre indexar y consultar; embeddings y generación cumplen funciones distintas. |
| `02_mapa_manual` | Dirección de los vectores y similitud coseno con coordenadas inventadas. |
| `03_similitudes_manuales` | Matriz de comparaciones calculadas con todos los componentes. |
| `04_ranking_manual` | Orden de recuperación con esos mismos vectores inventados. |

Los números del ejemplo manual son:

| Texto | Componente 1 | Componente 2 |
|---|---:|---:|
| Pregunta: taller de historietas | 0.93 | 0.72 |
| A: taller de cómics | 0.90 | 0.45 |
| B: préstamo de libros | 0.10 | 1.00 |
| C: uso de computadoras | -0.80 | 0.28 |

Estos vectores **no fueron producidos por un modelo**. Sirven para entender una
operación geométrica. La similitud coseno no es una probabilidad de que el texto
recuperado contenga la respuesta.

El módulo `henry_agents.rag_taller_visuales` contiene cinco funciones que devuelven
figuras Matplotlib:

```python
from henry_agents.rag_taller_visuales import (
    dibujar_flujo,
    dibujar_mapa_manual,
    dibujar_similitudes,
    dibujar_ranking,
    dibujar_proyeccion,
)
```

`dibujar_similitudes(etiquetas, vectores)` calcula los cosenos a partir de los
vectores completos. `dibujar_ranking(resultados, titulo)` recibe diccionarios con
`score`, `id` o `fragmento_id`, y `texto`; conserva el orden recibido.

`dibujar_proyeccion(etiquetas, vectores, grupos=None)` reduce los vectores a dos
dimensiones mediante PCA calculado con SVD. No necesita descargar otra librería.
El dibujo muestra cuánta variación conserva y recuerda que la búsqueda debe usar
los vectores completos. La ubicación o cercanía en dos dimensiones puede cambiar
y perder información; los ejes no representan categorías como «cómics» o «libros».

Para regenerar las cuatro figuras desde la raíz del repo:

```bash
uv run python -m henry_agents.rag_taller_visuales
```
