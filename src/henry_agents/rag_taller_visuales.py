"""Figuras de la clase introductoria de embeddings y RAG.

Las funciones devuelven figuras Matplotlib: se pueden mostrar en un notebook
o exportar sin conexión a Internet. El mapa manual usa coordenadas inventadas.
Las otras funciones reciben los vectores y resultados calculados en la clase.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AZUL = "#176A91"
VIOLETA = "#6941A5"
NARANJA = "#AA4D0C"
TINTA = "#243348"
GRIS = "#526176"
PALETA = [AZUL, VIOLETA, NARANJA, "#197859", "#AD3659", "#5A6070"]


def mostrar_figura(figura: Figure) -> None:
    """Muestra un PNG explícito, incluso con backend Agg, y libera la figura."""
    from IPython.display import Image, display

    try:
        with BytesIO() as imagen:
            figura.savefig(imagen, format="png", dpi=150,
                            bbox_inches="tight", facecolor="white")
            display(Image(data=imagen.getvalue(), format="png"))
    finally:
        plt.close(figura)


def _figura(ancho: float, alto: float) -> tuple[Figure, plt.Axes]:
    figura, eje = plt.subplots(figsize=(ancho, alto), layout="constrained")
    figura.patch.set_facecolor("white")
    eje.set_facecolor("white")
    eje.tick_params(labelsize=11, colors=TINTA)
    return figura, eje


def _vectores_validos(etiquetas: Sequence[str], vectores: Sequence) -> np.ndarray:
    matriz = np.asarray(vectores, dtype=float)
    if matriz.ndim != 2 or len(matriz) < 2 or matriz.shape[1] < 2:
        raise ValueError("Se necesitan al menos dos vectores de dos o más componentes.")
    if len(etiquetas) != len(matriz):
        raise ValueError("Debe haber una etiqueta por vector.")
    if not np.isfinite(matriz).all():
        raise ValueError("Los vectores deben contener números finitos.")
    if np.any(np.linalg.norm(matriz, axis=1) == 0):
        raise ValueError("Un vector cero no tiene similitud coseno definida.")
    return matriz


def dibujar_flujo() -> Figure:
    """Distingue indexación, recuperación y generación, con el mismo embedding."""
    figura, eje = _figura(14, 7.3)
    eje.set(xlim=(0, 14), ylim=(0, 7.3))
    eje.axis("off")
    eje.text(0.2, 7.04, "RAG: buscar evidencia antes de redactar", fontsize=20,
             color=TINTA, weight="bold", va="top")
    eje.text(0.2, 6.51, "Dos recorridos conectados por una base de datos vectorial",
             fontsize=13, color=GRIS)

    def caja(x: float, y: float, texto: str, color: str = AZUL) -> None:
        eje.add_patch(FancyBboxPatch((x - 0.92, y - 0.54), 1.84, 1.08,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    linewidth=1.8, edgecolor=color, facecolor="#F7F9FC"))
        eje.text(x, y, texto, ha="center", va="center", fontsize=11.5,
                 color=TINTA, linespacing=1.35)

    def flecha(x1: float, y1: float, x2: float, y2: float, color: str = AZUL) -> None:
        eje.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, color=color, linewidth=1.8))

    eje.text(0.2, 5.91, "1. INDEXAR: preparar el material una vez y actualizarlo si cambia",
             fontsize=12.5, weight="bold", color=AZUL)
    for x, texto, color in [
        (1.15, "Documentos\ncon una fuente", AZUL),
        (3.65, "Fragmentos\n+ identificadores", AZUL),
        (6.15, "Modelo de\nembeddings", VIOLETA),
        (8.65, "Qdrant\nvectores + texto\n+ metadatos", AZUL),
    ]:
        caja(x, 4.9, texto, color)
    for x in (1.15, 3.65, 6.15):
        flecha(x + 0.98, 4.9, x + 1.49, 4.9)
    eje.text(10.12, 4.92, "El embedding es una lista\nde números que representa\nun texto para compararlo.",
             fontsize=12, color=TINTA, va="center", linespacing=1.5)

    eje.text(0.2, 3.52, "2. CONSULTAR: responder una pregunta",
             fontsize=12.5, weight="bold", color=AZUL)
    posiciones = [1.15, 3.65, 6.15, 8.65, 11.15, 13.15]
    # La última salida se dibuja sin caja para mantener el espacio legible.
    for x, texto, color in [
        (1.15, "Pregunta\ndel estudiante", AZUL),
        (3.65, "MISMO modelo\nde embeddings", VIOLETA),
        (6.15, "Buscar vecinos\n+ filtros", AZUL),
        (8.65, "Contexto\ncon fragmentos", AZUL),
        (11.15, "Modelo de\ngeneración", NARANJA),
    ]:
        caja(x, 2.49, texto, color)
    for anterior, siguiente in zip(posiciones[:4], posiciones[1:5], strict=True):
        flecha(anterior + 0.98, 2.49, siguiente - 0.98, 2.49)
    flecha(12.13, 2.49, 12.55, 2.49, NARANJA)
    eje.text(13.05, 2.49, "Respuesta\n+ citas", fontsize=11.5,
             color=TINTA, ha="center", va="center", linespacing=1.5)
    # Qdrant abastece la búsqueda: conexión por debajo de la fila de indexación.
    eje.plot([8.65, 8.65, 6.15], [4.33, 3.99, 3.99], color=AZUL, linewidth=1.5)
    flecha(6.15, 3.99, 6.15, 3.06)
    eje.text(6.38, 3.82, "consulta los vectores guardados", fontsize=10.5, color=AZUL)

    eje.text(0.2, 1.3, "Mismo espacio vectorial", fontsize=12.5,
             color=VIOLETA, weight="bold")
    eje.text(0.2, 0.92, "Documentos y preguntas usan el mismo modelo, dimensión y configuración.",
             fontsize=11.5, color=TINTA)
    eje.text(0.2, 0.46, "El modelo de generación recibe texto recuperado; el ranking por sí solo no redacta una respuesta.",
             fontsize=11.5, color=TINTA)
    return figura


def dibujar_mapa_manual() -> Figure:
    """Ejemplo inventado para observar dirección y coseno, no embeddings reales."""
    figura, eje = _figura(11, 7.2)
    figura.get_layout_engine().set(rect=(0, 0.065, 1, 0.935))
    puntos = {
        "Pregunta: taller de historietas": (0.93, 0.72),
        "A: taller de cómics": (0.9, 0.45),
        "B: préstamo de libros": (0.1, 1.0),
        "C: uso de computadoras": (-0.8, 0.28),
    }
    for i, (etiqueta, punto) in enumerate(puntos.items()):
        color = PALETA[i]
        eje.add_patch(FancyArrowPatch((0, 0), punto, arrowstyle="-|>",
                                     mutation_scale=17, color=color, linewidth=2,
                                     linestyle="--" if i == 0 else "-"))
        eje.scatter(*punto, s=65, color=color, zorder=3)
        desplazamientos = [(10, 11), (8, -24), (8, 7), (-8, 12)]
        eje.annotate(etiqueta, punto, xytext=desplazamientos[i],
                     textcoords="offset points", fontsize=11.5, color=color,
                     ha="right" if i == 3 else "left")
    eje.axhline(0, color="#AFBAC7", linewidth=1)
    eje.axvline(0, color="#AFBAC7", linewidth=1)
    eje.set(xlim=(-1.25, 1.63), ylim=(-0.35, 1.25),
            xlabel="Componente 1 inventada", ylabel="Componente 2 inventada")
    eje.set_aspect("equal", adjustable="box")
    eje.set_title("Coordenadas inventadas para entender el coseno", fontsize=17,
                  color=TINTA, weight="bold", pad=38)
    eje.text(0.5, 1.035, "La pregunta y A apuntan en una dirección parecida.",
             transform=eje.transAxes, ha="center", fontsize=12, color=GRIS)
    eje.grid(alpha=0.16)
    figura.text(0.5, 0.012,
                "Estos números no salen de un modelo. El coseno mide dirección; no es probabilidad de una respuesta correcta.",
                ha="center", fontsize=10.5, color=TINTA)
    return figura


def dibujar_similitudes(etiquetas: Sequence[str], vectores: Sequence) -> Figure:
    """Mapa de cosenos calculados con los vectores completos, sin proyección."""
    matriz = _vectores_validos(etiquetas, vectores)
    normalizados = matriz / np.linalg.norm(matriz, axis=1, keepdims=True)
    similitudes = np.clip(normalizados @ normalizados.T, -1, 1)
    cantidad = len(matriz)
    figura, eje = _figura(max(9, cantidad * 0.85), max(7, cantidad * 0.7))
    imagen = eje.imshow(similitudes, vmin=-1, vmax=1, cmap="BrBG")
    labels = [fill(str(etiqueta), width=19) for etiqueta in etiquetas]
    eje.set_xticks(range(cantidad), labels, rotation=35, ha="right", fontsize=11)
    eje.set_yticks(range(cantidad), labels, fontsize=11)
    for fila in range(cantidad):
        for columna in range(cantidad):
            valor = similitudes[fila, columna]
            eje.text(columna, fila, f"{valor:.2f}", ha="center", va="center", fontsize=11,
                     color="white" if abs(valor) > 0.65 else TINTA)
    barra = figura.colorbar(imagen, ax=eje, shrink=0.78, pad=0.03)
    barra.set_label("Similitud coseno (−1 a 1)", fontsize=12, color=TINTA)
    barra.ax.tick_params(labelsize=11)
    eje.set_title("Comparar textos usando todos sus componentes\n"
                  "Un valor alto indica direcciones parecidas; no demuestra una relación verdadera.",
                  fontsize=14, color=TINTA, pad=20)
    return figura


def dibujar_ranking(resultados: Sequence[Mapping], titulo: str = "Resultados de la búsqueda") -> Figure:
    """Ranking de cosenos; conserva el orden recibido y acepta resultados Qdrant."""
    if not resultados:
        figura, eje = _figura(11, 4)
        eje.axis("off")
        eje.text(0.5, 0.55, "No se recuperaron fragmentos.", transform=eje.transAxes,
                 ha="center", fontsize=18, color=TINTA)
        eje.text(0.5, 0.35, "Revisa la pregunta, los filtros y el umbral.",
                 transform=eje.transAxes, ha="center", fontsize=12, color=GRIS)
        return figura
    etiquetas, valores = [], []
    for resultado in resultados:
        carga = resultado.get("payload") or {}
        identificador = resultado.get("fragmento_id", resultado.get("id", carga.get("fragmento_id", "Sin ID")))
        texto = str(resultado.get("texto", carga.get("texto", "")))
        etiquetas.append(fill(f"{identificador}: {texto[:82]}", width=42))
        valor = float(resultado["score"])
        if not np.isfinite(valor) or not -1.00001 <= valor <= 1.00001:
            raise ValueError("Este gráfico espera scores coseno entre −1 y 1.")
        valores.append(float(np.clip(valor, -1, 1)))
    figura, eje = _figura(11, max(5.2, len(resultados) * 1.12))
    eje.barh(range(len(valores)), valores, color=AZUL, height=0.56)
    eje.set_yticks(range(len(valores)), etiquetas, fontsize=11.5)
    eje.invert_yaxis()
    eje.axvline(0, color=GRIS, linewidth=1)
    minimo = min(0, min(valores) - 0.15)
    eje.set_xlim(minimo, 1.16)
    for fila, valor in enumerate(valores):
        eje.text(valor + (0.025 if valor >= 0 else -0.025), fila, f"{valor:.3f}",
                 va="center", ha="left" if valor >= 0 else "right", fontsize=12, color=TINTA)
    eje.set_xlabel("Similitud coseno: no es un porcentaje de certeza", fontsize=12, color=TINTA)
    eje.set_title(titulo, fontsize=17, weight="bold", color=TINTA, pad=16)
    eje.grid(axis="x", alpha=0.16)
    return figura


def dibujar_proyeccion(etiquetas: Sequence[str], vectores: Sequence,
                       grupos: Sequence[str] | None = None) -> Figure:
    """PCA mediante SVD de los vectores centrados; visualización, no búsqueda."""
    matriz = _vectores_validos(etiquetas, vectores)
    if grupos is not None and len(grupos) != len(matriz):
        raise ValueError("Debe haber un grupo por vector.")
    centrados = matriz - matriz.mean(axis=0)
    izquierda, singular, _ = np.linalg.svd(centrados, full_matrices=False)
    proyeccion = izquierda[:, :2] * singular[:2]
    energia_total = float(np.sum(singular ** 2))
    energia_visible = float(np.sum(singular[:2] ** 2))
    proporcion = energia_visible / energia_total if energia_total > 0 else 0
    figura, eje = _figura(11, 7.3)
    figura.get_layout_engine().set(rect=(0, 0.065, 1, 0.935))
    etiquetas_grupo = list(dict.fromkeys(grupos)) if grupos is not None else ["Textos"]
    grupos_asignados = list(grupos) if grupos is not None else ["Textos"] * len(matriz)
    for indice, grupo in enumerate(etiquetas_grupo):
        indices = [i for i, valor in enumerate(grupos_asignados) if valor == grupo]
        eje.scatter(proyeccion[indices, 0], proyeccion[indices, 1],
                    s=95, color=PALETA[indice % len(PALETA)], label=str(grupo),
                    edgecolor="white", linewidth=1, zorder=3)
    for indice, etiqueta in enumerate(etiquetas):
        # Alternar las etiquetas evita superposiciones en conjuntos pequeños.
        offset = (8, 10 if indice % 2 == 0 else -18)
        eje.annotate(fill(str(etiqueta), 24), proyeccion[indice], xytext=offset,
                     textcoords="offset points", fontsize=11, color=TINTA,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    eje.margins(0.23)
    eje.axhline(0, color="#B2BCC7", linewidth=0.8)
    eje.axvline(0, color="#B2BCC7", linewidth=0.8)
    eje.grid(alpha=0.14)
    eje.set_xlabel("Primer componente principal (sin significado asignado)", fontsize=11.5)
    eje.set_ylabel("Segundo componente principal", fontsize=11.5)
    eje.set_title("Una proyección de los embeddings a dos dimensiones\n"
                  f"Las dos dimensiones conservan {proporcion:.1%} de la variación de este conjunto.",
                  fontsize=15, color=TINTA, pad=18)
    if grupos is not None:
        eje.legend(title="Grupo", fontsize=11, title_fontsize=11, loc="best")
    figura.text(0.5, 0.012,
                "La proyección pierde información. El ranking se calcula con los vectores completos, no con este dibujo.",
                ha="center", fontsize=10.5, color=TINTA)
    return figura


def exportar_figuras_ejemplo(destino: str | Path | None = None) -> list[Path]:
    """Exporta assets para usar antes de ejecutar código, sin API ni descargas."""
    carpeta = (Path(destino) if destino is not None else
               Path(__file__).resolve().parents[2] / "clases" / "rag_embeddings" / "assets")
    carpeta.mkdir(parents=True, exist_ok=True)
    vectores = np.array([[0.93, 0.72], [0.9, 0.45], [0.1, 1.0], [-0.8, 0.28]])
    etiquetas = ["Pregunta", "A: cómics", "B: libros", "C: computadoras"]
    normalizados = vectores / np.linalg.norm(vectores, axis=1, keepdims=True)
    scores = normalizados[1:] @ normalizados[0]
    resultados = sorted([
        {"id": letra, "texto": texto, "score": float(score)}
        for letra, texto, score in zip(
            ["A", "B", "C"],
            ["Taller de cómics", "Préstamo de libros", "Uso de computadoras"],
            scores, strict=True)
    ], key=lambda item: item["score"], reverse=True)
    figuras = {
        "01_flujo_rag": dibujar_flujo(),
        "02_mapa_manual": dibujar_mapa_manual(),
        "03_similitudes_manuales": dibujar_similitudes(etiquetas, vectores),
        "04_ranking_manual": dibujar_ranking(resultados, "Ranking con coordenadas inventadas"),
    }
    archivos = []
    for nombre, figura in figuras.items():
        for extension in ("svg", "png"):
            ruta = carpeta / f"{nombre}.{extension}"
            figura.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
            archivos.append(ruta)
        plt.close(figura)
    return archivos


if __name__ == "__main__":
    for archivo in exportar_figuras_ejemplo():
        print(archivo.name)
