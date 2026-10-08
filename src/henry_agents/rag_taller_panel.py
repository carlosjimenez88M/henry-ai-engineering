"""Explorador visual de recuperación RAG para Jupyter, sin llamadas a una API.

Cada cambio consulta Qdrant local con embeddings reales ya guardados. La respuesta
se reproduce sólo cuando pregunta y contexto coinciden con una generación anterior.
Cambiar el contexto no ejecuta un modelo ni inventa una respuesta de reemplazo.
"""

from collections.abc import Mapping
from html import escape
from pathlib import Path

import ipywidgets as widgets
from IPython.display import HTML

from henry_agents.rag_taller import CONSULTAS, CacheNoDisponible, abrir_base, buscar
from henry_agents.rag_taller_replay import (
    firma_contexto,
    leer_respuestas,
    reproducir_respuesta,
)

SMALL = "text-embedding-3-small"
LARGE = "text-embedding-3-large"
FUENTES_ESPERADAS = {
    CONSULTAS[0]: {"CC-01-P1", "CC-01-P2"},
    CONSULTAS[1]: {"CC-01-P1"},
    CONSULTAS[2]: {"CC-05-P1"},
    CONSULTAS[3]: {"CC-01-P1", "CC-01-P2"},
}

ESTILOS = """
<style>
.rag-explorador { color: #243348; background: white; padding: 12px;
  font-family: system-ui, sans-serif; font-size: 14px; line-height: 1.5; }
.rag-explorador h3 { font-size: 17px; margin: 0 0 8px; }
.rag-explorador h4 { font-size: 14px; margin: 0 0 6px; }
.rag-explorador p { margin: 6px 0; }
.rag-explorador code { color: #243348; background: #eef2f6; padding: 1px 4px; }
.rag-explorador .rag-resumen { padding: 10px; background: #f4f7fa;
  border-left: 4px solid #176a91; margin-bottom: 12px; }
.rag-explorador .rag-columnas { display: grid; grid-template-columns: minmax(260px, 1fr)
  minmax(360px, 1.6fr); gap: 18px; }
.rag-explorador .rag-ranking { list-style: none; padding: 0; margin: 8px 0; }
.rag-explorador .rag-ranking li { padding: 7px 0; border-bottom: 1px solid #dce3eb; }
.rag-explorador .rag-barra-fila { display: flex; align-items: center; gap: 8px; }
.rag-explorador .rag-barra { position: relative; height: 13px; background: #eaf0f5;
  flex: 1; min-width: 85px; border: 1px solid #c3cfda; }
.rag-explorador .rag-barra::after { content: ''; position: absolute; left: 50%;
  top: 0; bottom: 0; width: 1px; background: #526176; }
.rag-explorador .rag-relleno { position: absolute; top: 0; bottom: 0; }
.rag-explorador .rag-score { font-variant-numeric: tabular-nums; min-width: 46px; }
.rag-explorador .rag-vigente { color: #176449; font-weight: 600; }
.rag-explorador .rag-archivado { color: #8c420d; font-weight: 600; }
.rag-explorador .rag-contexto { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px; }
.rag-explorador article { padding: 10px; border: 1px solid #cdd7e2;
  border-radius: 5px; overflow-wrap: anywhere; }
.rag-explorador article p { font-size: 13px; }
.rag-explorador .rag-nota { font-size: 13px; color: #526176; }
.rag-explorador .rag-respuesta { border-top: 2px solid #cdd7e2;
  padding-top: 12px; margin-top: 16px; }
.rag-explorador .rag-aviso { padding: 12px; border-left: 4px solid #8c420d;
  background: #fff7ef; }
.rag-explorador q { overflow-wrap: anywhere; }
@media (max-width: 780px) {
  .rag-explorador .rag-columnas, .rag-explorador .rag-contexto { grid-template-columns: 1fr; }
}
</style>
"""


def _texto(valor):
    return escape(str(valor), quote=True)


def _ranking_html(hallazgos):
    filas = []
    for puesto, hallazgo in enumerate(hallazgos, 1):
        score = min(1.0, max(-1.0, hallazgo.score))
        izquierda = 50 if score >= 0 else 50 + score * 50
        ancho = abs(score) * 50
        clase = "rag-vigente" if hallazgo.vigente else "rag-archivado"
        estado = "Vigente" if hallazgo.vigente else "Archivado"
        color = "#176a91" if hallazgo.vigente else "#8c420d"
        filas.append(
            f'<li><strong>{puesto}. {_texto(hallazgo.id)}</strong> '
            f'<span class="{clase}">{estado}</span>'
            '<div class="rag-barra-fila">'
            f'<div class="rag-barra" role="img" aria-label="Coseno {hallazgo.score:.3f}">'
            f'<span class="rag-relleno" style="left:{izquierda:.2f}%;'
            f'width:{ancho:.2f}%;background:{color}"></span></div>'
            f'<span class="rag-score">{hallazgo.score:.3f}</span></div></li>'
        )
    return (
        '<section aria-label="Ranking de recuperación"><h3>1. Resultados de Qdrant</h3>'
        '<p class="rag-nota">Escala de coseno: −1 a la izquierda, 0 en el centro y 1 a la derecha. '
        'Un score no es una probabilidad de que la respuesta sea correcta.</p>'
        f'<ol class="rag-ranking">{"".join(filas)}</ol></section>'
    )


def _contexto_html(hallazgos):
    tarjetas = []
    for hallazgo in hallazgos:
        estado = "Vigente" if hallazgo.vigente else "Archivado: revisar su fecha y vigencia"
        clase = "rag-vigente" if hallazgo.vigente else "rag-archivado"
        tarjetas.append(
            f'<article><h4><code>{_texto(hallazgo.id)}</code> {_texto(hallazgo.titulo)}</h4>'
            f'<p class="{clase}">{estado}</p><p>{_texto(hallazgo.texto)}</p></article>'
        )
    return (
        '<section aria-label="Contexto recuperado"><h3>2. Lee el contexto</h3>'
        '<p class="rag-nota">Estas son las piezas de texto que recibiría el generador, '
        'en el mismo orden del ranking.</p>'
        f'<div class="rag-contexto">{"".join(tarjetas)}</div></section>'
    )


def _cobertura_html(pregunta, hallazgos):
    esperadas = FUENTES_ESPERADAS.get(pregunta)
    if esperadas is None:
        return (
            '<p><strong>Pregunta fuera del corpus:</strong> los documentos del centro cultural '
            'no contienen la capital de Mongolia. Recuperar candidatos no significa tener evidencia.</p>'
        )
    presentes = {hallazgo.id for hallazgo in hallazgos}
    recuperadas = esperadas & presentes
    faltantes = esperadas - presentes
    ids_esperados = ", ".join(sorted(esperadas))
    aviso = (
        f' Faltan: <strong>{_texto(", ".join(sorted(faltantes)))}</strong>.'
        if faltantes else " Se recuperaron todas las fuentes esperadas."
    )
    return (
        f'<p><strong>Cobertura de fuentes: {len(recuperadas)} de {len(esperadas)}.</strong>'
        f' Fuentes esperadas por el docente: <code>{_texto(ids_esperados)}</code>.{aviso}</p>'
        '<p class="rag-nota">Esto comprueba presencia de IDs; debes leer el texto para comprobar '
        'que realmente sostiene la respuesta.</p>'
    )


def _respuesta_html(pregunta, hallazgos):
    # Un archivo corrupto o un manifiesto inválido es un error real: no se captura.
    preparacion = leer_respuestas()
    firma = firma_contexto(pregunta, hallazgos)
    if firma not in preparacion["entradas"]:
        return (
            '<section class="rag-respuesta" aria-label="Respuesta preparada">'
            '<h3>3. ¿Puede reproducirse una respuesta?</h3><div class="rag-aviso">'
            '<strong>No hay una respuesta guardada para esta pregunta y este contexto exacto.</strong>'
            '<p>Cambiar k, el modelo o el filtro puede cambiar los fragmentos. La respuesta anterior '
            'ya no se aplica: primero lee el contexto. Este panel no llama al generador.</p>'
            '<p>Para ver la generación preparada, prueba Large, k = 4 y sólo fuentes vigentes.</p>'
            '<p><strong>Llamadas nuevas a modelos: 0.</strong></p></div></section>'
        )
    try:
        respuesta = reproducir_respuesta(pregunta, hallazgos)
    except CacheNoDisponible:
        # La firma sí estaba guardada: una inconsistencia no equivale a ausencia de snapshot.
        raise
    citas = "".join(
        f'<li><code>{_texto(cita.fragmento_id)}</code>: '
        f'<q>{_texto(cita.cita_literal)}</q></li>' for cita in respuesta.citas
    )
    estado = (
        "El generador se abstuvo: no encontró evidencia suficiente"
        if respuesta.estado == "sin_evidencia" else "Respuesta con fuentes del contexto"
    )
    detalle_citas = f'<ul>{citas}</ul>' if citas else '<p>No incluye citas.</p>'
    return (
        '<section class="rag-respuesta" aria-label="Respuesta preparada">'
        '<h3>3. Generación anterior reproducida</h3>'
        f'<p><strong>{estado}.</strong></p><p>{_texto(respuesta.respuesta)}</p>'
        f'{detalle_citas}<p class="rag-nota">Modelo de la preparación: '
        f'<code>{_texto(respuesta.modelo_usado)}</code>. '
        f'Preparación: {_texto(preparacion["generado_utc"])}. '
        '<strong>Llamadas nuevas a modelos en este panel: 0.</strong></p></section>'
    )


def crear_explorador(ruta_db: str | Path, colecciones: Mapping[str, str]) -> widgets.VBox:
    """Devuelve controles y vista de RAG; necesita ambas colecciones ya indexadas.

    Pregunta, modelo, cantidad de fragmentos y vigencia disparan una búsqueda local
    nueva. Se puede probar sin un navegador cambiando ``value`` en los controles y
    leyendo ``panel.children[-1].value``. Ninguna ruta de este módulo llama una API.
    """
    if any(not isinstance(colecciones.get(modelo), str) or not colecciones[modelo].strip()
           for modelo in (SMALL, LARGE)):
        raise ValueError("colecciones debe asociar Small y Large con sus colecciones Qdrant.")
    ruta_db = Path(ruta_db)
    pregunta = widgets.Dropdown(
        options=[(f"{indice + 1}. {texto}", texto) for indice, texto in enumerate(CONSULTAS)],
        value=CONSULTAS[3], description="Pregunta:",
        layout=widgets.Layout(width="100%"), style={"description_width": "85px"},
    )
    modelo = widgets.Dropdown(
        options=[("Small · 1536 componentes", SMALL), ("Large · 3072 componentes", LARGE)],
        value=LARGE, description="Modelo:", layout=widgets.Layout(width="295px"),
        style={"description_width": "70px"},
    )
    cantidad = widgets.IntSlider(
        value=4, min=1, max=6, step=1, description="Fragmentos k:", continuous_update=False,
        layout=widgets.Layout(width="270px"), style={"description_width": "100px"},
    )
    vigentes = widgets.Checkbox(
        value=True, description="Sólo fuentes vigentes", indent=False,
        layout=widgets.Layout(width="200px"),
    )
    vista = widgets.HTML(layout=widgets.Layout(width="100%"))

    def actualizar(_cambio=None):
        with abrir_base(ruta_db) as base:
            hallazgos = buscar(
                base, colecciones[modelo.value], pregunta.value, modelo.value,
                k=cantidad.value, vigentes=vigentes.value, mode="offline",
            )
        # Las excepciones de recuperación se propagan; el contexto ya cerró Qdrant.
        nombre_modelo = "Small" if modelo.value == SMALL else "Large"
        filtro = "sólo vigentes" if vigentes.value else "incluye documentos archivados"
        contenido = (
            ESTILOS + '<div class="rag-explorador">'
            '<div class="rag-resumen">'
            f'<p><strong>Pregunta:</strong> {_texto(pregunta.value)}</p>'
            f'<p>Qdrant local · {_texto(nombre_modelo)} · k = {cantidad.value} · {filtro}.</p>'
            f'{_cobertura_html(pregunta.value, hallazgos)}</div>'
            f'<div class="rag-columnas">{_ranking_html(hallazgos)}{_contexto_html(hallazgos)}</div>'
            f'{_respuesta_html(pregunta.value, hallazgos)}</div>'
        )
        vista.value = HTML(contenido).data

    for control in (pregunta, modelo, cantidad, vigentes):
        control.observe(actualizar, names="value")
    controles = widgets.HBox(
        [modelo, cantidad, vigentes],
        layout=widgets.Layout(flex_flow="row wrap", width="100%"),
    )
    introduccion = widgets.HTML(
        '<p><strong>Explorador RAG.</strong> Cambia una variable y observa qué documentos llegan '
        'al contexto. Cada cambio consulta Qdrant; usa embeddings guardados y no llama una API.</p>'
    )
    panel = widgets.VBox([introduccion, pregunta, controles, vista])
    actualizar()
    return panel
