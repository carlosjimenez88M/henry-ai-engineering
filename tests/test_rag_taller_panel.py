"""Interacciones del explorador con Qdrant real, sin ejecutar APIs de modelos."""

import pytest

from henry_agents import rag_taller, rag_taller_panel, rag_taller_replay
from henry_agents.rag_taller import (
    CONSULTAS,
    CacheNoDisponible,
    abrir_base,
    cargar_documentos,
    crear_indice,
    fragmentar,
    vectores_para,
)
from henry_agents.rag_taller_panel import LARGE, SMALL, crear_explorador


@pytest.fixture(autouse=True)
def impedir_api(monkeypatch):
    def no_permitido(*_args, **_kwargs):
        pytest.fail("El explorador no debe llamar una API ni generar una respuesta nueva")

    monkeypatch.setattr(rag_taller, "solicitar_embeddings", no_permitido)
    monkeypatch.setattr(rag_taller, "chat_model", no_permitido)
    monkeypatch.setattr(rag_taller, "responder", no_permitido)


@pytest.fixture
def base_panel(tmp_path):
    ruta = tmp_path / "qdrant"
    colecciones = {SMALL: "centro_small", LARGE: "centro_large"}
    fragmentos = fragmentar(cargar_documentos())
    textos = [fragmento["texto"] for fragmento in fragmentos]
    with abrir_base(ruta) as base:
        for modelo, coleccion in colecciones.items():
            crear_indice(base, fragmentos, vectores_para(textos, modelo), coleccion, modelo)
    return ruta, colecciones, len(fragmentos)


def controles(panel):
    pregunta = panel.children[1]
    modelo, cantidad, vigentes = panel.children[2].children
    return pregunta, modelo, cantidad, vigentes, panel.children[-1]


def test_camila_recupera_requisitos_y_plazo_sin_api(base_panel):
    ruta, colecciones, _ = base_panel
    panel = crear_explorador(ruta, colecciones)
    pregunta, modelo, cantidad, vigentes, vista = controles(panel)
    assert pregunta.value == CONSULTAS[3]
    assert modelo.value == LARGE and cantidad.value == 4 and vigentes.value
    assert "Cobertura de fuentes: 2 de 2" in vista.value
    assert "CC-01-P1" in vista.value and "CC-01-P2" in vista.value
    assert "Generación anterior reproducida" in vista.value
    assert "Llamadas nuevas a modelos en este panel: 0" in vista.value
    assert "no es una probabilidad" in vista.value


def test_cambiar_k_muestra_fuentes_faltantes_y_no_reutiliza_respuesta(base_panel):
    ruta, colecciones, _ = base_panel
    panel = crear_explorador(ruta, colecciones)
    _, _, cantidad, _, vista = controles(panel)
    cantidad.value = 1
    assert "Cobertura de fuentes: 0 de 2" in vista.value
    assert "Faltan:" in vista.value
    assert "No hay una respuesta guardada" in vista.value
    assert "Generación anterior reproducida" not in vista.value
    cantidad.value = 4
    assert "Cobertura de fuentes: 2 de 2" in vista.value
    assert "Generación anterior reproducida" in vista.value


def test_filtro_excluye_norma_archivada_del_contexto(base_panel):
    ruta, colecciones, _ = base_panel
    panel = crear_explorador(ruta, colecciones)
    pregunta, _, _, vigentes, vista = controles(panel)
    pregunta.value = CONSULTAS[1]
    assert "CC-07-P1" not in vista.value
    vigentes.value = False
    assert "CC-07-P1" in vista.value
    assert "Archivado: revisar su fecha y vigencia" in vista.value
    vigentes.value = True
    assert "CC-07-P1" not in vista.value


def test_mongolia_recupera_candidatos_y_reproduce_abstencion(base_panel):
    ruta, colecciones, _ = base_panel
    panel = crear_explorador(ruta, colecciones)
    pregunta, _, _, _, vista = controles(panel)
    pregunta.value = CONSULTAS[4]
    assert "Pregunta fuera del corpus" in vista.value
    assert vista.value.count("<article>") == 4
    assert "El generador se abstuvo" in vista.value
    assert "No incluye citas" in vista.value
    assert "Llamadas nuevas a modelos en este panel: 0" in vista.value


def test_modelo_consulta_su_coleccion_sin_inventar_snapshot(base_panel, monkeypatch):
    ruta, colecciones, _ = base_panel
    panel = crear_explorador(ruta, colecciones)
    _, modelo, cantidad, _, vista = controles(panel)
    cantidad.value = 1
    buscar_real = rag_taller_panel.buscar
    consultas = []

    def registrar_busqueda(*args, **kwargs):
        consultas.append((args[1], args[3], kwargs["mode"]))
        return buscar_real(*args, **kwargs)

    def no_reproducir(*_args, **_kwargs):
        pytest.fail("Un contexto sin snapshot no debe usar otra respuesta preparada")

    monkeypatch.setattr(rag_taller_panel, "buscar", registrar_busqueda)
    monkeypatch.setattr(rag_taller_panel, "reproducir_respuesta", no_reproducir)
    modelo.value = SMALL
    assert consultas == [(colecciones[SMALL], SMALL, "offline")]
    assert "Qdrant local · Small" in vista.value
    assert "No hay una respuesta guardada" in vista.value
    assert "Generación anterior reproducida" not in vista.value


def test_texto_de_documentos_se_escapa_en_html(base_panel, monkeypatch):
    ruta, colecciones, _ = base_panel
    buscar_real = rag_taller_panel.buscar

    def buscar_con_texto_no_confiable(*args, **kwargs):
        hallazgos = buscar_real(*args, **kwargs)
        hallazgos[0] = hallazgos[0].model_copy(update={
            "titulo": "<script>alert(1)</script>",
            "texto": "<img src=x onerror=alert(1)> & texto de una fuente",
        })
        return hallazgos

    monkeypatch.setattr(rag_taller_panel, "buscar", buscar_con_texto_no_confiable)
    panel = crear_explorador(ruta, colecciones)
    html = panel.children[-1].value
    assert "<script>" not in html and "<img src=" not in html
    assert "&lt;script&gt;" in html and "&lt;img src=" in html and "&amp;" in html
    assert "No hay una respuesta guardada" in html


def test_archivo_corrupto_no_se_oculta_como_snapshot_ausente(base_panel, tmp_path, monkeypatch):
    ruta, colecciones, _ = base_panel
    corrupto = tmp_path / "respuestas_corruptas.json"
    corrupto.write_text("{esto no es JSON}", encoding="utf-8")
    monkeypatch.setattr(rag_taller_replay, "RESPUESTAS_PATH", corrupto)
    with pytest.raises(CacheNoDisponible, match="No se puede leer el JSON"):
        crear_explorador(ruta, colecciones)


def test_error_de_busqueda_se_propaga_y_qdrant_queda_reabrible(base_panel, monkeypatch):
    ruta, colecciones, cantidad = base_panel

    def fallar(*_args, **_kwargs):
        raise RuntimeError("Fallo real de recuperación")

    monkeypatch.setattr(rag_taller_panel, "buscar", fallar)
    with pytest.raises(RuntimeError, match="Fallo real de recuperación"):
        crear_explorador(ruta, colecciones)
    with abrir_base(ruta) as base:
        assert base.count(colecciones[LARGE]).count == cantidad
