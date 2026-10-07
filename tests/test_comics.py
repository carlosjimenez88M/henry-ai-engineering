"""Contratos, dos feedback loops, SDK estructurado y errores HTTP sin secretos."""

import json
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from henry_agents import comics
from henry_agents.comics_api import crear_api

PEDIDO = {
    "tema": "Un apagón interrumpe la feria de Ciudad Prisma",
    "heroes": ["batman", "spiderman"], "tono": "misterio", "max_vinetas": 4,
}


@pytest.mark.parametrize("cambio", [
    {"tema": "  "}, {"tema": "x" * 501}, {"heroes": []}, {"heroes": ["superman"]},
    {"heroes": ["batman", "batman"]}, {"tono": "violencia"}, {"max_vinetas": 1},
    {"max_vinetas": 7}, {"max_vinetas": True}, {"max_vinetas": 4.0},
    {"max_vinetas": "4"}, {"modelo": "gpt-6-astra"},
])
def test_pedido_rejects_invalid_or_extra_input(cambio):
    with pytest.raises(ValidationError):
        comics.PedidoComic.model_validate({**PEDIDO, **cambio})


def test_source_selection_is_fresh_and_only_contains_requested_characters():
    selected = comics.seleccionar_fuentes(PEDIDO)
    assert {f["heroe"] for f in selected} == set(PEDIDO["heroes"])
    selected[0]["texto"] = "Cambio local"
    assert comics.cargar_fuentes()[0]["texto"] != "Cambio local"
    assert all(f["id"].startswith("COM-") for f in comics.cargar_fuentes())


def test_mutated_model_instances_are_revalidated_before_use():
    pedido = comics.PedidoComic.model_validate(PEDIDO)
    pedido.max_vinetas = True
    with pytest.raises(ValidationError):
        comics.generar_comic(pedido)


def test_offline_two_feedback_loops_apply_changes_then_final_review():
    result = comics.generar_comic(PEDIDO)
    assert result.modo == "offline" and result.estado == "aprobado"
    assert result.llamadas_modelo == 0 and result.uso_tokens.total_tokens == 0
    assert result.uso_tokens.completo and result.uso_tokens.llamadas_con_uso == 0
    assert [r.ciclo for r in result.revisiones] == [1, 2, 3]
    assert [r.aprobada for r in result.revisiones] == [False, False, True]
    assert [e.etapa for e in result.eventos] == [
        "brief", "borrador", "critica", "reescritura", "critica", "reescritura", "critica",
    ]
    assert [e.ciclo for e in result.eventos] == [0, 0, 1, 1, 2, 2, 3]
    assert all(e.origen == "reglas" and e.modelo is None for e in result.eventos)
    assert len(result.guion.vinetas) == PEDIDO["max_vinetas"]
    assert all("Causa:" in v.descripcion and "Consecuencia:" in v.descripcion
               and "Encuadre:" in v.descripcion for v in result.guion.vinetas)
    assert comics.validar_guion(PEDIDO, result.brief, result.guion) == []


def test_three_versions_capture_feedback_and_are_independent_snapshots():
    result = comics.generar_comic(PEDIDO)
    original, primera, segunda = result.versiones
    assert "Causa:" not in original.vinetas[0].descripcion
    assert "Causa:" in primera.vinetas[0].descripcion
    assert "Encuadre:" not in primera.vinetas[0].descripcion
    assert "Encuadre:" in segunda.vinetas[0].descripcion
    assert result.guion == segunda
    assert original != primera and primera != segunda
    primera.vinetas[0].descripcion = "Un cambio posterior exclusivo de la primera revisión."
    primera.vinetas[0].fuentes.append("CAMBIO-LOCAL")
    assert "CAMBIO-LOCAL" not in original.vinetas[0].fuentes
    assert "CAMBIO-LOCAL" not in segunda.vinetas[0].fuentes
    assert "CAMBIO-LOCAL" not in result.guion.vinetas[0].fuentes
    segunda.vinetas[0].descripcion = "Un cambio posterior exclusivo del snapshot final."
    assert result.guion.vinetas[0].descripcion != segunda.vinetas[0].descripcion


@pytest.mark.parametrize("cantidad", [2, 3, 4, 5, 6])
def test_panel_range_and_all_three_characters_remain_valid(cantidad):
    pedido = {**PEDIDO, "heroes": ["batman", "spiderman", "fantasticos"], "max_vinetas": cantidad}
    result = comics.generar_comic(pedido)
    assert len(result.guion.vinetas) == cantidad
    assert len(result.fuentes) == 6
    assert comics.validar_guion(pedido, result.brief, result.guion) == []


def test_public_phases_make_feedback_effect_visible():
    brief = comics.extraer_brief(PEDIDO)
    draft = comics.crear_borrador(PEDIDO, brief)
    critique = comics.criticar_guion(PEDIDO, brief, draft, ciclo=1)
    updated = comics.reescribir_guion(PEDIDO, brief, draft, critique, ciclo=1)
    assert not critique.aprobada
    assert draft.vinetas[0].descripcion != updated.vinetas[0].descripcion
    assert "Causa:" not in draft.vinetas[0].descripcion
    assert "Causa:" in updated.vinetas[0].descripcion
    assert not comics.criticar_guion(PEDIDO, brief, updated, ciclo=2).aprobada
    with pytest.raises(ValueError):
        comics.reescribir_guion(PEDIDO, brief, updated, critique, ciclo=2)


def test_wrong_references_literal_quotes_and_missing_hero_are_detected():
    brief = comics.extraer_brief(PEDIDO)
    forged = brief.model_copy(deep=True)
    forged.fuentes[0].id = "COM-INVENTADA"
    assert comics.validar_brief(PEDIDO, forged)
    forged = brief.model_copy(deep=True)
    forged.fuentes[0].cita_literal = "Este texto no aparece en ninguna fuente."
    assert comics.validar_brief(PEDIDO, forged)
    omitted = brief.model_copy(update={"fuentes": [f for f in brief.fuentes if "BAT" in f.id]})
    assert any("héroe" in error for error in comics.validar_brief(PEDIDO, omitted))
    guion = comics.crear_borrador(PEDIDO, brief)
    guion.vinetas[0].fuentes = ["COM-INVENTADA"]
    assert comics.validar_guion(PEDIDO, brief, guion)


def test_valid_ids_do_not_claim_complete_semantic_fidelity():
    result = comics.generar_comic(PEDIDO)
    # El control de Python demuestra procedencia; no lee toda la verdad del texto.
    result.guion.vinetas[0].descripcion = "Batman usa poderes sobrenaturales inventados para resolver todo."
    assert comics.validar_guion(PEDIDO, result.brief, result.guion) == []
    assert not comics.criticar_guion(PEDIDO, result.brief, result.guion, ciclo=3).aprobada


class ModeloComicDoble:
    """Doble del protocolo include_raw; simula uso, jamás hace una llamada de red."""

    def __init__(self, rechazar_final=False, fuente_inventada=False, uso=True,
                 falla=False, salida_invalida=False):
        self.calls = []
        self.rechazar_final = rechazar_final
        self.fuente_inventada = fuente_inventada
        self.uso = uso
        self.falla = falla
        self.salida_invalida = salida_invalida

    def with_structured_output(self, schema, *, include_raw):
        assert include_raw is True
        parent = self

        class Bound:
            def invoke(self, messages):
                if parent.falla:
                    raise ConnectionError("Falla con secreto sk-super-secreto no publicable")
                assert messages[0][0] == "system" and messages[-1][0] == "human"
                payload = json.loads(messages[-1][1])
                parent.calls.append((schema.__name__, deepcopy(payload)))
                if schema is comics.BriefComic:
                    parsed = comics.extraer_brief(payload["pedido"])
                    if parent.fuente_inventada:
                        parsed.fuentes[0].id = "COM-INVENTADA"
                elif schema is comics.GuionComic and "revision" not in payload:
                    parsed = comics.crear_borrador(payload["pedido"], payload["brief"])
                elif schema is comics.GuionComic:
                    parsed = comics.reescribir_guion(
                        payload["pedido"], payload["brief"], payload["guion"], payload["revision"],
                        ciclo=payload["ciclo"],
                    )
                else:
                    parsed = comics.criticar_guion(
                        payload["pedido"], payload["brief"], payload["guion"], ciclo=payload["ciclo"],
                    )
                    if parent.rechazar_final and payload["ciclo"] == 3:
                        parsed.aprobada = False
                        parsed.observaciones = ["La crítica final no aprobó la fidelidad narrativa"]
                        parsed.cambios_sugeridos = ["Revisar una contradicción antes de publicar"]
                usage = {"input_tokens": 120, "output_tokens": 80, "total_tokens": 200} if parent.uso else None
                return {"parsed": {} if parent.salida_invalida else parsed,
                        "raw": AIMessage(content="", usage_metadata=usage), "parsing_error": None}

        return Bound()


@pytest.fixture
def live_env(monkeypatch):
    # Una clave ficticia satisface el control de configuración. Todos los modelos
    # están inyectados; intentar construir uno real haría fallar estos tests.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-no-real")
    monkeypatch.setattr(comics, "chat_model", lambda *a, **k: pytest.fail("No construir SDK real"))


def test_live_seven_logical_calls_and_raw_usage_are_measured_with_doubles(live_env):
    luna, sol = ModeloComicDoble(), ModeloComicDoble()
    result = comics.generar_comic(PEDIDO, mode="live", modelos={"default": luna, "agent": sol})
    assert result.estado == "aprobado" and result.modo == "live"
    assert result.llamadas_modelo == 7 and len(luna.calls) == 4 and len(sol.calls) == 3
    assert result.uso_tokens.input_tokens == 840
    assert result.uso_tokens.output_tokens == 560 and result.uso_tokens.total_tokens == 1400
    assert result.uso_tokens.completo and result.uso_tokens.llamadas_con_uso == 7
    assert all(e.origen == "modelo" for e in result.eventos)


def test_live_versions_preserve_each_parsed_story_without_extra_calls(live_env):
    luna, sol = ModeloComicDoble(), ModeloComicDoble()
    result = comics.generar_comic(PEDIDO, mode="live", modelos={"default": luna, "agent": sol})
    assert len(sol.calls) == 3 and len(luna.calls) == 4 and result.llamadas_modelo == 7
    assert result.versiones[0].model_dump() == sol.calls[1][1]["guion"]
    assert result.versiones[1].model_dump() == sol.calls[2][1]["guion"]
    assert result.versiones[2].model_dump() == luna.calls[-1][1]["guion"]
    result.versiones[2].vinetas[0].fuentes.append("CAMBIO-LOCAL")
    assert "CAMBIO-LOCAL" not in result.guion.vinetas[0].fuentes


def test_final_negative_review_is_reported_as_requires_revision(live_env):
    model = ModeloComicDoble(rechazar_final=True)
    result = comics.generar_comic(PEDIDO, mode="live", modelos={"default": model, "agent": model})
    assert result.estado == "requiere_revision"
    assert not result.revisiones[-1].aprobada and result.llamadas_modelo == 7


def test_missing_usage_is_unknown_rather_than_fabricated_zero(live_env):
    model = ModeloComicDoble(uso=False)
    result = comics.generar_comic(PEDIDO, mode="live", modelos={"default": model, "agent": model})
    assert not result.uso_tokens.completo and result.uso_tokens.llamadas_con_uso == 0
    assert result.uso_tokens.total_tokens is None
    assert result.uso_tokens.input_tokens is None and result.llamadas_modelo == 7


@pytest.mark.parametrize("opciones", [{"fuente_inventada": True}, {"salida_invalida": True}])
def test_live_forged_or_invalid_response_never_becomes_offline_success(live_env, opciones):
    model = ModeloComicDoble(**opciones)
    with pytest.raises(comics.SalidaComicInvalida) as error:
        comics.generar_comic(PEDIDO, mode="live", modelos={"default": model, "agent": model})
    assert (error.value.etapa, error.value.ciclo) == ("brief", 0)
    assert len(model.calls) == 1


def test_provider_failure_propagates_without_secrets_or_fallback(live_env):
    model = ModeloComicDoble(falla=True)
    with pytest.raises(comics.ProveedorComicError) as error:
        comics.generar_comic(PEDIDO, mode="live", modelos={"default": model, "agent": model})
    assert "sk-super-secreto" not in str(error.value)
    assert (error.value.etapa, error.value.ciclo) == ("brief", 0)


class ModeloFallaEnFase:
    """Introduce un fallo en un punto del flujo completo; no usa red."""

    def __init__(self, invocacion, tipo="proveedor"):
        self.invocacion = invocacion
        self.tipo = tipo
        self.intentos = 0
        self.base = ModeloComicDoble()

    def with_structured_output(self, schema, *, include_raw):
        bound = self.base.with_structured_output(schema, include_raw=include_raw)
        parent = self

        class Bound:
            def invoke(self, messages):
                parent.intentos += 1
                if parent.intentos == parent.invocacion and parent.tipo == "proveedor":
                    raise RuntimeError("Detalle privado: sk-super-secreto y prompt reservado")
                envelope = bound.invoke(messages)
                if parent.intentos == parent.invocacion:
                    if parent.tipo == "schema":
                        envelope["parsed"] = {"secreto": "sk-super-secreto"}
                    elif parent.tipo == "parsing":
                        envelope["parsing_error"] = ValueError("sk-super-secreto")
                    elif parent.tipo == "guion":
                        envelope["parsed"].vinetas[0].fuentes = ["FUENTE-INVENTADA"]
                    elif parent.tipo == "revision":
                        revision = envelope["parsed"]
                        revision.ciclo = 2 if revision.ciclo == 1 else 1
                return envelope

        return Bound()


FASES = [
    (1, "brief", 0), (2, "borrador", 0), (3, "critica", 1),
    (4, "reescritura", 1), (5, "critica", 2), (6, "reescritura", 2), (7, "critica", 3),
]


@pytest.mark.parametrize("invocacion,etapa,ciclo", FASES)
def test_provider_failure_identifies_each_phase_without_original_detail(live_env, invocacion, etapa, ciclo):
    model = ModeloFallaEnFase(invocacion)
    with pytest.raises(comics.ProveedorComicError) as error:
        comics.generar_comic(PEDIDO, mode="live", modelos={"default": model, "agent": model})
    assert (error.value.etapa, error.value.ciclo) == (etapa, ciclo)
    assert model.intentos == invocacion
    assert "sk-super-secreto" not in str(error.value) and "prompt reservado" not in str(error.value)
    assert error.value.__suppress_context__ is True


@pytest.mark.parametrize("tipo", ["schema", "parsing"])
@pytest.mark.parametrize("invocacion,etapa,ciclo", FASES[-2:])
def test_output_validation_retains_late_cycle_context(live_env, tipo, invocacion, etapa, ciclo):
    model = ModeloFallaEnFase(invocacion, tipo)
    with pytest.raises(comics.SalidaComicInvalida) as error:
        comics.generar_comic(PEDIDO, mode="live", modelos={"default": model, "agent": model})
    assert (error.value.etapa, error.value.ciclo) == (etapa, ciclo)
    assert "sk-super-secreto" not in str(error.value)


@pytest.mark.parametrize("invocacion,etapa,ciclo,tipo", [
    (2, "borrador", 0, "guion"), (4, "reescritura", 1, "guion"),
    (6, "reescritura", 2, "guion"), (7, "critica", 3, "revision"),
])
def test_python_controls_retain_phase_context_after_schema_validation(live_env, invocacion, etapa, ciclo, tipo):
    model = ModeloFallaEnFase(invocacion, tipo)
    with pytest.raises(comics.SalidaComicInvalida) as error:
        comics.generar_comic(PEDIDO, mode="live", modelos={"default": model, "agent": model})
    assert (error.value.etapa, error.value.ciclo) == (etapa, ciclo)


def test_model_construction_failure_is_safe_and_identifies_brief(monkeypatch, live_env):
    def construir(*args, **kwargs):
        raise RuntimeError("Configuración privada sk-super-secreto")

    monkeypatch.setattr(comics, "chat_model", construir)
    with pytest.raises(comics.ProveedorComicError) as error:
        comics.generar_comic(PEDIDO, mode="live")
    assert (error.value.etapa, error.value.ciclo) == ("brief", 0)
    assert "sk-super-secreto" not in str(error.value)


def test_structured_binding_failure_is_safe_and_identifies_brief(live_env):
    class NoBinding:
        def with_structured_output(self, *args, **kwargs):
            raise RuntimeError("SDK privado sk-super-secreto")

    with pytest.raises(comics.ProveedorComicError) as error:
        comics.generar_comic(PEDIDO, mode="live", modelos={"default": NoBinding()})
    assert (error.value.etapa, error.value.ciclo) == ("brief", 0)
    assert "sk-super-secreto" not in str(error.value)


def test_absent_server_credentials_do_not_trigger_model_construction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(comics, "chat_model", lambda *a, **k: pytest.fail("No crear modelo sin clave"))
    with pytest.raises(comics.ConfiguracionComicError):
        comics.generar_comic(PEDIDO, mode="live")


def test_api_offline_health_sources_and_response_contract():
    client = TestClient(crear_api(mode="offline"))
    assert client.get("/health").json()["mode"] == "offline"
    assert len(client.get("/fuentes").json()["fuentes"]) == 6
    response = client.post("/comics", json=PEDIDO)
    assert response.status_code == 200
    result = comics.ResultadoComic.model_validate(response.json())
    assert result.estado == "aprobado" and result.llamadas_modelo == 0
    assert len(result.revisiones) == 3


@pytest.mark.parametrize("extra", ["mode", "modelo", "OPENAI_API_KEY", "clave"])
def test_api_rejects_client_config_and_does_not_echo_secrets(extra):
    response = TestClient(crear_api(mode="offline")).post(
        "/comics", json={**PEDIDO, extra: "sk-super-secreto"},
    )
    assert response.status_code == 422
    assert "sk-super-secreto" not in response.text
    assert "extra_forbidden" in response.text


def test_api_invalid_panel_count_returns_422_before_any_model_call(live_env):
    model = ModeloComicDoble()
    client = TestClient(crear_api(mode="live", modelos={"default": model, "agent": model}))
    assert client.post("/comics", json={**PEDIDO, "max_vinetas": True}).status_code == 422
    assert model.calls == []


def test_api_missing_key_and_provider_failure_return_503_without_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # La fábrica y /health funcionan incluso si el servidor live está mal configurado.
    client = TestClient(crear_api(mode="live"))
    assert client.get("/health").status_code == 200
    assert client.post("/comics", json=PEDIDO).status_code == 503
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-no-real")
    broken = ModeloComicDoble(falla=True)
    client = TestClient(crear_api(mode="live", modelos={"default": broken, "agent": broken}))
    response = client.post("/comics", json=PEDIDO)
    assert response.status_code == 503 and "sk-super-secreto" not in response.text


def test_api_invalid_generated_sources_return_502(live_env):
    model = ModeloComicDoble(fuente_inventada=True)
    client = TestClient(crear_api(mode="live", modelos={"default": model, "agent": model}))
    response = client.post("/comics", json=PEDIDO)
    assert response.status_code == 502
    assert response.json()["detail"] == "La salida del modelo no cumple los controles de este servicio."


def test_api_mode_comes_from_server_environment(monkeypatch):
    monkeypatch.setenv("COURSE_MODE", "live")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert crear_api().state.mode == "live"
    with pytest.raises(ValueError):
        crear_api(mode="inventado")
