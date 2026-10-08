"""Diagnóstico real offline y fallos controlados, sin red ni exposición de credenciales."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from henry_agents import rag_taller

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_rag_environment.py"
spec = importlib.util.spec_from_file_location("check_rag_environment", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture(autouse=True)
def prohibir_api_real(monkeypatch):
    def prohibido(*args, **kwargs):
        raise AssertionError("Esta prueba no permite llamadas API reales")
    monkeypatch.setattr(rag_taller, "solicitar_embeddings", prohibido)
    monkeypatch.setattr(rag_taller, "chat_model", prohibido)
    import openai
    monkeypatch.setattr(openai, "OpenAI", prohibido)


def por_id(report, identificador):
    return next(check for check in report["checks"] if check["id"] == identificador)


@pytest.fixture
def offline_rapido(monkeypatch):
    """Los tests de diagnóstico de errores evitan repetir el caso real ya probado."""
    monkeypatch.setattr(checker, "_offline_e2e", lambda *args: {
        "fragmentos_indexados": 14, "solicitudes_api_actuales": 0,
    })


def config_doble(monkeypatch, *, nombre="gpt-6-luna", error=None, clave="credencial_privada_ejemplo"):
    original = checker.importlib.import_module
    def configure(mode):
        assert mode == "live"
        if error is not None:
            raise error
        monkeypatch.setenv("OPENAI_API_KEY", clave)
        return mode
    config = SimpleNamespace(configure=configure, model_name=lambda role: nombre,
                             MODELOS={"gpt-6-luna": {}})
    def importar(nombre_modulo):
        return config if nombre_modulo == "henry_agents.config" else original(nombre_modulo)
    monkeypatch.setattr(checker.importlib, "import_module", importar)
    return config


def test_actual_offline_environment_indexes_and_replays_without_api(tmp_path):
    destino = tmp_path / "reports" / "verification.json"
    report = checker.verificar_entorno(report_path=destino)
    assert report["ok"] and report["modo"] == "offline"
    assert report["env"]["clave_configurada"] is None
    assert report["metadata"]["offline"]["fragmentos_indexados"] == 14
    assert report["metadata"]["offline"]["cobertura_multifuente"]
    assert report["metadata"]["offline"]["solicitudes_api_actuales"] == 0
    assert len(report["versions"]) == 8
    assert por_id(report, "import:panel")["ok"]
    assert json.loads(destino.read_text(encoding="utf-8")) == report


def test_report_paths_are_repo_relative_or_explicit_absolute(tmp_path):
    expected = tmp_path / "reports" / "rag-intro-environment" / "verification.json"
    assert checker.ruta_informe(tmp_path) == expected
    assert checker.ruta_informe(tmp_path, "custom/report.json") == tmp_path / "custom" / "report.json"
    absoluto = tmp_path / "explicit.json"
    assert checker.ruta_informe(Path("/otra_raiz"), absoluto) == absoluto


def test_default_report_can_be_written_under_requested_repository_root(tmp_path, offline_rapido):
    report = checker.verificar_entorno(root=tmp_path)
    assert (tmp_path / "reports" / "rag-intro-environment" / "verification.json").is_file()
    assert not report["ok"] and not por_id(report, "venv")["ok"]


@pytest.mark.parametrize("version", [(3, 10, 14), (3, 14, 0)])
def test_unsupported_python_fails_with_actionable_message(monkeypatch, tmp_path, offline_rapido, version):
    monkeypatch.setattr(checker.sys, "version_info", version)
    report = checker.verificar_entorno(report_path=tmp_path / "report.json")
    assert not report["ok"] and "Python 3.11" in por_id(report, "python")["mensaje"]


@pytest.mark.parametrize("version", [None, "0.9.1", "credencial_privada_ejemplo"])
def test_missing_old_or_invalid_package_version_fails_safely(monkeypatch, tmp_path, offline_rapido, version):
    original = checker.importlib.metadata.version
    def consultar(paquete):
        if paquete == "qdrant-client":
            if version is None:
                raise checker.importlib.metadata.PackageNotFoundError(paquete)
            return version
        return original(paquete)
    monkeypatch.setattr(checker.importlib.metadata, "version", consultar)
    report = checker.verificar_entorno(report_path=tmp_path / "report.json")
    assert not report["ok"] and not por_id(report, "paquete:qdrant-client")["ok"]
    assert "uv sync" in por_id(report, "paquete:qdrant-client")["mensaje"]
    assert "credencial_privada_ejemplo" not in json.dumps(report)


def test_import_error_details_are_not_leaked(monkeypatch, tmp_path, offline_rapido):
    original = checker.importlib.import_module
    def importar(nombre):
        if nombre == checker.MODULOS["panel"]:
            raise RuntimeError("credencial_privada_ejemplo")
        return original(nombre)
    monkeypatch.setattr(checker.importlib, "import_module", importar)
    report = checker.verificar_entorno(report_path=tmp_path / "report.json")
    assert not report["ok"] and not por_id(report, "import:panel")["ok"]
    assert "credencial_privada_ejemplo" not in json.dumps(report)


def test_cache_or_replay_error_is_actionable_and_no_fallback_runs(monkeypatch, tmp_path):
    def fallar(*args):
        raise rag_taller.CacheNoDisponible("credencial_privada_ejemplo")
    monkeypatch.setattr(checker, "_offline_e2e", fallar)
    report = checker.verificar_entorno(report_path=tmp_path / "report.json")
    assert not report["ok"] and not por_id(report, "offline_e2e")["ok"]
    assert "cachés" in por_id(report, "offline_e2e")["mensaje"]
    assert "credencial_privada_ejemplo" not in json.dumps(report)


def test_live_configuration_logs_only_boolean_and_allowlisted_model(monkeypatch, tmp_path, offline_rapido):
    clave = "sk-claveprivada-987654321"
    config_doble(monkeypatch, clave=clave)
    llamadas = []
    monkeypatch.setattr(checker, "_live_e2e", lambda taller: llamadas.append(taller) or {
        "solicitudes_embeddings": 1, "solicitudes_generacion": 1, "cita_materiales_verificada": True,
    })
    report = checker.verificar_entorno(live=True, report_path=tmp_path / "report.json")
    assert report["ok"] and len(llamadas) == 1
    assert report["env"]["clave_configurada"] is True
    assert report["metadata"]["modelo_rag"] == "gpt-6-luna"
    serializado = json.dumps(report)
    assert clave not in serializado and clave[-4:] not in serializado


def test_live_missing_key_or_config_error_does_not_call_provider(monkeypatch, tmp_path, offline_rapido):
    config_doble(monkeypatch, error=ValueError("sk-claveprivada-987654321"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    def no_llamar(*args):
        raise AssertionError("No debe probar live sin configuración")
    monkeypatch.setattr(checker, "_live_e2e", no_llamar)
    report = checker.verificar_entorno(live=True, report_path=tmp_path / "report.json")
    assert not report["ok"] and report["env"]["clave_configurada"] is False
    assert not por_id(report, "env_live")["ok"]
    assert "sk-claveprivada" not in json.dumps(report)


def test_unknown_model_cannot_print_a_secret_disguised_as_model(monkeypatch, tmp_path, offline_rapido):
    secreto = "sk-claveprivada-987654321"
    config_doble(monkeypatch, nombre=secreto)
    report = checker.verificar_entorno(live=True, report_path=tmp_path / "report.json")
    assert not report["ok"] and "modelo_rag" not in report["metadata"]
    assert secreto not in json.dumps(report)


def test_live_failure_has_no_exception_details_and_no_offline_substitution(monkeypatch, tmp_path, offline_rapido):
    config_doble(monkeypatch)
    def fallar(*args):
        raise RuntimeError("sk-claveprivada-987654321")
    monkeypatch.setattr(checker, "_live_e2e", fallar)
    report = checker.verificar_entorno(live=True, report_path=tmp_path / "report.json")
    assert not report["ok"] and not por_id(report, "live_e2e")["ok"]
    assert "live" not in report["metadata"]
    assert "sk-claveprivada" not in json.dumps(report)


def test_live_is_skipped_if_offline_prerequisites_fail(monkeypatch, tmp_path):
    config_doble(monkeypatch)
    def fallar(*args):
        raise ValueError("cache corrupta")
    monkeypatch.setattr(checker, "_offline_e2e", fallar)
    called = []
    monkeypatch.setattr(checker, "_live_e2e", lambda *args: called.append(True))
    report = checker.verificar_entorno(live=True, report_path=tmp_path / "report.json")
    assert not report["ok"] and called == []


def test_keyboard_interrupt_is_not_swallowed(monkeypatch, tmp_path):
    def interrumpir(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(checker, "_offline_e2e", interrumpir)
    with pytest.raises(KeyboardInterrupt):
        checker.verificar_entorno(report_path=tmp_path / "report.json")


def test_unwritable_report_fails_without_printing_exception(monkeypatch, tmp_path, offline_rapido):
    destino = tmp_path / "report.json"
    original = Path.write_text
    def escribir(path, *args, **kwargs):
        if path == destino:
            raise PermissionError("sk-claveprivada-987654321")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", escribir)
    report = checker.verificar_entorno(report_path=destino)
    assert not report["ok"] and not report["informe_guardado"]
    assert "permisos" in por_id(report, "informe")["mensaje"]
    assert "sk-claveprivada" not in json.dumps(report)


def test_cli_offline_exit_code_report_and_output_are_clean(tmp_path, capsys):
    destino = tmp_path / "verification.json"
    assert checker.main(["--report", str(destino)]) == 0
    salida = capsys.readouterr().out
    assert "Entorno listo" in salida
    assert not any(simbolo in salida for simbolo in ["✅", "⚠️", "❌", "💲"])
    assert json.loads(destino.read_text(encoding="utf-8"))["ok"]


def test_cli_failure_returns_one_with_actionable_output(monkeypatch, capsys):
    monkeypatch.setattr(checker, "verificar_entorno", lambda **kwargs: {
        "ok": False, "informe_guardado": True,
        "checks": [{"ok": False, "mensaje": "Ejecuta uv sync --locked."}],
    })
    assert checker.main([]) == 1
    assert "uv sync --locked" in capsys.readouterr().out


def test_live_e2e_calls_embeddings_once_and_generation_once_with_local_qdrant(monkeypatch):
    peticiones = []
    original_vector = rag_taller.vectores_para([rag_taller.CONSULTAS[2]], "text-embedding-3-large")[0]
    # El doble devuelve un vector ya conocido. Esta prueba verifica el cableado, no la semántica.
    def embedding(textos, modelo):
        peticiones.append(("embedding", textos, modelo))
        return {"vectores": [original_vector], "usage": {"prompt_tokens": 9, "total_tokens": 9}}
    def responder(pregunta, hallazgos, mode):
        peticiones.append(("generacion", pregunta, mode))
        assert len(hallazgos) == 4 and all(h.vigente for h in hallazgos)
        return SimpleNamespace(estado="respondido", llamadas_modelo=1,
                               citas=[SimpleNamespace(fragmento_id="CC-04-P2")], uso_tokens=None)
    monkeypatch.setattr(rag_taller, "solicitar_embeddings", embedding)
    monkeypatch.setattr(rag_taller, "responder", responder)
    monkeypatch.setattr(rag_taller, "validar_citas", lambda respuesta, hallazgos: [])
    resultado = checker._live_e2e(rag_taller)
    assert [p[0] for p in peticiones] == ["embedding", "generacion"]
    assert peticiones[0][1] == [checker.PREGUNTA_LIVE]
    assert resultado["solicitudes_embeddings"] == resultado["solicitudes_generacion"] == 1
    assert resultado["tokens_generacion"] is None
