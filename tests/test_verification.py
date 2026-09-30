"""No confundir resultados viejos con la verificación del código actual."""

import importlib.util
import json
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import jupytext
import nbformat
import pytest


@pytest.fixture
def verifier():
    source = Path(__file__).resolve().parents[1] / "scripts/verify.py"
    spec = importlib.util.spec_from_file_location("course_verifier", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "clases").mkdir()
    for index in range(1, 5):
        script = tmp_path / "clases" / f"0{index}_demo.py"
        script.write_text('# %%\nprint("ok")\n')
        nbformat.write(jupytext.read(script), script.with_suffix(".ipynb"))
    return tmp_path


def test_new_failure_replaces_old_pass_and_preserves_phase(verifier, sample_repo, monkeypatch):
    output = sample_repo / "reports/offline"
    output.mkdir(parents=True)
    (output / "verification.json").write_text('{"status":"passed","run_id":"old"}')
    monkeypatch.setattr(verifier, "execute_script", lambda *args: None)

    def fail(*args):
        raise subprocess.CalledProcessError(1, "simulated-kernel")

    monkeypatch.setattr(verifier, "execute_notebook", fail)
    report = verifier.run_verification(sample_repo, "offline")
    saved = json.loads((output / "verification.json").read_text())
    assert saved == report and saved["run_id"] != "old"
    assert saved["status"] == "failed" and saved["finished_at"]
    assert saved["results"][0]["script"] == "passed"
    assert saved["results"][0]["notebook"] == "failed"
    assert saved["results"][1]["script"] == "pending"


def test_stale_notebook_fails_before_execution(verifier, sample_repo, monkeypatch):
    called = []
    monkeypatch.setattr(verifier, "execute_script", lambda *args: called.append(True))
    (sample_repo / "clases/01_demo.py").write_text('# %%\nprint("changed")\n')
    report = verifier.run_verification(sample_repo, "offline")
    assert report["status"] == "failed"
    assert report["error_type"] == "ValueError"
    assert called == []


def test_success_has_fingerprints_and_all_eight_executions(verifier, sample_repo, monkeypatch):
    monkeypatch.setattr(verifier, "execute_script", lambda *args: None)
    monkeypatch.setattr(verifier, "execute_notebook", lambda *args: None)
    report = verifier.run_verification(sample_repo, "offline")
    assert report["status"] == "passed"
    assert len(report["input_sha256"]) == 8
    assert len(report["results"]) == 4
    assert all(r["script"] == r["notebook"] == "passed" for r in report["results"])


def test_changes_during_run_invalidate_result(verifier, sample_repo, monkeypatch):
    monkeypatch.setattr(verifier, "execute_script", lambda *args: None)

    def change(source, *args):
        source.write_text(source.read_text() + "\n# changed during execution\n")

    monkeypatch.setattr(verifier, "execute_notebook", change)
    report = verifier.run_verification(sample_repo, "offline")
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"


def test_missing_distribution_invalidates_previous_success(verifier, sample_repo, monkeypatch):
    output = sample_repo / "reports/offline"
    output.mkdir(parents=True)
    (output / "verification.json").write_text('{"status":"passed","run_id":"old"}')

    def missing(package):
        raise PackageNotFoundError(package)

    monkeypatch.setattr(verifier.importlib.metadata, "version", missing)
    report = verifier.run_verification(sample_repo, "offline")
    saved = json.loads((output / "verification.json").read_text())
    assert saved == report and saved["run_id"] != "old"
    assert saved["status"] == "failed" and saved["finished_at"]
    assert saved["error_type"] == "PackageNotFoundError"
