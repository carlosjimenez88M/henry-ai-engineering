"""Verifica scripts y notebooks; conserva evidencia nueva incluso si una ejecución falla."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import jupytext
import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def fingerprints(root):
    """Huella de las entradas que afectan la ejecución; nunca incluye .env."""
    paths = [root / "pyproject.toml", root / "uv.lock", root / ".python-version"]
    for directory, suffixes in [
        ("src", {".py", ".json"}),
        ("scripts", {".py"}),
        ("clases", {".py", ".ipynb"}),
    ]:
        paths.extend(p for p in (root / directory).rglob("*") if p.suffix in suffixes)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if p.is_file()
    }


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def validate_pairs(root):
    sources = sorted((root / "clases").glob("0*.py"))
    if len(sources) != 4:
        raise ValueError("Se esperaban exactamente cuatro scripts de clase")
    for source in sources:
        notebook = nbformat.read(source.with_suffix(".ipynb"), as_version=4)
        script = jupytext.read(source)
        if [(c.cell_type, c.source) for c in notebook.cells] != [
            (c.cell_type, c.source) for c in script.cells
        ]:
            raise ValueError(f"Notebook desactualizado: {source.stem}; ejecutá make notebooks")
    return sources


def execute_script(source, root, output, mode):
    env = {**os.environ, "COURSE_MODE": mode}
    with (output / f"{source.stem}.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(source)],
            cwd=root,
            env=env,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
        )


def execute_notebook(source, root, output, mode):
    kernels = output / "kernels" / "henry-verify"
    kernels.mkdir(parents=True, exist_ok=True)
    (kernels / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Henry verification",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    manager = KernelSpecManager(kernel_dirs=[str(kernels.parent)])
    km = AsyncKernelManager(kernel_name="henry-verify", kernel_spec_manager=manager)
    notebook = nbformat.read(source.with_suffix(".ipynb"), as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="henry-verify",
        km=km,
        resources={"metadata": {"path": str(root)}},
    )
    env = {
        **os.environ,
        "COURSE_MODE": mode,
        "JUPYTER_PLATFORM_DIRS": "1",
        "IPYTHONDIR": str(output / "ipython"),
        "JUPYTER_RUNTIME_DIR": str(output / "runtime"),
    }
    try:
        client.execute(env=env)
    finally:
        # Conservar celdas ya ejecutadas y el error para poder diagnosticar la falla.
        nbformat.write(notebook, output / source.with_suffix(".ipynb").name)
        if km.has_kernel:
            asyncio.run(km.shutdown_kernel(now=True))


def run_verification(root, mode):
    root = Path(root).resolve()
    if mode not in {"offline", "live"}:
        raise ValueError("Modo inválido")
    output = root / "reports" / mode
    output.mkdir(parents=True, exist_ok=True)
    path = output / "verification.json"
    report = {
        "run_id": str(uuid4()),
        "mode": mode,
        "status": "running",
        "started_at": timestamp(),
        "finished_at": None,
        "python": platform.python_version(),
        "platform": platform.system(),
        "versions": {},
        "input_sha256": {},
        "results": [],
    }
    # Invalidar un éxito anterior ANTES de comenzar cualquier ejecución.
    save_report(path, report)
    current = None
    try:
        report["versions"] = {
            p: importlib.metadata.version(p)
            for p in ["langgraph", "langchain-core", "langchain-openai", "nbclient"]
        }
        report["input_sha256"] = fingerprints(root)
        sources = validate_pairs(root)
        report["results"] = [
            {"class": s.stem, "script": "pending", "notebook": "pending"} for s in sources
        ]
        save_report(path, report)
        for source, current in zip(sources, report["results"], strict=True):
            start = time.perf_counter()
            for phase, execute in [("script", execute_script), ("notebook", execute_notebook)]:
                current[phase] = "running"
                save_report(path, report)
                execute(source, root, output, mode)
                current[phase] = "passed"
                save_report(path, report)
                print("PASS", phase, source.name, flush=True)
            current["seconds"] = round(time.perf_counter() - start, 2)
        if report["input_sha256"] != fingerprints(root):
            raise RuntimeError(
                "El código cambió durante la verificación; repetir sobre una versión estable"
            )
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        if current:
            for phase in ("script", "notebook"):
                if current[phase] == "running":
                    current[phase] = "failed"
        # El detalle está en los logs/notebook, no en mensajes que puedan mostrar credenciales.
        print(f"FAIL {type(error).__name__}; ver {output}", file=sys.stderr)
    finally:
        report["finished_at"] = timestamp()
        save_report(path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    return 0 if run_verification(root, args.mode)["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
