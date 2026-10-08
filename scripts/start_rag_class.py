"""Comprueba el entorno y abre la clase RAG con su kernel del proyecto."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from check_rag_environment import verificar_entorno

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Sólo comprobar y preparar el kernel")
    args = parser.parse_args(argv)
    for variable, nombre in [("IPYTHONDIR", "ipython"), ("MPLCONFIGDIR", "matplotlib"),
                             ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")]:
        carpeta = ROOT / ".local" / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(carpeta))
    report = verificar_entorno()
    if not report["ok"]:
        for check in report["checks"]:
            if not check["ok"]:
                print("[FALLO]", check["mensaje"])
        return 1
    subprocess.run([
        sys.executable, "-m", "ipykernel", "install", "--sys-prefix",
        "--name", "henry-ai-engineering", "--display-name", "Henry AI Engineering (.venv)",
    ], check=True, capture_output=True, text=True)
    print("Entorno comprobado y kernel Henry AI Engineering (.venv) preparado.", flush=True)
    if args.check:
        return 0
    notebook = ROOT / "clases/rag_embeddings/00_rag_y_embeddings.ipynb"
    return subprocess.call([sys.executable, "-m", "jupyterlab", str(notebook)], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
