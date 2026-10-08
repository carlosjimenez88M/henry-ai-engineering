"""Sincroniza scripts percent (.py) y notebooks (.ipynb) de clases/.

Por defecto, los .py son la fuente editable y se generan notebooks sin outputs:
    uv run python scripts/sync_notebooks.py

Si editaste un notebook en VS Code y querés conservar esos cambios, primero llevalos al .py:
    uv run python scripts/sync_notebooks.py --desde-notebooks
(los outputs no se copian; después se regeneran los notebooks limpios).

Para una sola carpeta, y dejando los notebooks ejecutados offline con sus salidas:
    uv run python scripts/sync_notebooks.py --solo python_ai --ejecutar
"""

import argparse
import os
from pathlib import Path

import jupytext
import nbformat

root = Path(__file__).resolve().parents[1]


def notebooks_a_scripts(carpeta):
    for notebook in sorted(carpeta.rglob("0*.ipynb")):
        contenido = jupytext.read(notebook)
        # Sin metadatos de notebook ni de celda: el .py queda como lo escribiría una persona.
        contenido.metadata.setdefault("jupytext", {}).update(
            {"notebook_metadata_filter": "-all", "cell_metadata_filter": "-all"}
        )
        jupytext.write(contenido, notebook.with_suffix(".py"), fmt="py:percent")
        print("←", notebook.with_suffix(".py").name)


def scripts_a_notebooks(carpeta, ejecutar=False):
    for source in sorted(carpeta.rglob("0*.py")):
        notebook = jupytext.read(source)
        # VS Code ofrece el .venv del proyecto (ver .vscode/settings.json); "python3" es
        # el nombre genérico que Jupyter también reconoce si se abre con make lab.
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3 (.venv del curso)",
            "language": "python",
            "name": "python3",
        }
        notebook.metadata["language_info"] = {"name": "python", "version": "3.13"}
        if source.parent.name == "rag_embeddings":
            notebook.metadata["kernelspec"] = {
                "display_name": "Henry AI Engineering (.venv)",
                "language": "python",
                "name": "henry-ai-engineering",
            }
        if ejecutar:
            ejecutar_offline(notebook, source.parent)
        nbformat.write(notebook, source.with_suffix(".ipynb"))
        print("→", source.with_suffix(".ipynb").name)


def ejecutar_offline(notebook, carpeta):
    """Ejecuta en un kernel nuevo, offline: las salidas guardadas no gastan API."""
    from nbclient import NotebookClient

    os.environ["COURSE_MODE"] = "offline"
    # Sin marcas de ejecución: reejecutar sin cambios no ensucia el diff.
    NotebookClient(
        notebook, timeout=300, record_timing=False, resources={"metadata": {"path": str(carpeta)}}
    ).execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desde-notebooks", action="store_true")
    parser.add_argument("--solo", help="subcarpeta de clases/, por ejemplo python_ai")
    parser.add_argument("--ejecutar", action="store_true", help="guardar los notebooks con salidas")
    args = parser.parse_args()
    carpeta = root / "clases" / args.solo if args.solo else root / "clases"
    if not carpeta.is_dir():
        parser.error(f"No existe {carpeta}")
    if args.desde_notebooks:
        notebooks_a_scripts(carpeta)
    scripts_a_notebooks(carpeta, ejecutar=args.ejecutar)
