"""Sincroniza scripts percent (.py) y notebooks (.ipynb) de clases/.

Por defecto, los .py son la fuente editable y se generan notebooks sin outputs:
    uv run python scripts/sync_notebooks.py

Si editaste un notebook en VS Code y querés conservar esos cambios, primero llevalos al .py:
    uv run python scripts/sync_notebooks.py --desde-notebooks
(los outputs no se copian; después se regeneran los notebooks limpios).
"""

import argparse
from pathlib import Path

import jupytext
import nbformat

root = Path(__file__).resolve().parents[1]


def notebooks_a_scripts():
    for notebook in sorted((root / "clases").rglob("0*.ipynb")):
        contenido = jupytext.read(notebook)
        # Sin metadatos de notebook ni de celda: el .py queda como lo escribiría una persona.
        contenido.metadata.setdefault("jupytext", {}).update(
            {"notebook_metadata_filter": "-all", "cell_metadata_filter": "-all"}
        )
        jupytext.write(contenido, notebook.with_suffix(".py"), fmt="py:percent")
        print("←", notebook.with_suffix(".py").name)


def scripts_a_notebooks():
    for source in sorted((root / "clases").rglob("0*.py")):
        notebook = jupytext.read(source)
        # VS Code ofrece el .venv del proyecto (ver .vscode/settings.json); "python3" es
        # el nombre genérico que Jupyter también reconoce si se abre con make lab.
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3 (.venv del curso)",
            "language": "python",
            "name": "python3",
        }
        notebook.metadata["language_info"] = {"name": "python", "version": "3.13"}
        nbformat.write(notebook, source.with_suffix(".ipynb"))
        print("→", source.with_suffix(".ipynb").name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desde-notebooks", action="store_true")
    if parser.parse_args().desde_notebooks:
        notebooks_a_scripts()
    scripts_a_notebooks()
