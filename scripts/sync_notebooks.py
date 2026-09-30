"""Los scripts percent son la fuente editable; genera notebooks sin outputs."""

from pathlib import Path

import jupytext
import nbformat

root = Path(__file__).resolve().parents[1]
for source in sorted((root / "clases").glob("0*.py")):
    notebook = jupytext.read(source)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (Henry M3)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.13"}
    nbformat.write(notebook, source.with_suffix(".ipynb"))
    print(source.with_suffix(".ipynb").name)
