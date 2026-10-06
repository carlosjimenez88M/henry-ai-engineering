"""Diagnóstico del entorno del curso, pensado para personas que recién empiezan.

Uso:  uv run python scripts/doctor.py          (sin costo)
      uv run python scripts/doctor.py --live   (además hace una llamada mínima a OpenAI)
"""

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
problemas = []


def ok(mensaje):
    print(f"  ✅ {mensaje}")


def aviso(mensaje, solucion):
    print(f"  ⚠️  {mensaje}\n      → {solucion}")


def falla(mensaje, solucion):
    problemas.append(mensaje)
    print(f"  ❌ {mensaje}\n      → {solucion}")


def revisar_python():
    print("\n1. Python y entorno virtual")
    version = sys.version_info
    if (3, 11) <= version[:2] <= (3, 13):
        ok(f"Python {version.major}.{version.minor}.{version.micro}")
    else:
        falla(f"Python {version.major}.{version.minor} no es compatible", "Ejecutá: uv sync")
    esperado = ROOT / ".venv"
    if Path(sys.prefix).resolve() == esperado.resolve():
        ok(f"Usando el entorno del curso: {esperado}")
    else:
        falla(
            f"Este Python no es el del curso ({sys.executable})",
            "Ejecutá los comandos con 'uv run ...' desde la carpeta del curso.",
        )


def revisar_paquetes():
    print("\n2. Librerías")
    for paquete in [
        "langchain",
        "langgraph",
        "langchain-openai",
        "deepagents",
        "ipykernel",
        "henry-modulo3",
    ]:
        try:
            ok(f"{paquete} {importlib.metadata.version(paquete)}")
        except importlib.metadata.PackageNotFoundError:
            falla(f"Falta {paquete}", "Ejecutá: uv sync")
    try:
        import henry_agents.agentic  # noqa: F401

        ok("El paquete del curso (henry_agents) se importa correctamente")
    except Exception as error:  # Mostrar el tipo ayuda a diagnosticar sin exponer secretos.
        falla(f"No se pudo importar henry_agents ({type(error).__name__})", "Ejecutá: uv sync")


def revisar_env():
    print("\n3. Configuración (.env)")
    archivo = ROOT / ".env"
    if not archivo.exists() or not archivo.read_text(encoding="utf-8").strip():
        aviso(
            ".env no existe o está vacío: se usará el modo offline (gratis)",
            "Para live: copiá .env.example como .env y completá OPENAI_API_KEY.",
        )
    from henry_agents.config import MODELOS_OPENAI, configure, model_name

    try:
        modo = configure()
    except ValueError as error:
        falla(str(error), "Revisá COURSE_MODE y OPENAI_API_KEY en .env")
        return
    ok(f"Modo: {modo}")
    clave = os.getenv("OPENAI_API_KEY", "")
    if clave:
        ok(f"OPENAI_API_KEY configurada (termina en …{clave[-4:]})")
    else:
        aviso("OPENAI_API_KEY vacía: solo modo offline", "No hace falta para practicar.")
    for rol in ("default", "agent", "rag", "embeddings"):
        nombre = model_name(rol)
        conocidos = ({"text-embedding-3-large", "text-embedding-3-small"}
                     if rol == "embeddings" else MODELOS_OPENAI)
        if nombre in conocidos:
            ok(f"Modelo ({rol}): {nombre}")
        else:
            aviso(f"Modelo ({rol}) desconocido para el curso: {nombre}", "Revisá .env.example")


def extensiones_vscode():
    """Lista extensiones instaladas usando el comando 'code' o la carpeta de extensiones."""
    for comando in ("code", "cursor"):
        if shutil.which(comando):
            try:
                salida = subprocess.run(
                    [comando, "--list-extensions"], capture_output=True, text=True, timeout=30
                )
                if salida.returncode == 0 and salida.stdout.strip():
                    return {e.lower() for e in salida.stdout.split()}
            except (OSError, subprocess.SubprocessError):
                pass
    encontradas = set()
    for carpeta in (Path.home() / ".vscode/extensions", Path.home() / ".cursor/extensions"):
        indice = carpeta / "extensions.json"  # Índice que mantiene el propio editor.
        if indice.exists():
            try:
                for extension in json.loads(indice.read_text(encoding="utf-8")):
                    encontradas.add(extension["identifier"]["id"].lower())
            except (ValueError, KeyError, TypeError):
                pass
        if carpeta.is_dir():
            for item in carpeta.iterdir():
                # Carpetas con formato publicador.nombre-versión[-plataforma].
                coincidencia = re.match(r"^([\w-]+\.[\w-]+?)-\d+\.\d+", item.name.lower())
                if coincidencia:
                    encontradas.add(coincidencia.group(1))
    return encontradas


def revisar_vscode():
    print("\n4. VS Code")
    instaladas = extensiones_vscode()
    if not instaladas:
        aviso(
            "No pude ver las extensiones de VS Code",
            "En VS Code instalá 'Python' y 'Jupyter' (Microsoft) desde Extensiones.",
        )
    else:
        for extension, nombre in [("ms-python.python", "Python"), ("ms-toolsai.jupyter", "Jupyter")]:
            if extension in instaladas:
                ok(f"Extensión {nombre} instalada")
            else:
                falla(
                    f"Falta la extensión {nombre} ({extension}); sin ella los notebooks no corren",
                    f"VS Code → Extensiones (Ctrl/Cmd+Shift+X) → buscar '{extension}' → Instalar",
                )
    if (ROOT / ".vscode/settings.json").exists():
        ok("Configuración del proyecto para VS Code presente (.vscode/settings.json)")
    else:
        aviso("Falta .vscode/settings.json", "Restauralo desde el repositorio.")
    print("      Recordá abrir en VS Code ESTA carpeta (File → Open Folder):", ROOT.name)


def revisar_notebooks():
    print("\n5. Clases")
    import jupytext
    import nbformat

    for script in sorted((ROOT / "clases").rglob("0*.py")):
        notebook = script.with_suffix(".ipynb")
        if not notebook.exists():
            falla(f"Falta {notebook.name}", "Ejecutá: uv run python scripts/sync_notebooks.py")
            continue
        esperadas = [(c.cell_type, c.source) for c in jupytext.read(script).cells]
        actuales = [(c.cell_type, c.source) for c in nbformat.read(notebook, as_version=4).cells]
        if esperadas == actuales:
            ok(notebook.name)
        else:
            aviso(
                f"{notebook.name} difiere de su script {script.name}",
                "Si editaste el notebook, pasá los cambios al .py; luego sync_notebooks.py",
            )


def probar_live():
    print("\n6. Llamada mínima a OpenAI (consume una fracción de centavo)")
    from henry_agents.config import chat_model

    try:
        respuesta = chat_model(reasoning_effort="low", max_tokens=200).invoke(
            "Respondé solo con la palabra: listo"
        )
        ok(f"Respuesta: {respuesta.text.strip()!r} | tokens: {respuesta.usage_metadata}")
    except Exception as error:
        falla(
            f"La llamada falló: {type(error).__name__}",
            "Revisá la clave, el saldo de la cuenta y el nombre del modelo en .env",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="probar una llamada real a OpenAI")
    args = parser.parse_args()
    print("Diagnóstico del curso ·", ROOT)
    revisar_python()
    revisar_paquetes()
    revisar_env()
    revisar_vscode()
    revisar_notebooks()
    if args.live:
        probar_live()
    print()
    if problemas:
        print(f"Hay {len(problemas)} problema(s) para resolver. Ver docs/INSTALACION.md.")
        return 1
    print("Todo listo. Empieza en clases/agentic_workflows/00_python_para_empezar.ipynb")
    print("y elige el kernel .venv. Para la ruta avanzada: clases/00_mundo_agentico.ipynb.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
