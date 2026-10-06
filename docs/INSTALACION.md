# Instalación paso a paso en VS Code

Esta guía no supone experiencia previa. Seguila en orden; cada paso dice cómo comprobar
que salió bien. Necesitas Internet para la primera instalación. Después, la ruta
inicial de workflows se puede ejecutar sin red ni cuenta de API.

**Qué vas a instalar:**

| Programa | Para qué |
|---|---|
| VS Code | El editor donde abrimos y ejecutamos los notebooks |
| Extensiones **Python** y **Jupyter** de VS Code | Sin **Jupyter**, VS Code no puede ejecutar notebooks (`.ipynb`) |
| `uv` | Descarga la versión correcta de Python y todas las librerías del curso |

No hace falta instalar Python por separado: `uv` se encarga. Si ya tenés otro Python
(por ejemplo, de Homebrew o de python.org), no importa: el curso usa el suyo propio.

---

## 1. Instalar VS Code

Descargalo de <https://code.visualstudio.com> e instalalo como cualquier programa.

## 2. Instalar las extensiones Python y Jupyter

1. Abrí VS Code.
2. Abrí el panel de extensiones: `Cmd + Shift + X` (Mac) o `Ctrl + Shift + X` (Windows).
3. Buscá **Python** (de Microsoft) → **Install**.
4. Buscá **Jupyter** (de Microsoft) → **Install**.

✅ **Comprobación:** las dos aparecen en la lista de extensiones instaladas.

> Al abrir la carpeta del curso, VS Code también puede sugerir "instalar las extensiones
> recomendadas". Aceptá: son estas mismas.

## 3. Instalar uv

Abrí una terminal (Mac: aplicación **Terminal**; Windows: **PowerShell**) y pegá:

**Mac / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Cerrá la terminal y abrí una nueva** (así reconoce el comando nuevo).

✅ **Comprobación:** `uv --version` muestra un número de versión.

## 4. Abrir la carpeta del curso en VS Code

Si todavía no tienes el repo, abre
[henry-ai-engineering en GitHub](https://github.com/carlosjimenez88M/henry-ai-engineering),
selecciona **Code → Download ZIP** y descomprime el archivo. Abre la carpeta que
contiene `pyproject.toml`, `uv.lock` y `README.md`; puede terminar en `-master`.
Si trabajas con la copia local preparada por tu docente, utiliza esa misma carpeta.

1. **File → Open Folder…** (Archivo → Abrir carpeta).
2. Elegí la carpeta **`henry-ai-engineering`**. No una carpeta superior ni `clases/`.
3. Si VS Code pregunta si confiás en los autores de la carpeta, respondé que sí.

> Abrir la carpeta correcta es importante: ahí está `.vscode/settings.json`, que le indica
> a VS Code qué Python usar.

## 5. Crear el entorno del curso

En VS Code: **Terminal → New Terminal**. En la terminal que aparece abajo, escribí:

```bash
uv sync --locked
```

La primera vez descarga Python 3.13 y las librerías (puede tardar unos minutos).
Se crea una carpeta `.venv`: es el **entorno virtual** del curso.

✅ **Comprobación:** aparece la carpeta `.venv` en el explorador de archivos de VS Code.

## 6. Ejecutar el diagnóstico

```bash
uv run python scripts/doctor.py
```

Muestra una lista con ✅, ⚠️ y ❌. Los ⚠️ son avisos (por ejemplo, "no hay clave de
OpenAI": no hace falta para practicar). Cada ❌ incluye cómo resolverlo.

✅ **Comprobación:** al final dice "Todo listo".

## 7. Abrir el primer notebook y elegir el kernel

1. En el explorador, abrí `clases/agentic_workflows/00_python_para_empezar.ipynb`.
2. Arriba a la derecha hacé clic en **Select Kernel** (Seleccionar kernel).
3. Elegí **Python Environments…** → **`.venv (Python 3.13…)`** de esta carpeta.
4. Ejecutá la primera celda de código con **Shift + Enter**.

✅ **Comprobación:** la celda muestra la versión de Python, una ruta de entorno que
termina en `.venv` y "La Esquina — tienda ficticia".

**Listo. Sigue [la ruta inicial](../clases/agentic_workflows/README.md) en orden.**
Después puedes continuar el recorrido ampliado de LangGraph y Deep Agents desde
`clases/00_mundo_agentico.ipynb`.

---

## Opcional: modo live con OpenAI

Solo si tenés una clave de la API de OpenAI (consume saldo de tu cuenta).

1. Abrí el archivo `.env` (si no existe, copiá `.env.example` y renombralo como `.env`).
2. Completá `OPENAI_API_KEY=sk-...` y cambiá `COURSE_MODE=live`.
3. Guardá el archivo y **reiniciá el kernel** del notebook (botón **Restart**).
4. Probá la conexión con una llamada mínima:

   ```bash
   uv run python scripts/doctor.py --live
   ```

Modelos configurados por defecto (se cambian en `.env`):

| Variable | Valor | Uso |
|---|---|---|
| `OPENAI_MODEL` | `gpt-6-luna` | Actividades breves y especialistas |
| `OPENAI_MODEL_AGENT` | `gpt-6.1-sol` | Coordinadores y Agentic RAG |
| `OPENAI_MODEL_RAG` | `gpt-6-luna` | Generación RAG y revisión de fidelidad |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Experimento semántico opcional |
| `OPENAI_REASONING_EFFORT` | `low` | Cuánto "piensa" el modelo; más alto = más costo |

**Nunca compartas el `.env` ni lo subas a Git**: contiene tu clave.
Los modelos se verificaron el 6 de octubre de 2026; consulta [MODELOS.md](MODELOS.md).
Las banderas live de los notebooks iniciales 03, 08 y 09 vienen apagadas. Para una
comparación RAG real, ejecuta `uv run python scripts/evaluate_rag.py --mode live --case RAG-06`.

---

## Problemas frecuentes

| Lo que ves | Por qué pasa | Cómo se resuelve |
|---|---|---|
| El `.ipynb` se abre como texto/JSON, o no hay botones para ejecutar | Falta la extensión **Jupyter** | Paso 2. Después, recargá VS Code (`Cmd/Ctrl + Shift + P` → *Developer: Reload Window*) |
| En **Select Kernel** no aparece `.venv` | No se ejecutó `uv sync`, o se abrió otra carpeta | Paso 4 y 5. Luego `Cmd/Ctrl + Shift + P` → *Python: Select Interpreter* → *Enter interpreter path* → `.venv/bin/python` (Windows: `.venv\Scripts\python.exe`) |
| VS Code pide "instalar ipykernel" | Elegiste un Python que no es el del curso | Cancelá y elegí el kernel `.venv` (ya trae ipykernel) |
| `ModuleNotFoundError: No module named 'henry_agents'` (o `langgraph`) | El notebook usa otro Python | Cambiá el kernel a `.venv` y reiniciá el kernel |
| `uv: command not found` / "uv no se reconoce" | La terminal se abrió antes de instalar uv | Cerrá y abrí la terminal (o VS Code completo) |
| Mac: `Broken Python installation, platform.mac_ver() returned an empty value` | El Python de Homebrew del sistema está dañado | No lo uses: `uv` trae su propio Python. Elegí el kernel `.venv` |
| `uv sync` falla con errores de certificado o de red (redes institucionales con proxy) | El proxy intercepta HTTPS | Probá `uv sync --native-tls`; si sigue, pedí a soporte acceso a `pypi.org` y `files.pythonhosted.org` |
| Cambié el `.env` y no pasa nada | El kernel cargó la configuración al iniciar | Reiniciá el kernel |
| `Falta OPENAI_API_KEY en .env` | `COURSE_MODE=live` sin clave | Completá la clave o volvé a `COURSE_MODE=offline` |
| Error 401 / 429 en live | Clave inválida, sin saldo o demasiadas solicitudes | Revisá la clave y el saldo; esperá un minuto y reintentá |
| Los grafos se dibujan con texto y no con imagen | No hay Internet (la imagen usa un servicio web) | Está bien: el dibujo en texto muestra lo mismo |
| Windows: `make` no se reconoce | `make` no viene con Windows | Usá los comandos `uv run ...` de esta guía; no hace falta `make` |

Si nada de esto resuelve el problema, copiá la salida de
`uv run python scripts/doctor.py` y compartila con el equipo docente
(no incluye tu clave: solo muestra sus últimos 4 caracteres).

## Para docentes: editar las clases

Los archivos `.py` de `clases/` y sus subcarpetas son la **fuente editable**; los `.ipynb` se generan a partir
de ellos. Si editás un notebook directamente en VS Code, llevá los cambios al `.py`
antes de verificar:

```bash
uv run python scripts/sync_notebooks.py --desde-notebooks   # notebook → .py → notebook limpio
uv run python scripts/verify.py --mode offline              # todas las clases en kernels nuevos
uv run python scripts/verify.py --mode offline --track workflows  # las 10 iniciales
uv run python scripts/evaluate_rag.py --mode offline        # doce preguntas para comparar RAG
```

Evitá `input()` en las celdas: la verificación automática no puede responderlo.
Usá una variable editable (por ejemplo, `MAX_LLAMADAS = 1`) con un comentario.
