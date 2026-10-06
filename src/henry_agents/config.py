"""Configuración central del curso: modo, modelos, precios y costo. Sin llamadas al importar."""

import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Única fuente de modelos y precios del curso (USD por millón de tokens: entrada, salida).
# Consultado en https://developers.openai.com/api/docs/models el 6/10/2026.
# Revisar antes de cada cohorte: los precios cambian.
MODELOS = {
    "gpt-6-luna": {
        "precio": (0.10, 0.50),
        "esfuerzos": {"none", "low", "medium", "high", "xhigh", "max"},
        "uso": "Rápido y económico: clasificar, extraer, agentes simples y especialistas",
    },
    "gpt-6.1-sol": {
        "precio": (2.00, 10.00),
        "esfuerzos": {"low", "medium", "high", "xhigh", "max"},
        "uso": "Equilibrio: agentes con varias herramientas y coordinadores",
    },
    "gpt-6-astra": {
        "precio": (10.00, 50.00),
        "esfuerzos": {"low", "medium", "high", "xhigh", "max"},
        "uso": "Máxima capacidad: tareas largas y difíciles; usar con criterio",
    },
}
# Compatibilidad con la clase 00 y el diagnóstico preexistentes. Derivada de la
# tabla central: no mantiene una segunda lista de modelos o precios.
MODELOS_OPENAI = {nombre: datos["uso"] for nombre, datos in MODELOS.items()}

MODELO_EMBEDDINGS = "text-embedding-3-large"
MODELOS_VERIFICADOS_EL = "2026-10-06"
MODELO_POR_DEFECTO = "gpt-6-luna"
MODELO_AGENTE_POR_DEFECTO = "gpt-6.1-sol"


def configure(mode=None):
    """Carga .env y devuelve el modo: 'offline' (gratis, sin clave) o 'live' (OpenAI)."""
    load_dotenv(ROOT / ".env", override=False)
    # Las trazas remotas quedan apagadas salvo que el docente las active explícitamente.
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    selected = mode or os.getenv("COURSE_MODE") or "offline"
    if selected not in {"offline", "live"}:
        raise ValueError("COURSE_MODE debe ser offline o live")
    if selected == "live" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Falta OPENAI_API_KEY en .env (ver docs/INSTALACION.md)")
    return selected


def model_name(role="default"):
    """Nombre por rol; consultar un nombre no requiere clave ni hace llamadas."""
    load_dotenv(ROOT / ".env", override=False)
    roles = {
        "default": ("OPENAI_MODEL", MODELO_POR_DEFECTO),
        "agent": ("OPENAI_MODEL_AGENT", MODELO_AGENTE_POR_DEFECTO),
        "rag": ("OPENAI_MODEL_RAG", MODELO_POR_DEFECTO),
        "embeddings": ("OPENAI_EMBEDDING_MODEL", MODELO_EMBEDDINGS),
    }
    if role not in roles:
        raise ValueError(f"Rol desconocido: {role}")
    variable, fallback = roles[role]
    return os.getenv(variable) or fallback


def chat_model(role="default", **overrides):
    """Modelo de chat de OpenAI configurado para el curso.

    - Responses API: la interfaz recomendada por OpenAI para la familia GPT-6.
    - reasoning_effort: cuánto "piensa" antes de responder. Más esfuerzo = más tokens.
    - max_tokens incluye los tokens de razonamiento: si es bajo, la respuesta queda cortada.
    """
    from langchain_openai import ChatOpenAI

    if role == "embeddings":
        raise ValueError("Los embeddings usan OpenAIEmbeddings, no un modelo de chat")
    configure("live")
    nombre = overrides.pop("model", None) or model_name(role)
    effort = overrides.pop("reasoning_effort", None) or os.getenv("OPENAI_REASONING_EFFORT") or "low"
    permitidos = MODELOS.get(nombre, {}).get("esfuerzos")
    if permitidos and effort not in permitidos:
        raise ValueError(f"{nombre} no acepta reasoning_effort={effort!r}; usar {sorted(permitidos)}")
    settings = {
        "model": nombre,
        "use_responses_api": True,
        "reasoning_effort": effort,
        "timeout": 120,
        "max_retries": 2,
        "max_tokens": 8000,
        **overrides,
    }
    return ChatOpenAI(**settings)


def costo_usd(modelo, tokens_entrada, tokens_salida):
    """Costo estimado en dólares de una cantidad de tokens con un modelo del curso."""
    entrada, salida = MODELOS[modelo]["precio"]
    return (tokens_entrada * entrada + tokens_salida * salida) / 1_000_000


@contextmanager
def medir_costo():
    """Mide tokens y costo REAL de lo que se ejecute adentro del bloque `with`.

    with medir_costo() as medicion:
        agente.invoke(...)
    En offline no hay modelo real: la medición queda en cero, y eso también es un dato.
    """
    from langchain_core.callbacks import get_usage_metadata_callback

    with get_usage_metadata_callback() as callback:
        medicion = {"por_modelo": callback.usage_metadata}
        yield medicion
    total = 0.0
    for nombre, uso in callback.usage_metadata.items():
        base = next((m for m in MODELOS if nombre.startswith(m)), None)
        costo = costo_usd(base, uso["input_tokens"], uso["output_tokens"]) if base else 0.0
        total += costo
        print(
            f"💲 {nombre}: {uso['input_tokens']} tokens de entrada, "
            f"{uso['output_tokens']} de salida ≈ ${costo:.5f}"
        )
    if not callback.usage_metadata:
        print("💲 Sin llamadas a un modelo real (modo offline): costo $0.")
    medicion["usd"] = total

# Compatibilidad: nombre usado por scripts/doctor.py y material anterior.
MODELOS_OPENAI = {nombre: datos["uso"] for nombre, datos in MODELOS.items()}
