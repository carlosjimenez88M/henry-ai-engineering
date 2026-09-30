"""Configuración central: sin claves incrustadas ni llamadas al importar."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def configure(mode=None):
    load_dotenv(ROOT / ".env", override=False)
    # Mantener los registros locales aunque el .env antiguo active trazado remoto.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    selected = mode or os.getenv("COURSE_MODE", "offline")
    if selected not in {"offline", "live"}:
        raise ValueError("COURSE_MODE debe ser offline o live")
    if selected == "live" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Falta OPENAI_API_KEY en .env")
    return selected


def chat_model():
    from langchain_openai import ChatOpenAI

    configure("live")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        timeout=30,
        max_retries=1,
        max_tokens=600,
    )
