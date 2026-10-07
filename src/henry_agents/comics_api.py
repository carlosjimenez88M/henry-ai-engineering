"""FastAPI para el laboratorio de cómics; el servidor controla modo y modelos.

Ejecutar localmente:
COURSE_MODE=offline uv run uvicorn henry_agents.comics_api:app --host 127.0.0.1 --port 8000
El modo live se configura en el entorno del servidor y requiere .env válido.
Importar este módulo no crea modelos ni hace llamadas de API.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from henry_agents.comics import (
    ConfiguracionComicError,
    PedidoComic,
    ProveedorComicError,
    ResultadoComic,
    SalidaComicInvalida,
    cargar_fuentes,
    generar_comic,
)
from henry_agents.config import configure


def crear_api(mode=None, modelos=None):
    # Cargar .env sin construir modelos ni exigir una clave durante la importación.
    configure("offline")
    selected = mode or os.getenv("COURSE_MODE") or "offline"
    if selected not in {"offline", "live"}:
        raise ValueError("El modo del servidor debe ser offline o live")
    api = FastAPI(title="Laboratorio Python y AI: cómics", version="1.0.0")
    api.state.mode = selected

    @api.exception_handler(RequestValidationError)
    async def entrada_invalida(request, error):
        # La respuesta de validación no repite valores del body, incluida una clave
        # que alguien enviara por error a pesar de no formar parte del contrato.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(item["loc"]), "type": item["type"], "msg": item["msg"]}
            for item in error.errors()
        ]})

    @api.get("/health")
    def health():
        return {"status": "ok", "mode": selected, "invocaciones_maximas_por_pedido": 7}

    @api.get("/fuentes")
    def fuentes():
        return {"universo": "escenario didactico ficticio", "fuentes": cargar_fuentes()}

    @api.post("/comics", response_model=ResultadoComic)
    def comics(pedido: PedidoComic):
        # def permite que FastAPI use su pool de hilos para este trabajo síncrono.
        try:
            return generar_comic(pedido, mode=selected, modelos=modelos)
        except (ConfiguracionComicError, ProveedorComicError):
            raise HTTPException(
                status_code=503, detail="El servicio de modelos no está disponible. Revisar la configuración del servidor.",
            ) from None
        except SalidaComicInvalida:
            raise HTTPException(
                status_code=502, detail="La salida del modelo no cumple los controles de este servicio.",
            ) from None

    return api


app = crear_api()
