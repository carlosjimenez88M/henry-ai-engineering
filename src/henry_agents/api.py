"""API docente sin acciones externas. Ejecutar en localhost."""

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from henry_agents.operations import run_observed

app = FastAPI(title="Henry Support Desk", version="1.0.0")


class Query(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Query):
    # El modo lo determina el servidor, nunca el cliente HTTP.
    return run_observed(body.query)
