import importlib.metadata
import os

from henry_agents.config import configure

mode = configure()
print("Modo:", mode)
for package in ["langgraph", "langchain-core", "langchain-openai", "nbclient"]:
    print(package, importlib.metadata.version(package))
for key in ["OPENAI_API_KEY"]:
    print(key, "configurada" if os.getenv(key) else "no configurada")
