from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

# ❗ Nada de cargar el modelo aquí arriba a la importación.
# Si quieres cargarlo, haz lazy-load dentro de una función o endpoint.

