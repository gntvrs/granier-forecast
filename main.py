from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Cargar modelo entrenado
model = joblib.load("modelo_consumo_semanal.pkl")

class ConsumoInput(BaseModel):
    Lag_1: float
    Lag_2: float
    Media_3_Semanas: float
    Vol_Ym1: float
    Crecimiento_WoW_Ym1: float

@app.post("/predecir/")
def predecir_consumo(data: ConsumoInput):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    return {"prediccion_volumen_semana": round(pred, 2)}
