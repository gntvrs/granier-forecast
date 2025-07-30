from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Cargar modelo entrenado
from google.cloud import storage
import tempfile

def load_model_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    _, local_path = tempfile.mkstemp()
    blob.download_to_filename(local_path)
    return joblib.load(local_path)

model = load_model_from_gcs("granier-modelos", "consumo/modelo_v1_20250730.pkl")


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
