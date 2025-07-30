from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from google.cloud import storage
import tempfile

app = FastAPI()

# --- Utilidades ---

def get_latest_model_name(bucket_name: str, txt_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(txt_path)
    return blob.download_as_text().strip()

def load_model_from_gcs(bucket_name: str, blob_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    _, local_path = tempfile.mkstemp()
    blob.download_to_filename(local_path)
    return joblib.load(local_path)

# --- Cargar modelo más reciente desde GCS ---
BUCKET_NAME = "granier-modelos"
LATEST_MODEL_TXT = "consumo/last_model.txt"

model_filename = get_latest_model_name(BUCKET_NAME, LATEST_MODEL_TXT)
model = load_model_from_gcs(BUCKET_NAME, f"consumo/{model_filename}")

# --- API Input ---
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

