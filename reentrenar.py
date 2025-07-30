from google.cloud import bigquery, storage
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import uuid

def entrenar_y_guardar_modelo():
    client = bigquery.Client()
    query = "SELECT * FROM `business-intelligence-444511.granier_modelado.Consumo_Semanal_Train`"
    df = client.query(query).to_dataframe()

    # Separar X e y
    X = df[["Lag_1", "Lag_2", "Media_3_Semanas", "Vol_Ym1", "Crecimiento_WoW_Ym1"]]
    y = df["Volumen"]

    model = RandomForestRegressor()
    model.fit(X, y)

    # Guardar y subir a GCS
    filename = f"modelo_v{pd.Timestamp.today().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}.pkl"
    joblib.dump(model, filename)

    storage_client = storage.Client()
    bucket = storage_client.bucket("granier-modelos")
    blob = bucket.blob(f"consumo/{filename}")
    blob.upload_from_filename(filename)
    print(f"Modelo subido como {filename}")

    # Actualizar archivo con el nombre del modelo más reciente
    latest_blob = bucket.blob("consumo/last_model.txt")
    latest_blob.upload_from_string(filename)


if __name__ == "__main__":
    entrenar_y_guardar_modelo()
