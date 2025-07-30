from google.cloud import bigquery, storage
import pandas as pd
import joblib
import tempfile

# Configuración
BQ_PROJECT = "business-intelligence-444511"
BQ_INPUT_TABLE = "granier_logistica.ConsumoPrediccion_Semanal"
BQ_OUTPUT_TABLE = "granier_logistica.ConsumoPredicho_Semanal"
BUCKET_NAME = "granier-modelos"
LATEST_MODEL_TXT = "consumo/last_model.txt"

# Función para cargar modelo desde GCS
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

# Predicción autoregresiva
def predecir_autoregresivo():
    # 1. Cargar modelo
    model_name = get_latest_model_name(BUCKET_NAME, LATEST_MODEL_TXT)
    model = load_model_from_gcs(BUCKET_NAME, f"consumo/{model_name}")

    # 2. Cargar tabla de entrada
    bq = bigquery.Client()
    query = f"SELECT * FROM `{BQ_PROJECT}.{BQ_INPUT_TABLE}`"
    df = bq.query(query).to_dataframe()

    # 3. Ordenar por combinación y semana
    df.sort_values(["Articulo", "Centro", "Semana_Martes"], inplace=True)

    # 4. Almacén para resultados
    predicciones = []

    # 5. Predecir por grupo Articulo–Centro
    for (articulo, centro), grupo in df.groupby(["Articulo", "Centro"]):
        grupo = grupo.copy()
        lag_1 = grupo["Lag_1"].tolist()
        lag_2 = grupo["Lag_2"].tolist()
        media_3 = grupo["Media_3_Semanas"].tolist()
        vol_ym1 = grupo["Vol_Ym1"].tolist()
        creci_yoy = grupo["Crecimiento_WoW_Ym1"].tolist()

        historial_pred = []

        for i in range(len(grupo)):
            # Si falta media_3 inicial, la calculamos
            if pd.isna(media_3[i]):
                valores = [v for v in [lag_1[i], lag_2[i]] if pd.notna(v)]
                media_3[i] = round(sum(valores) / len(valores), 2) if valores else None

            # Si faltan features necesarias, se salta
            if any(pd.isna(v) for v in [lag_1[i], lag_2[i], media_3[i], vol_ym1[i], creci_yoy[i]]):
                pred = None
            else:
                X = pd.DataFrame([{
                    "Lag_1": lag_1[i],
                    "Lag_2": lag_2[i],
                    "Media_3_Semanas": media_3[i],
                    "Vol_Ym1": vol_ym1[i],
                    "Crecimiento_WoW_Ym1": creci_yoy[i],
                }])
                pred = model.predict(X)[0]
                pred = round(float(pred), 2)

                # Actualizar lags para la próxima iteración
                if i + 1 < len(grupo):
                    lag_2[i + 1] = lag_1[i + 1]
                    lag_1[i + 1] = pred
                    media_vals = [v for v in [lag_1[i + 1], lag_2[i + 1]] if pd.notna(v)]
                    media_3[i + 1] = round(sum(media_vals) / len(media_vals), 2) if media_vals else None

            row = grupo.iloc[i].to_dict()
            row["Prediccion_Consumo"] = pred
            predicciones.append(row)

    # 6. Crear DataFrame final
    df_pred = pd.DataFrame(predicciones)

    # 7. Subir a BigQuery
    df_pred.to_gbq(
        destination_table=BQ_OUTPUT_TABLE,
        project_id=BQ_PROJECT,
        if_exists="replace"
    )

    print(f"✅ Predicción completada y subida a {BQ_OUTPUT_TABLE}")

if __name__ == "__main__":
    predecir_autoregresivo()
