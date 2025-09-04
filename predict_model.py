from google.cloud import bigquery, storage
import pandas as pd
import joblib
import tempfile
import numpy as np

# === Configuración ===
BQ_PROJECT = "business-intelligence-444511"
BQ_INPUT_TABLE = "granier_logistica.ConsumoPrediccion_Semanal"
BQ_OUTPUT_TABLE = "granier_logistica.ConsumoPredicho_Semanal"  # dataset.tabla (sin proyecto)
BUCKET_NAME = "granier-modelos"
LATEST_MODEL_TXT = "consumo/last_model.txt"

# --- Utilidades: cargar modelo desde GCS ---
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

# --- Predicción autoregresiva ---
def predecir_autoregresivo():
    # 1) Cargar modelo
    model_name = get_latest_model_name(BUCKET_NAME, LATEST_MODEL_TXT)
    model = load_model_from_gcs(BUCKET_NAME, f"consumo/{model_name}")

    # 2) Cargar tabla de entrada
    bq = bigquery.Client(project=BQ_PROJECT)
    df = bq.query(f"SELECT * FROM `{BQ_PROJECT}.{BQ_INPUT_TABLE}`").to_dataframe()

    # 3) Ordenar por combinación y semana
    df = df.sort_values(["Articulo", "Centro", "Semana_Martes"]).reset_index(drop=True)

    # 4) Predicción por grupo Articulo–Centro
    pred_rows = []
    for (articulo, centro), grupo in df.groupby(["Articulo", "Centro"], sort=False):
        grupo = grupo.copy().reset_index(drop=True)

        # Arrays “mutables” para lags/medias (evitan SettingWithCopy y facilitan updates)
        lag_1      = grupo["Lag_1"].astype(float).tolist()
        lag_2      = grupo["Lag_2"].astype(float).tolist()
        media_3    = pd.to_numeric(grupo["Media_3_Semanas"], errors="coerce").tolist()
        vol_ym1    = pd.to_numeric(grupo["Vol_Ym1"], errors="coerce").tolist()
        creci_yoy  = pd.to_numeric(grupo["Crecimiento_WoW_Ym1"], errors="coerce").tolist()

        for i in range(len(grupo)):
            # Si falta media_3 actual, intenta recomputarla con lo disponible
            if pd.isna(media_3[i]):
                base_vals = [v for v in [lag_1[i], lag_2[i]] if pd.notna(v)]
                media_3[i] = round(sum(base_vals) / len(base_vals), 2) if base_vals else np.nan

            # Si faltan features, no se predice esta fila
            features = [lag_1[i], lag_2[i], media_3[i], vol_ym1[i], creci_yoy[i]]
            if any(pd.isna(v) for v in features):
                pred = np.nan
            else:
                X = pd.DataFrame([{
                    "Lag_1": lag_1[i],
                    "Lag_2": lag_2[i],
                    "Media_3_Semanas": media_3[i],
                    "Vol_Ym1": vol_ym1[i],
                    "Crecimiento_WoW_Ym1": creci_yoy[i],
                }])
                pred = float(model.predict(X)[0])
                pred = round(pred, 2)

                # === ACTUALIZACIÓN AUTORREGRESIVA PARA LA SIGUIENTE FILA ===
                if i + 1 < len(grupo):
                    # OJO: Lag_2 (i+1) debe ser el Lag_1 de la fila i (no el de i+1)
                    lag_2[i + 1] = lag_1[i]
                    lag_1[i + 1] = pred
                    # Recalcula media_3 para i+1 con lo disponible
                    base_vals = [v for v in [lag_1[i + 1], lag_2[i + 1]] if pd.notna(v)]
                    media_3[i + 1] = round(sum(base_vals) / len(base_vals), 2) if base_vals else np.nan

            row = grupo.iloc[i].to_dict()
            row["Prediccion_Consumo"] = None if pd.isna(pred) else pred
            pred_rows.append(row)

    # 5) DataFrame final y tipos
    df_pred = pd.DataFrame(pred_rows)

    # 6) Subida a BigQuery (sin pandas_gbq)
    table_id = f"{BQ_PROJECT}.{BQ_OUTPUT_TABLE}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    load_job = bq.load_table_from_dataframe(df_pred, table_id, job_config=job_config)
    load_job.result()

    print(f"✅ Predicción completada y subida a {table_id}")

if __name__ == "__main__":
    predecir_autoregresivo()
