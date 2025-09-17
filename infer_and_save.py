from google.cloud import bigquery, storage
import pandas as pd
import numpy as np
import joblib, tempfile

PROJECT = "business-intelligence-444511"
INPUT_TABLE = "granier_logistica.ConsumoPrediccion_Semanal_V2"
OUT_RF = "granier_logistica.ConsumoPredicho_Semanal_RF"
OUT_Q75 = "granier_logistica.ConsumoPredicho_Semanal_Q75"
BUCKET = "granier-modelos"
TXT_RF = "consumo/last_model_v2_fast_rf.txt"
TXT_Q75 = "consumo/last_model_v2_fast_q75.txt"

def _load_model_from_txt(txt_blob):
    cs = storage.Client()
    b = cs.bucket(BUCKET)
    model_name = b.blob(txt_blob).download_as_text().strip()
    blob = b.blob(f"consumo/{model_name}")
    _, tmp = tempfile.mkstemp()
    blob.download_to_filename(tmp)
    return joblib.load(tmp), model_name

def _rolling_update(stats, new_val):
    window = stats["window"]
    window.append(float(new_val))
    if len(window) > 8:
        window.pop(0)
    def mean_last(k):
        w = window[-k:] if window else []
        return round(float(np.mean(w)), 2) if w else np.nan
    stats["MA_3"] = mean_last(3)
    stats["MA_4"] = mean_last(4)
    stats["MA_6"] = mean_last(6)
    stats["MA_8"] = mean_last(8)
    stats["STD_8"] = round(float(np.std(window, ddof=1)), 2) if len(window) >= 2 else 0.0

def infer_and_save():
    bq = bigquery.Client(project=PROJECT)
    df = bq.query(f"SELECT * FROM `{PROJECT}.{INPUT_TABLE}`").to_dataframe(create_bqstorage_client=True)

    # Carga modelos
    rf, rf_blob = _load_model_from_txt(TXT_RF)
    q75, q75_blob = _load_model_from_txt(TXT_Q75)
    print("[INFO] RF:", rf_blob)
    print("[INFO] Q75:", q75_blob)

    # Orden por grupo y semana
    # Importante: la tabla ya trae 6 semanas futuras; ordenamos por semana ascendente
    df["Semana_Date"] = pd.to_datetime(df["Semana_Martes"].apply(
        lambda s: pd.to_datetime(s + "-1", format="%G-W%V-%u")
    ))
    df = df.sort_values(["Centro","Articulo","Semana_Date"]).reset_index(drop=True)

    pred_rows_rf = []
    pred_rows_q75 = []

    feature_cols = [
        "Centro","Grupo_articulo",
        "Lag_1","Lag_2","Lag_3","Lag_4","Lag_5","Lag_6","Lag_7","Lag_8",
        "MA_3","MA_4","MA_6","MA_8","STD_8",
        "Vol_Ym1_clean","Crec_clean","YoY_level","Delta_MA3_vs_YoY"
    ]

    for (centro, articulo), g in df.groupby(["Centro","Articulo"], sort=False):
        g = g.copy().reset_index(drop=True)
    
        # Estado inicial con 8 lags (pueden venir NaN; el imputer del pipeline se encarga)
        lag1 = g.loc[0, "Lag_1"]; lag2 = g.loc[0, "Lag_2"]; lag3 = g.loc[0, "Lag_3"]; lag4 = g.loc[0, "Lag_4"]
        lag5 = g.loc[0, "Lag_5"]; lag6 = g.loc[0, "Lag_6"]; lag7 = g.loc[0, "Lag_7"]; lag8 = g.loc[0, "Lag_8"]
    
        # Ventana inicial: de más antiguo a más reciente, omitiendo NaN
        init_vals = [lag8, lag7, lag6, lag5, lag4, lag3, lag2, lag1]
        stats = {"window": [], "MA_3": g.loc[0, "MA_3"], "MA_4": g.loc[0, "MA_4"], "MA_6": g.loc[0, "MA_6"],
                 "MA_8": g.loc[0, "MA_8"], "STD_8": g.loc[0, "STD_8"]}
        for v in init_vals:
            if pd.notna(v):
                _rolling_update(stats, v)
    
        for i in range(len(g)):
            row = g.iloc[i].to_dict()
    
            # Usa el estado actual de lags (si vienen NaN en la fila, nos da igual: pasamos el estado)
            lags = {
                "Lag_1": lag1, "Lag_2": lag2, "Lag_3": lag3, "Lag_4": lag4,
                "Lag_5": lag5, "Lag_6": lag6, "Lag_7": lag7, "Lag_8": lag8,
            }
    
            # Completa MAs/STD con estado si vienen NaN
            for k in ["MA_3","MA_4","MA_6","MA_8","STD_8"]:
                if pd.isna(row[k]):
                    row[k] = stats[k]
    
            X_row = {
                "Centro": row["Centro"], "Grupo_articulo": row["Grupo_articulo"],
                **lags,
                "MA_3": row["MA_3"], "MA_4": row["MA_4"], "MA_6": row["MA_6"], "MA_8": row["MA_8"], "STD_8": row["STD_8"],
                "Vol_Ym1_clean": row["Vol_Ym1_clean"], "Crec_clean": row["Crec_clean"],
                "YoY_level": row["YoY_level"], "Delta_MA3_vs_YoY": row["Delta_MA3_vs_YoY"],
            }
            X_df = pd.DataFrame([X_row], columns=feature_cols)
    
            yhat_rf  = float(rf.predict(X_df)[0])
            yhat_q75 = float(q75.predict(X_df)[0])

            # Guardar filas
            base_out = {
                "Semana_Martes": row["Semana_Martes"],
                "Anio": int(row["Anio"]),
                "Semana_Num": int(row["Semana_Num"]),
                "Articulo": int(row["Articulo"]),
                "Centro": row["Centro"],
                "Grupo_articulo": row["Grupo_articulo"],
            }
            pred_rows_rf.append({**base_out, "Prediccion_Consumo": round(yhat_rf, 2)})
            pred_rows_q75.append({**base_out, "Prediccion_Consumo": round(yhat_q75, 2)})

            # === UPDATE AUTORREGRESIVO PARA LA SIGUIENTE SEMANA ===
            # Avanza lags con la predicción del modelo “central” (RF) o podrías usar blend.
            # Aquí usamos RF como truth interno para mantener consistencia.
            pred_state = yhat_rf
            lag8, lag7, lag6, lag5, lag4, lag3, lag2, lag1 = lag7, lag6, lag5, lag4, lag3, lag2, lag1, pred_state
            _rolling_update(stats, pred_state)

    # Subir a BigQuery
    df_rf = pd.DataFrame(pred_rows_rf)
    df_q  = pd.DataFrame(pred_rows_q75)

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    bq.load_table_from_dataframe(df_rf, f"{PROJECT}.{OUT_RF}", job_config=job_config).result()
    bq.load_table_from_dataframe(df_q,  f"{PROJECT}.{OUT_Q75}", job_config=job_config).result()
    print(f"[OK] RF  -> {PROJECT}.{OUT_RF}")
    print(f"[OK] Q75 -> {PROJECT}.{OUT_Q75}")

if __name__ == "__main__":
    infer_and_save()
