# predictor.py
import os    
import json
import joblib
import math
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests

import supabase_db
from ai_analyzer import EnvironmentalAnalyzer, TaskRecommender, convert_numpy_values

LOG = logging.getLogger("predictor")
LOG.setLevel(logging.INFO)

MODEL_PATH = os.environ.get("MODEL_PATH", "model_rf.joblib")
MIN_ROWS_TO_TRAIN = int(os.environ.get("MIN_ROWS_TO_TRAIN", "200"))

# -------------------------
# External weather helper
# -------------------------
def fetch_openmeteo(lat: float, lon: float):
    """Devuelve pronóstico de Open-Meteo (GRATIS - sin API key)"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_max&temperature_unit=celsius&timezone=auto"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            # Convertir a formato similar a OpenWeather para compatibilidad
            return {
                "daily": [
                    {
                        "temp": {
                            "day": data["daily"]["temperature_2m_max"][i],
                            "min": data["daily"]["temperature_2m_min"][i]
                        },
                        "humidity": data["daily"]["relative_humidity_2m_max"][i]
                    }
                    for i in range(len(data["daily"]["temperature_2m_max"]))
                ]
            }
        LOG.warning("OpenMeteo returned %s", r.status_code)
    except Exception as e:
        LOG.exception("fetch_openmeteo failed: %s", e)
    return {}

# -------------------------
# Synthetic generator
# -------------------------
def generate_synthetic(start: datetime = None, hours: int = 24 * 90, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    if start is None:
        start = datetime.now() - timedelta(hours=hours)
    rows = []
    for i in range(hours):
        ts = start + timedelta(hours=i)
        h = ts.hour
        base_temp = 22 + 6 * math.sin((h / 24.0) * 2 * math.pi)
        temp = base_temp + np.random.normal(0, 1.2)
        base_hum = 65 - 10 * math.sin((h / 24.0) * 2 * math.pi)
        hum = base_hum + (22 - temp) * 0.6 + np.random.normal(0, 3.0)
        hum = max(5, min(100, hum))
        rows.append({"fecha_hora": ts, "temperatura": round(float(temp), 2), "humedad": round(float(hum), 2)})
    return pd.DataFrame(rows)

# -------------------------
# DB reader
# -------------------------
def read_last_n_from_db(n=1000) -> pd.DataFrame:
    """Obtiene últimas n lecturas de Supabase"""
    rows = supabase_db.get_historial(n)
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    if "fecha_hora" in df.columns:
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    
    if "temperatura" not in df.columns and "temp" in df.columns:
        df["temperatura"] = df["temp"]
    if "humedad" not in df.columns and "hum" in df.columns:
        df["humedad"] = df["hum"]
    
    return df.sort_values("fecha_hora").reset_index(drop=True)

# -------------------------
# Feature engineering
# -------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values("fecha_hora").reset_index(drop=True)
    d["hour"] = d["fecha_hora"].dt.hour
    d["dayofyear"] = d["fecha_hora"].dt.dayofyear
    d["temp"] = d["temperatura"]
    d["hum"] = d["humedad"]
    d["hum_lag1"] = d["hum"].shift(1).bfill()
    d["hum_lag3"] = d["hum"].shift(3).bfill()
    d["hum_roll6"] = d["hum"].rolling(window=6, min_periods=1).mean()
    d["hum_roll24"] = d["hum"].rolling(window=24, min_periods=1).mean()
    d["temp_lag1"] = d["temp"].shift(1).bfill()
    return d

# -------------------------
# Heuristic labels (0 low /1 med /2 high)
# -------------------------
def heuristic_label(df: pd.DataFrame) -> pd.Series:
    hum = df["hum"].values
    n = len(hum)
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        start = max(0, i - 5)
        window = hum[start:i + 1]
        avg_window = window.mean()
        if hum[i] >= 75 or avg_window >= 75:
            labels[i] = 2
        elif hum[i] >= 65 or avg_window >= 65:
            labels[i] = 1
        else:
            labels[i] = 0
    return pd.Series(labels, index=df.index)

# -------------------------
# Train / Save / Load
# -------------------------
def train_and_save_model(df: pd.DataFrame, model_path: str = MODEL_PATH) -> dict:
    if df is None or df.empty or len(df) < MIN_ROWS_TO_TRAIN:
        return {"ok": False, "msg": "Insufficient rows for training"}
    d = df.copy().reset_index(drop=True)
    d = make_features(d)
    d["risk"] = heuristic_label(d)
    features = ["hour", "dayofyear", "temp", "temp_lag1", "hum", "hum_lag1", "hum_lag3", "hum_roll6", "hum_roll24"]
    X = d[features].fillna(0)
    y = d["risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump({"model": model, "features": features}, model_path)
    return {"ok": True, "acc": float(acc), "n_samples": len(d)}

def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# -------------------------
# Predict functions
# -------------------------
def predict_risk_from_recent(num_hours: int = 24, model_path: str = MODEL_PATH) -> Dict:
    df = read_last_n_from_db(max(300, num_hours * 3))
    if df is None or df.empty:
        return {"ok": False, "msg": "No lecturas en DB"}
    d = df.copy().tail(num_hours).reset_index(drop=True)
    d = make_features(d)
    model_obj = load_model(model_path)
    if model_obj is None:
        return {"ok": False, "msg": "No hay modelo entrenado"}
    model = model_obj["model"]
    features = model_obj["features"]
    X = d[features].fillna(0)
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    results = []
    for idx, row in d.iterrows():
        p = int(preds[idx])
        prob = None
        if probs is not None:
            prob = {str(i): float(probs[idx][i]) for i in range(probs.shape[1])}
        results.append({
            "fecha_hora": row["fecha_hora"].isoformat(),
            "pred": int(p),
            "prob": prob
        })
    counts = pd.Series(preds).value_counts().to_dict()
    return {"ok": True, "summary": counts, "predictions": results}

# -------------------------
# Daily / multi-day forecast (daily aggregated)
# -------------------------
def predict_next_days(num_days: int = 7, lat: float = None, lon: float = None) -> List[Dict]:
    # valores por defecto si lat/lon no vienen
    if lat is None:
        lat = float(os.environ.get("LAT", "10.426593484472422"))
    if lon is None:
        lon = float(os.environ.get("LON", "-75.53768518832231"))

    """Predice riesgo por día combinando modelo interno (promedios recientes) + Open-Meteo forecast (gratis)."""
    # 1) Intentar cargar modelo
    model_obj = load_model()
    if model_obj is None:
        # fallback heuristico simple: usar promedio rolling + trend
        df = read_last_n_from_db(500)
        if df is None or df.empty:
            df = generate_synthetic(hours=24 * 30)
        recent = df.tail(48)
        avg_temp = float(recent["temperatura"].mean())
        avg_hum = float(recent["humedad"].mean())
        out = []
        for i in range(num_days):
            ddate = (datetime.now() + timedelta(days=i)).date().isoformat()
            # small random walk
            t = round(avg_temp + np.random.normal(0, 1.5), 2)
            h = round(max(5, min(100, avg_hum + np.random.normal(0, 4))), 2)
            out.append({"dia": ddate, "temperatura": t, "humedad": h})
        return out

    # 2) Usar Open-Meteo forecast (GRATIS - sin API key)
    weather = {}
    if lat is not None and lon is not None:
        weather = fetch_openmeteo(lat, lon)

    df_recent = read_last_n_from_db(500)
    if df_recent is None or df_recent.empty:
        df_recent = generate_synthetic(hours=24 * 30)
    model = model_obj["model"]
    features = model_obj["features"]

    out = []
    # prepare daily aggregation from Open-Meteo if exists
    daily_forecasts = []
    if "daily" in weather and isinstance(weather["daily"], list):
        daily_forecasts = weather["daily"][:num_days]
    for i in range(num_days):
        target_date = (datetime.now() + timedelta(days=i)).date()
        # build a fake hourly-like row by combining recent stats and external day forecast
        avg_temp = float(df_recent["temperatura"].tail(24).mean())
        avg_hum = float(df_recent["humedad"].tail(24).mean())
        ext_temp = None
        ext_hum = None
        if i < len(daily_forecasts):
            d = daily_forecasts[i]
            ext_temp = d.get("temp", {}).get("day") if isinstance(d.get("temp"), dict) else d.get("temp")
            ext_hum = d.get("humidity")
        # craft feature row
        row = {
            "fecha_hora": datetime.combine(target_date, datetime.min.time()),
            "temperatura": ext_temp if ext_temp is not None else avg_temp + np.random.normal(0, 1.0),
            "humedad": ext_hum if ext_hum is not None else max(5, min(100, avg_hum + np.random.normal(0, 3.0))),
        }
        df_tmp = pd.DataFrame([row])
        df_tmp = make_features(df_tmp)
        X = df_tmp[features].fillna(0)
        pred = model.predict(X)[0]
        out.append({"dia": target_date.isoformat(), "temperatura": float(row["temperatura"]), "humedad": float(row["humedad"]), "risk": int(pred)})
    return out

def generate_tasks_from_predictions(predictions: List[dict]) -> List[dict]:
    """
    Genera tareas automáticas basadas en las predicciones de riesgo.
    DEPRECATED: Usar generate_tasks_from_analysis() en su lugar.
    
    - pred = 2 → riesgo alto → tarea urgente
    - pred = 1 → riesgo medio → tarea preventiva
    """
    recommender = TaskRecommender()
    # Usar el recommender con las predicciones crudas
    tasks = recommender.generate_recommendations(
        {"ok": True, "analysis": {}},
        predictions=predictions
    )
    # Convertir valores numpy a tipos Python nativos
    return convert_numpy_values(tasks)

# ========================
# New Advanced IA Functions
# ========================

def analyze_environmental_data(df: pd.DataFrame = None, gemini_key: str = None) -> Dict:
    """
    Análisis ambiental avanzado usando el módulo IA.
    Si no se proporciona df, lee de la BD.
    
    Args:
        df: DataFrame con datos históricos (si es None, lee de BD)
        gemini_key: Clave de Google Gemini para generar explicaciones (opcional)
    
    Returns:
        Dict con análisis completo: anomalías, tendencias, risk scores, etc.
    """
    if df is None:
        df = read_last_n_from_db(n=500)
    
    if df is None or df.empty:
        return {"ok": False, "msg": "No data available"}
    
    analyzer = EnvironmentalAnalyzer(lookback_hours=168)
    return analyzer.analyze_dataframe(df, gemini_key=gemini_key)

def generate_tasks_from_analysis(df: pd.DataFrame = None, predictions: List[dict] = None) -> List[dict]:
    """
    Genera recomendaciones de tareas usando análisis ambiental INTELIGENTE.
    
    Args:
        df: DataFrame con datos históricos (si es None, lee de BD)
        predictions: Predicciones del modelo ML (opcional)
        
    Returns:
        Lista de tareas recomendadas con acciones
    """
    # Realizar análisis ambiental avanzado
    analysis = analyze_environmental_data(df)
    
    if not analysis.get("ok"):
        return []
    
    # Generar recomendaciones basadas en análisis
    recommender = TaskRecommender()
    tasks = recommender.generate_recommendations(analysis, predictions=predictions)
    
    # Convertir valores numpy a tipos Python nativos
    tasks = convert_numpy_values(tasks)
    
    LOG.info(f"Generated {len(tasks)} intelligent tasks from analysis")
    return tasks

def save_pronostico_and_tasks_smart(
    pronostico: dict, 
    tasks: List[dict],
    check_duplicates: bool = True,
    process_id: str = "default"
):
    """
    Versión SUPABASE de save_pronostico_and_tasks con:
    - 🔒 Mutex/Locking (evita condiciones de carrera)
    - 🚫 Protección contra duplicados
    - 📊 Logging detallado
    """
    # 🔒 Adquirir mutex para evitar condiciones de carrera
    if not supabase_db.TaskGenerationLock.acquire_lock(process_id):
        LOG.warning(f"Could not acquire lock, skipping task generation")
        return {"ok": False, "error": "Could not acquire lock"}
    
    try:
        # 1) Guardar pronóstico
        pron_result = supabase_db.save_pronostico(
            detalle_json=pronostico,
            resumen=pronostico.get("summary", {})
        )
        pron_id = pron_result.get("id") if pron_result else None
        LOG.info(f"Created pronostico #{pron_id}")
        
        # 2) Insertar tareas con protección contra duplicados
        inserted_count = 0
        skipped_count = 0
        
        for t in tasks:
            tarea_name = t.get("tarea", "Unknown")
            inicio = t.get("inicio")
            
            if check_duplicates:
                # 🚫 Verificar duplicado: misma tarea, pendiente
                existing_tasks = supabase_db.get_tareas(estado="pendiente")
                if any(task.get("tarea") == tarea_name and 
                       str(task.get("inicio")).split("T")[0] == str(inicio).split("T")[0]
                       for task in existing_tasks):
                    LOG.info(f"[DUPLICATE SKIPPED] {tarea_name} on {inicio}")
                    skipped_count += 1
                    continue
            
            # ✅ Insertar nueva tarea
            try:
                supabase_db.insert_tarea(
                    tarea=tarea_name,
                    inicio=inicio,
                    fin=t.get("fin"),
                    riesgo=t.get("riesgo", "bajo"),
                    pronostico_id=pron_id
                )
                inserted_count += 1
                LOG.debug(f"[INSERTED] {tarea_name}")
            except Exception as e:
                LOG.warning(f"[ERROR INSERTING] {tarea_name}: {str(e)[:100]}")
                skipped_count += 1
        
        # 🎯 Log final
        LOG.info(
            f"✅ Task generation complete: {inserted_count} inserted, "
            f"{skipped_count} skipped (duplicates), pronostico_id={pron_id}"
        )
        
        return {
            "ok": True,
            "pronostico_id": pron_id,
            "inserted": inserted_count,
            "skipped": skipped_count,
            "total": len(tasks)
        }
        
    except Exception as e:
        LOG.exception(f"❌ Error in save_pronostico_and_tasks_smart: {e}")
        return {
            "ok": False,
            "error": str(e)
        }
    finally:
        # Liberar el lock
        supabase_db.TaskGenerationLock.release_lock(process_id)

# -------------------------
# Save pronostico + tasks into DB (used by scheduler)
# -------------------------
def save_pronostico_and_tasks(pronostico: dict, tasks: List[dict]):
    """Wrapper para compatibilidad - llama a save_pronostico_and_tasks_smart"""
    return save_pronostico_and_tasks_smart(pronostico, tasks, check_duplicates=True)
