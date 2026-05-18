# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
import os
import numpy as np
from dotenv import load_dotenv

# ✅ Cargar configuración
load_dotenv("api.env")
from config import LAT, LON, OPENWEATHER_KEY, GEMINI_API_KEY

# Importar Supabase
import supabase_db as db_module

# IA / Scheduler imports
import predictor
import notifier
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ========================
# Database Helper Functions
# ========================

def insert_lectura(temperatura: float, humedad: float):
    """Inserta lectura en Supabase"""
    return db_module.insert_lectura(temperatura, humedad)

def get_last_lectura():
    """Obtiene última lectura de Supabase"""
    return db_module.get_last_lectura()

def get_historial_lecturas(limit: int = 50):
    """Obtiene historial de Supabase"""
    return db_module.get_historial(limit)

# ========================
# Funciones auxiliares
# ========================

def convert_numpy_to_python(obj):
    """
    Convierte recursivamente valores numpy a tipos Python nativos
    para que puedan serializarse a JSON correctamente.
    """
    # Manejo especial para numpy.bool_
    if isinstance(obj, bool) and type(obj).__module__ == 'numpy':
        return bool(obj)
    
    # Manejo para todos los tipos numpy
    if hasattr(obj, 'dtype'):
        # Es un array o scalar de numpy
        if obj.dtype == bool or obj.dtype == 'bool':
            return bool(obj)
        elif np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
    
    # Manejo estándar
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ------------------------
# Mantener tus rutas IoT
# ------------------------

@app.route("/")
def home():
    return jsonify({"msg": "API IoT funcionando"})

@app.route("/guardar", methods=["POST"])
def guardar():
    try:
        data = request.get_json()
        temperatura = data.get("temperatura")
        humedad = data.get("humedad")

        if temperatura is None or humedad is None:
            return jsonify({"error": "Datos incompletos"}), 400

        result = insert_lectura(temperatura, humedad)
        return jsonify({"msg": "✅ Datos guardados correctamente", "data": result}), 201

    except Exception as e:
        logger.exception("Error en /guardar")
        return jsonify({"error": str(e)}), 500

@app.route("/ultimo")
def ultimo():
    try:
        row = get_last_lectura()
        if row:
            return jsonify(row)
        else:
            return jsonify({}), 200
    except Exception as e:
        logger.exception("Error en /ultimo")
        return jsonify({"error": str(e)}), 500

@app.route("/historial")
def historial():
    try:
        rows = get_historial_lecturas(50)
        return jsonify(rows)
    except Exception as e:
        logger.exception("Error en /historial")
        return jsonify({"error": str(e)}), 500

# 🔹 Alias para compatibilidad con Flutter
@app.route("/api/historial")
def api_historial():
    return historial()

@app.route("/api/ultimo")
def api_ultimo():
    return ultimo()

@app.route("/notificaciones")
def notificaciones():
    try:
        row = get_last_lectura()

        notifs = []
        if row:
            if row.get("temperatura") is not None and row["temperatura"] > 30:
                notifs.append(f"⚠️ Temperatura alta: {row['temperatura']} °C")
            if row.get("humedad") is not None and row["humedad"] < 30:
                notifs.append(f"❌ Humedad baja: {row['humedad']} %")
            if not notifs:
                notifs.append("✔️ Todo en rango normal")

        return jsonify({"notificaciones": notifs})
    except Exception as e:
        logger.exception("Error en /notificaciones")
        return jsonify({"error": str(e)}), 500

# ------------------------
# Endpoints IA / Tareas
# ------------------------

@app.route("/api/pronostico")
def api_pronostico():
    try:
        dias = int(request.args.get("dias", 7))
        data = predictor.predict_next_days(
            num_days=dias,
            lat=LAT,
            lon=LON
        )
        return jsonify({"pronostico": data})
    except Exception as e:
        logger.exception("Error en /api/pronostico")
        return jsonify({"error": str(e)}), 500

# 🤖 NEW ENDPOINT: Análisis Inteligente Avanzado
@app.route("/api/analisis-inteligente")
def api_analisis_inteligente():
    """
    Análisis ambiental INTELIGENTE con:
    - Detección de anomalías (Z-score, IQR)
    - Análisis de tendencias (regresión linear)
    - Correlaciones entre variables
    - Risk scoring dinámico
    - Explicación IA del riesgo con Google Gemini (si se proporciona gemini_key)
    - Recomendaciones personalizadas
    - Pronóstico del clima con Open-Meteo (GRATIS)
    
    Query params:
        gemini_key (opcional): Clave de Google Gemini para generar explicaciones IA
    """
    try:
        # Obtener clave de Gemini de query params (opcional)
        gemini_key = request.args.get("gemini_key", "").strip()
        
        # 1) Realizar análisis avanzado
        analysis = predictor.analyze_environmental_data(gemini_key=gemini_key if gemini_key else None)
        
        if not analysis.get("ok"):
            logger.warning(f"Analysis failed: {analysis.get('msg')}")
            return jsonify({"ok": False, "msg": analysis.get("msg")}), 400
        
        # 2) Generar tareas inteligentes basadas en análisis
        tasks = predictor.generate_tasks_from_analysis()
        logger.info(f"Generated {len(tasks)} tasks")
        
        # 3) Extraer información para frontend
        analysis_data = analysis.get("analysis", {})
        risk_assessment = analysis_data.get("risk_assessment", {})
        
        # Extraer factores de riesgo legibles
        risk_factors = risk_assessment.get("factors_list", [])
        ai_explanation = analysis_data.get("ai_explanation", "")
        confidence = risk_assessment.get("confidence", 0.8)
        
        # 4) Retornar análisis + tareas + explicación
        # Convertir valores numpy a tipos nativos de Python para JSON
        analysis_data_clean = convert_numpy_to_python(analysis_data)
        risk_factors_clean = convert_numpy_to_python(risk_factors)
        tasks_clean = convert_numpy_to_python(tasks)
        confidence_clean = float(confidence) if confidence else 0.5
        
        response = {
            "ok": True,
            "analysis": analysis_data_clean,
            "risk_factors": risk_factors_clean,
            "ai_explanation": str(ai_explanation),
            "recommended_tasks": tasks_clean,
            "task_count": len(tasks),
            "confidence": confidence_clean,
        }
        
        logger.info(f"API response ready: {len(str(response))} bytes")
        return jsonify(response)
        
    except Exception as e:
        logger.exception(f"Error en /api/analisis-inteligente: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"ok": False, "error": str(e), "type": type(e).__name__}), 500

@app.route("/api/mantenimiento")
def api_mantenimiento():
    """
    MEJORADO: Genera recomendaciones de tareas usando análisis INTELIGENTE.
    Ahora detecta anomalías, tendencias y genera tareas personalizadas.
    """
    try:
        # Generar tareas inteligentes
        df = predictor.read_last_n_from_db(500)
        tasks = predictor.generate_tasks_from_analysis(df)
        
        if not tasks:
            # Fallback: si no hay análisis suficiente, usar predicciones
            pred_res = predictor.predict_risk_from_recent(24, model_path=predictor.MODEL_PATH)
            if pred_res.get("ok"):
                tasks = predictor.generate_tasks_from_predictions(pred_res["predictions"])
            else:
                return jsonify({"ok": False, "msg": "No predictions available"}), 400
        
        # Guardar pronóstico y tareas en BD con PROTECCIÓN contra duplicados
        try:
            pred_res = predictor.predict_risk_from_recent(24, model_path=predictor.MODEL_PATH)
            if pred_res.get("ok"):
                predictor.save_pronostico_and_tasks_smart(pred_res, tasks, check_duplicates=True)
                logger.info(f"Saved {len(tasks)} intelligent tasks with duplicate check")
        except Exception:
            logger.exception("Warning: fallo al guardar pronóstico/tareas, continuar...")

        return jsonify({
            "ok": True, 
            "mantenimiento": tasks,
            "count": len(tasks),
            "message": "Tareas generadas por análisis IA inteligente"
        })
    except Exception as e:
        logger.exception("Error en /api/mantenimiento")
        return jsonify({"ok": False, "error": str(e)}), 500

# Rutas para tareas CRUD simples (app móvil)
@app.route("/api/tareas", methods=["GET"])
def api_get_tareas():
    """
    Devuelve tareas (pendientes y completadas) limit=100 por defecto, ordenadas por created_at desc.
    """
    limit = int(request.args.get("limit", 100))
    try:
        rows = db_module.get_tareas(limit=limit)
        # normalizar datetimes
        for r in rows:
            for k in ("created_at", "updated_at", "inicio", "fin"):
                if k in r and isinstance(r[k], str):
                    r[k] = r[k]  # ya está en formato ISO
        return jsonify(rows)
    except Exception as e:
        logger.exception("Error en /api/tareas GET")
        return jsonify({"error": str(e)}), 500

@app.route("/api/tareas", methods=["POST"])
def api_create_task():
    """
    Crea una tarea manual. JSON: {tarea, inicio, fin, motivo, riesgo (opcional)}
    """
    try:
        data = request.get_json() or {}
        tarea = data.get("tarea") or "Tarea manual"
        inicio = data.get("inicio")  # string 'YYYY-MM-DD' preferible
        fin = data.get("fin") or inicio
        riesgo = data.get("riesgo", "bajo")

        result = db_module.insert_tarea(
            tarea=tarea,
            inicio=inicio,
            fin=fin,
            riesgo=riesgo
        )
        task_id = result.get("id") if result else None
        return jsonify({"ok": True, "id": task_id}), 201
    except Exception as e:
        logger.exception("Error en /api/tareas POST")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/tareas/<int:task_id>/complete", methods=["POST"])
def api_complete_task(task_id):
    try:
        ok = notifier.mark_task_done(task_id)
        if ok:
            return jsonify({"ok": True})
        else:
            return jsonify({"ok": False, "msg": "No se pudo marcar completada"}), 500
    except Exception as e:
        logger.exception("Error en /api/tareas/complete")
        return jsonify({"error": str(e)}), 500

@app.route("/api/tareas/<int:task_id>", methods=["DELETE"])
def api_delete_task(task_id):
    try:
        db_module.delete_tarea(task_id)
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Error en /api/tareas DELETE")
        return jsonify({"ok": False, "error": str(e)}), 500

# ------------------------
# Scheduler (Background)
# ------------------------

def job_daily():
    """
    Trabajo programado MEJORADO: entrena modelo, realiza análisis inteligente y genera tareas.
    Diseñado para correr una vez al día. Para demo puedes cambiar intervalo en start_scheduler.
    
    Ahora genera tareas usando análisis IA avanzado con detección de anomalías y tendencias.
    """ 
    logger.info("🤖 Scheduler job_daily MEJORADO start: %s", datetime.now().isoformat())
    try:
        # 1) leer datos; si insuficientes, generar sintéticos
        df = predictor.read_last_n_from_db(2000)
        if df is None or df.empty or len(df) < predictor.MIN_ROWS_TO_TRAIN:
            logger.info("Pocos datos en DB; generando sintético para demo")
            df = predictor.generate_synthetic(hours=24 * 90)

        # 2) entrenar y guardar modelo
        train_res = predictor.train_and_save_model(df, model_path=predictor.MODEL_PATH)
        logger.info("train result: %s", train_res)

        # 3) NUEVO: Realizar análisis inteligente y generar tareas
        logger.info("Generating intelligent tasks from environmental analysis...")
        tasks = predictor.generate_tasks_from_analysis(df)
        
        # Fallback a predicciones si el análisis no genera tareas
        if not tasks:
            logger.info("No tasks from analysis, using ML predictions...")
            pred = predictor.predict_risk_from_recent(num_hours=24, model_path=predictor.MODEL_PATH)
            if pred.get("ok"):
                tasks = predictor.generate_tasks_from_predictions(pred["predictions"])
            else:
                logger.info("No se generó predicción: %s", pred.get("msg"))
        
        # 4) Guardar pronóstico + tareas con PROTECCIÓN contra duplicados
        if tasks:
            try:
                pred = predictor.predict_risk_from_recent(num_hours=24, model_path=predictor.MODEL_PATH)
                if pred.get("ok"):
                    result = predictor.save_pronostico_and_tasks_smart(
                        pred, 
                        tasks, 
                        check_duplicates=True,
                        process_id="scheduler-job_daily"  # 🔒 Mutex ID
                    )
                    if result.get("ok"):
                        logger.info(
                            f"✅ Saved pronostico + {result['inserted']} intelligent tasks "
                            f"(skipped {result['skipped']} duplicates)"
                        )
                    else:
                        logger.warning(f"Failed to save tasks: {result.get('error')}")
            except Exception:
                logger.exception("Error guardando pronostico/tareas en BD")
        else:
            logger.info("No tasks generated in job_daily")
            
    except Exception:
        logger.exception("Error en job_daily")


scheduler = None

def start_scheduler():
    global scheduler
    if scheduler is not None:
        return
    scheduler = BackgroundScheduler()
    # Para demo puedes usar minutes=1, en producción hours=24
    scheduler.add_job(job_daily, 'interval', hours=24, next_run_time=datetime.now())
    scheduler.start()
    logger.info("Background scheduler started")

# Iniciar scheduler automáticamente al importar la app (necesario para Gunicorn)
start_scheduler()

# ------------------------
# Run
# ------------------------

if __name__ == "__main__":
    # En producción (Render), se usa gunicorn, este bloque no se ejecuta.
    # Solo para desarrollo local:
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
