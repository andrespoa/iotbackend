"""
Configuración centralizada para el backend
Configurado para Supabase (Render)
"""
import os
from dotenv import load_dotenv

load_dotenv("api.env")

# ========================
# DATABASE CONFIGURATION
# ========================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL o SUPABASE_KEY no configurados en api.env")

# ========================
# FLASK CONFIGURATION
# ========================

FLASK_ENV = os.environ.get("FLASK_ENV", "production")
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

# ========================
# EXTERNAL APIs
# ========================

OPENWEATHER_KEY = os.environ.get("OPENWEATHER_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ========================
# LOCATION (LAB)
# ========================

LAT = float(os.environ.get("LAT", "10.426571100984994"))
LON = float(os.environ.get("LON", "-75.5376662222412"))

# ========================
# MACHINE LEARNING
# ========================

MODEL_PATH = os.environ.get("MODEL_PATH", "model_rf.joblib")
MIN_ROWS_TO_TRAIN = int(os.environ.get("MIN_ROWS_TO_TRAIN", "200"))

# ========================
# CORS
# ========================

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
