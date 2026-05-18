# ai_analyzer.py
"""
 Módulo IA Mejorado para Análisis Inteligente de Datos Ambientales
Reemplaza la lógica heurística simple con análisis estadístico real.
"""

import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from scipy import stats

# Suprimir warnings de numpy sobre precision loss en zscore
# Usar filterwarnings para suprimir warnings específicos de numpy
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*Precision loss.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

LOG = logging.getLogger("ai_analyzer")
LOG.setLevel(logging.INFO)


def convert_numpy_values(obj):
    """
    Convierte recursivamente valores numpy a tipos Python nativos.
    Esto previene errores de serialización JSON.
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
        return {k: convert_numpy_values(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_values(item) for item in obj]
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


class EnvironmentalAnalyzer:
    """
    Analizador inteligente de datos ambientales con detección de anomalías,
    análisis de tendencias y recomendaciones personalizadas.
    """

    def __init__(self, lookback_hours: int = 168):  # 1 semana por defecto
        """
        Args:
            lookback_hours: Ventana histórica para análisis de tendencias
        """
        self.lookback_hours = lookback_hours
        self.z_threshold = 2.5  # Detectar anomalías > 2.5 std devs
        self.iqr_multiplier = 1.5  # Para método IQR

    def analyze_dataframe(self, df: pd.DataFrame, gemini_key: str = None) -> Dict:
        """
        Análisis completo de un dataframe con sensores.
        
        Args:
            df: DataFrame con columnas 'temperatura', 'humedad', 'fecha_hora'
            openai_key: Clave de OpenAI (opcional) para generar explicaciones
            
        Returns:
            Dict con análisis completo incluyendo anomalías, tendencias, risk scores.
        """
        if df is None or df.empty:
            return {"ok": False, "msg": "No data provided"}

        df = df.copy().sort_values("fecha_hora").reset_index(drop=True)

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "data_points": len(df),
            "temperature": self._analyze_metric(df, "temperatura"),
            "humidity": self._analyze_metric(df, "humedad"),
            "anomalies": self._detect_anomalies(df),
            "trends": self._analyze_trends(df),
            "correlations": self._analyze_correlations(df),
            "risk_assessment": self._compute_risk_assessment(df),
        }
        
        # Generar explicación IA si se proporciona clave Gemini
        if gemini_key and HAS_GEMINI:
            ai_explanation = self.generate_ai_explanation(
                analysis["risk_assessment"],
                analysis["temperature"],
                analysis["humidity"],
                analysis["trends"],
                gemini_key
            )
            if ai_explanation:
                analysis["ai_explanation"] = ai_explanation["explanation"]
                analysis["confidence"] = ai_explanation.get("confidence", 0.85)
            else:
                analysis["ai_explanation"] = ""
                analysis["confidence"] = 0.8
        else:
            analysis["ai_explanation"] = ""
            # Calcular confianza basada en cantidad de datos
            analysis["confidence"] = min(0.95, 0.5 + len(df) / 1000)

        # Convertir todos los valores numpy a tipos Python nativos
        try:
            analysis = convert_numpy_values(analysis)
        except Exception as e:
            LOG.error(f"Error converting numpy values: {e}")
            # Si falla la conversión, al menos retornar algo útil
        
        return {"ok": True, "analysis": analysis}

    def _analyze_metric(self, df: pd.DataFrame, metric: str) -> Dict:
        """Análisis estadístico de una métrica individual."""
        data = df[metric].dropna()
        
        if len(data) == 0:
            return {"error": f"No {metric} data"}

        return {
            "current": float(data.iloc[-1]),
            "mean": float(data.mean()),
            "median": float(data.median()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "q1": float(data.quantile(0.25)),
            "q3": float(data.quantile(0.75)),
            "trend_direction": self._get_trend_direction(data),
            "trend_strength": self._calculate_trend_strength(data),
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Detección multimétodo de anomalías:
        - Z-score (desviación estándar)
        - IQR (rango intercuartil)
        - Isolation Forest simple (basado en desviaciones)
        """
        anomalies = {
            "temperatura": {"zscore": [], "iqr": []},
            "humedad": {"zscore": [], "iqr": []},
        }

        for metric in ["temperatura", "humedad"]:
            data = df[metric].dropna()
            if len(data) < 3:
                continue

            # Método Z-Score (con manejo de datos casi idénticos)
            try:
                # Calcular zscore con manejo de varianza cero
                data_std = data.std()
                if data_std > 0.001:  # Datos tienen variabilidad
                    with np.errstate(divide='ignore', invalid='ignore'):
                        z_scores = np.abs(stats.zscore(data))
                        # Filtrar NaN e infinitos
                        z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    # Datos casi idénticos, no hay anomalías por zscore
                    z_scores = np.zeros(len(data))
            except (RuntimeWarning, ZeroDivisionError) as e:
                LOG.debug(f"zscore skipped for {metric}: {e}")
                z_scores = np.zeros(len(data))
            
            zscore_anomalies = np.where(z_scores > self.z_threshold)[0].tolist()
            if zscore_anomalies:
                anomalies[metric]["zscore"] = [
                    {
                        "index": int(idx),
                        "value": float(data.iloc[idx]),
                        "z_score": float(z_scores[idx]),
                        "timestamp": df.iloc[idx]["fecha_hora"].isoformat() if "fecha_hora" in df.columns else None,
                    }
                    for idx in zscore_anomalies[-10:]  # Últimas 10 anomalías
                ]

            # Método IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            iqr_anomalies = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
            if iqr_anomalies:
                anomalies[metric]["iqr"] = [
                    {
                        "index": int(idx),
                        "value": float(data.iloc[idx]),
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                        "timestamp": df.iloc[idx]["fecha_hora"].isoformat() if "fecha_hora" in df.columns else None,
                    }
                    for idx in iqr_anomalies[-10:]  # Últimas 10
                ]

        return anomalies

    def _analyze_trends(self, df: pd.DataFrame) -> Dict:
        """
        Análisis de tendencias usando regresión linear.
        Detecta si está creciendo, decreciendo o estable.
        """
        trends = {}

        for metric in ["temperatura", "humedad"]:
            data = df[metric].dropna()
            if len(data) < 2:
                trends[metric] = {"error": "Insufficient data"}
                continue

            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)

            # Calcular cambio porcentual
            if data.iloc[0] != 0:
                pct_change = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0]) * 100
            else:
                pct_change = 0

            trends[metric] = {
                "direction": "↑ Aumentando" if slope > 0.01 else "↓ Disminuyendo" if slope < -0.01 else "→ Estable",
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "pct_change_total": round(pct_change, 2),
                "significant": bool(float(p_value) < 0.05),
                "days_to_critical": self._estimate_days_to_critical(data, slope, metric),
            }

        return trends

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Analiza correlación entre temperatura y humedad.
        """
        if "temperatura" not in df.columns or "humedad" not in df.columns:
            return {"error": "Missing metrics"}

        temp = df["temperatura"].dropna()
        hum = df["humedad"].dropna()

        if len(temp) < 2 or len(hum) < 2:
            return {"error": "Insufficient data"}

        # Alinear índices
        common_idx = temp.index.intersection(hum.index)
        if len(common_idx) < 2:
            return {"error": "No common data points"}

        correlation = temp[common_idx].corr(hum[common_idx])

        return {
            "temp_humidity_correlation": round(float(correlation), 3) if not np.isnan(correlation) else None,
            "interpretation": self._interpret_correlation(correlation),
        }

    def _compute_risk_assessment(self, df: pd.DataFrame) -> Dict:
        """
        Scoring dinámico de riesgo basado en múltiples factores:
        - Desviación actual de la norma
        - Tendencias peligrosas
        - Anomalías detectadas
        - Correlaciones inusuales
        """
        risk_factors = {}
        risk_factors_list = []  # Lista legible de factores

        # 1️⃣ Factor: Desviación actual
        temp_data = df["temperatura"].dropna()
        hum_data = df["humedad"].dropna()

        current_temp = temp_data.iloc[-1] if len(temp_data) > 0 else None
        current_hum = hum_data.iloc[-1] if len(hum_data) > 0 else None

        if current_temp is not None:
            temp_std = temp_data.std()
            temp_mean = temp_data.mean()
            temp_deviation = abs(current_temp - temp_mean) / (temp_std + 0.1)
            risk_factors["temperature_deviation"] = min(1.0, temp_deviation / 3.0)
            
            if current_temp > 30:
                risk_factors_list.append(f"Temperatura alta: {current_temp:.1f}°C")
            elif current_temp < 15:
                risk_factors_list.append(f"Temperatura baja: {current_temp:.1f}°C")

        if current_hum is not None:
            hum_std = hum_data.std()
            hum_mean = hum_data.mean()
            
            # Humedad peligrosa: < 30% o > 80%
            if current_hum < 30 or current_hum > 80:
                risk_factors["humidity_critical"] = 0.9
                risk_factors_list.append(f"Humedad crítica: {current_hum:.1f}%")
            elif current_hum < 40 or current_hum > 70:
                risk_factors["humidity_warn"] = 0.5
                risk_factors_list.append(f"Humedad fuera de rango: {current_hum:.1f}%")
            else:
                hum_deviation = abs(current_hum - hum_mean) / (hum_std + 0.1)
                risk_factors["humidity_deviation"] = min(1.0, hum_deviation / 3.0)

        # 2️⃣ Factor: Tendencias
        trends = self._analyze_trends(df)
        if trends.get("temperatura", {}).get("slope", 0) > 0.1:
            risk_factors["temp_increasing"] = 0.3
            risk_factors_list.append("Temperatura en aumento")
        if trends.get("humedad", {}).get("slope", 0) > 0.15:
            risk_factors["humidity_increasing"] = 0.4
            risk_factors_list.append("Humedad en aumento")

        # Calcular risk score promedio
        if risk_factors:
            overall_risk = sum(risk_factors.values()) / len(risk_factors)
        else:
            overall_risk = 0.0

        # Calcular confianza basada en cantidad de factores detectados
        confidence = min(0.95, 0.5 + len(risk_factors) * 0.15)
        if len(df) > 100:
            confidence = min(0.98, confidence + 0.1)

        # Clasificación de riesgo
        if overall_risk < 0.3:
            risk_level = "BAJO"
            color = "🟢 Verde"
        elif overall_risk < 0.6:
            risk_level = "MEDIO"
            color = "🟡 Amarillo"
        else:
            risk_level = "ALTO"
            color = "🔴 Rojo"

        return {
            "overall_score": round(overall_risk, 3),
            "level": risk_level,
            "color": color,
            "confidence": round(confidence, 3),
            "factors": {k: round(v, 3) for k, v in risk_factors.items()},
            "factors_list": risk_factors_list,  # Lista legible para IA
        }

    # =====================
    # Funciones auxiliares
    # =====================

    def _get_trend_direction(self, data: pd.Series) -> str:
        """Determina dirección de tendencia."""
        if len(data) < 2:
            return "Indefinido"
        if data.iloc[-1] > data.iloc[0]:
            return "↑ Aumentando"
        elif data.iloc[-1] < data.iloc[0]:
            return "↓ Disminuyendo"
        else:
            return "→ Estable"

    def _calculate_trend_strength(self, data: pd.Series) -> float:
        """Calcula fuerza de la tendencia (0-1)."""
        if len(data) < 2:
            return 0.0
        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data)
        return min(1.0, abs(r_value) ** 2)

    def _estimate_days_to_critical(self, data: pd.Series, slope: float, metric: str) -> float:
        """
        Estima días hasta un valor crítico basado en pendiente actual.
        """
        if slope == 0:
            return float("inf")

        current = data.iloc[-1]
        
        if metric == "temperatura":
            # Temperatura crítica: > 35°C o < 10°C
            critical_high = 35.0
            critical_low = 10.0
            if slope > 0:
                days = (critical_high - current) / slope / 24
            else:
                days = (critical_low - current) / slope / 24
        else:  # humedad
            # Humedad crítica: > 85% o < 25%
            critical_high = 85.0
            critical_low = 25.0
            if slope > 0:
                days = (critical_high - current) / slope / 24
            else:
                days = (critical_low - current) / slope / 24

        return max(0.0, round(days, 2))

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpreta correlación entre variables."""
        if abs(correlation) < 0.2:
            return "Prácticamente nula"
        elif abs(correlation) < 0.5:
            return "Débil"
        elif abs(correlation) < 0.7:
            return f"{'Positiva' if correlation > 0 else 'Negativa'} moderada"
        else:
            return f"{'Positiva' if correlation > 0 else 'Negativa'} fuerte"

    def generate_ai_explanation(
        self,
        risk_assessment: Dict,
        temperature_analysis: Dict,
        humidity_analysis: Dict,
        trends_analysis: Dict,
        gemini_key: str
    ) -> Optional[Dict]:
        """
        Genera una explicación inteligente del riesgo usando Google Gemini.
        
        Args:
            risk_assessment: Resultado de _compute_risk_assessment()
            temperature_analysis: Análisis de temperatura
            humidity_analysis: Análisis de humedad
            trends_analysis: Análisis de tendencias
            gemini_key: Clave de API de Google Gemini
            
        Returns:
            Dict con 'explanation' y 'confidence', o None si falla
        """
        if not HAS_GEMINI or not gemini_key:
            return None
        
        try:
            # Preparar información para el prompt
            risk_level = risk_assessment.get("level", "DESCONOCIDO")
            risk_score = risk_assessment.get("overall_score", 0)
            factors_list = risk_assessment.get("factors_list", [])
            
            temp_current = temperature_analysis.get("current", "N/A")
            temp_trend = temperature_analysis.get("trend_direction", "N/A")
            
            hum_current = humidity_analysis.get("current", "N/A")
            hum_trend = humidity_analysis.get("trend_direction", "N/A")
            
            # Construir prompt para Gemini
            prompt = f"""Analiza este nivel de riesgo ambiental y proporciona una explicación clara y concisa en español de 2-3 párrafos:

**Nivel de Riesgo:** {risk_level} (Score: {risk_score:.2f})

**Datos Ambientales Actuales:**
- Temperatura: {temp_current}°C - Tendencia: {temp_trend}
- Humedad: {hum_current}% - Tendencia: {hum_trend}

**Factores Identificados:**
{chr(10).join('• ' + f for f in factors_list) if factors_list else '• No hay factores críticos detectados'}

Por favor, proporciona:
1. Por qué hay este nivel de riesgo (máximo 2 líneas)
2. Qué variables lo causan (máximo 2 líneas)
3. Una recomendación de acción (máximo 1 línea)

Usa formato markdown con emojis para mayor claridad. Responde en español colombiano. Sé conciso y técnico."""

            # Llamar a Google Gemini
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=300
                )
            )
            
            explanation = response.text.strip()
            
            LOG.info(f"Generated Gemini explanation for risk level {risk_level}")
            
            return {
                "explanation": explanation,
                "confidence": 0.9
            }
            
        except Exception as e:
            LOG.warning(f"Failed to generate Gemini explanation: {e}")
            return None


class TaskRecommender:
    """
    Genera recomendaciones de tareas inteligentes basadas en análisis.
    """

    def __init__(self):
        self.task_templates = {
            "high_humidity": {
                "tarea": "🌡️ Intervención urgente: Reducir humedad crítica",
                "duracion_horas": 2,
                "riesgo": 0.95,
                "prioridad": "CRÍTICA",
                "acciones": [
                    "Activar deshumidificadores",
                    "Verificar hermeticidad de puertas/ventanas",
                    "Aumentar ventilación forzada",
                    "Revisar sistema de aire acondicionado",
                ]
            },
            "low_humidity": {
                "tarea": "🌡️ Aumentar humedad ambiental",
                "duracion_horas": 4,
                "riesgo": 0.7,
                "prioridad": "ALTA",
                "acciones": [
                    "Activar humidificadores",
                    "Verificar tuberías de agua",
                    "Aumentar plantas en el área",
                ]
            },
            "high_temp": {
                "tarea": "❄️ Control de temperatura: Reducir calor",
                "duracion_horas": 3,
                "riesgo": 0.85,
                "prioridad": "ALTA",
                "acciones": [
                    "Verificar sistemas de refrigeración",
                    "Aumentar ventilación",
                    "Revisar aislamiento térmico",
                    "Controlar fuentes de calor",
                ]
            },
            "low_temp": {
                "tarea": "🔥 Control de temperatura: Aumentar calor",
                "duracion_horas": 3,
                "riesgo": 0.75,
                "prioridad": "ALTA",
                "acciones": [
                    "Verificar sistemas de calefacción",
                    "Revisar puertas/ventanas abiertas",
                    "Aumentar aislamiento",
                ]
            },
            "anomaly": {
                "tarea": "🔍 Investigar anomalía ambiental",
                "duracion_horas": 2,
                "riesgo": 0.6,
                "prioridad": "MEDIA",
                "acciones": [
                    "Calibrar sensores",
                    "Verificar fuentes de lectura errónea",
                    "Revisar sensores",
                ]
            },
            "trending_danger": {
                "tarea": "📈 Monitorear tendencia peligrosa",
                "duracion_horas": 1,
                "riesgo": 0.5,
                "prioridad": "MEDIA",
                "acciones": [
                    "Aumentar frecuencia de monitoreo",
                    "Preparar intervenciones preventivas",
                ]
            }
        }

    def generate_recommendations(self, analysis: Dict, predictions: List[Dict] = None) -> List[Dict]:
        """
        Genera recomendaciones de tareas basadas en análisis y predicciones.
        
        Args:
            analysis: Resultado de analyze_dataframe()
            predictions: (Opcional) Predicciones del modelo ML
            
        Returns:
            Lista de tareas recomendadas
        """
        if not analysis.get("ok"):
            return []

        analysis_data = analysis.get("analysis", {})
        tasks = []

        # 1️⃣ Basarse en riesgo general
        risk_assessment = analysis_data.get("risk_assessment", {})
        risk_level = risk_assessment.get("level", "BAJO")

        if risk_level == "ALTO":
            # Detectar tipo específico de problema
            humidity = analysis_data.get("humidity", {})
            temperature = analysis_data.get("temperature", {})
            
            current_hum = humidity.get("current")
            current_temp = temperature.get("current")

            if current_hum and current_hum > 80:
                tasks.append(self._create_task("high_humidity", risk_assessment.get("overall_score", 0.9)))
            elif current_hum and current_hum < 30:
                tasks.append(self._create_task("low_humidity", risk_assessment.get("overall_score", 0.7)))

            if current_temp and current_temp > 32:
                tasks.append(self._create_task("high_temp", risk_assessment.get("overall_score", 0.85)))
            elif current_temp and current_temp < 12:
                tasks.append(self._create_task("low_temp", risk_assessment.get("overall_score", 0.75)))

        elif risk_level == "MEDIO":
            trends = analysis_data.get("trends", {})
            temp_trend = trends.get("temperatura", {})
            hum_trend = trends.get("humedad", {})

            if temp_trend.get("significant") and temp_trend.get("slope", 0) > 0.05:
                tasks.append(self._create_task("trending_danger", 0.5))

            if hum_trend.get("significant") and hum_trend.get("slope", 0) > 0.1:
                tasks.append(self._create_task("trending_danger", 0.5))

        # 2️⃣ Detectar anomalías
        anomalies = analysis_data.get("anomalies", {})
        if (anomalies.get("temperature", {}).get("zscore") or 
            anomalies.get("humidity", {}).get("zscore")):
            tasks.append(self._create_task("anomaly", 0.6))

        # 3️⃣ Predicciones del modelo (si existen)
        if predictions:
            for pred in predictions[-24:]:  # Últimas 24 horas
                if pred.get("pred") == 2:  # Riesgo alto
                    if not any(t["tarea"] == self.task_templates["high_humidity"]["tarea"] for t in tasks):
                        tasks.append(self._create_task("high_humidity", 0.9))

        return tasks

    def _create_task(self, task_type: str, risk_score: float) -> Dict:
        """Crea una tarea a partir de un template."""
        template = self.task_templates.get(task_type, {})
        
        now = datetime.now()
        duration = template.get("duracion_horas", 2)
        fin_time = now + timedelta(hours=duration)

        task = {
            "tarea": template.get("tarea", ""),
            "tipo": task_type,
            "inicio": now.isoformat(),
            "fin": fin_time.isoformat(),
            "riesgo": float(min(1.0, risk_score * template.get("riesgo", 0.5))),
            "motivo": f"Detectado por análisis IA: {task_type}",
            "prioridad": template.get("prioridad", "MEDIA"),
            "acciones_recomendadas": template.get("acciones", []),
        }
        return convert_numpy_values(task)
