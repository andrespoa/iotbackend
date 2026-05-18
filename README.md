# Sistema IoT de Monitoreo Ambiental Inteligente

Backend robusto diseñado para la gestión de datos de sensores IoT, análisis predictivo y mantenimiento preventivo automatizado.

## 🚀 Tecnologías Principales

- **Framework:** Flask (Python)
- **Base de Datos:** Supabase (PostgreSQL) - Migrado desde MySQL.
- **IA/ML:** Google Gemini API (Análisis semántico) y Scikit-learn (Modelos de riesgo).
- **Despliegue:** Configurado para Render con Gunicorn.

## 🧠 Funcionalidades Inteligentes

- **Análisis Ambiental Avanzado:** Detección de anomalías mediante Z-Score e IQR.
- **Predicción de Riesgo:** Motor de inferencia que evalúa tendencias de temperatura y humedad.
- **Generación de Tareas:** Creación automática de planes de acción preventivos en la base de datos.
- **Explicación IA:** Integración con Gemini para traducir métricas complejas en recomendaciones legibles.

## 🛠️ Instalación y Uso

1. Instalar dependencias: `pip install -r requirements.txt`.
2. Configurar el archivo `api.env` con las credenciales de Supabase y las API Keys necesarias.
3. Ejecutar el servidor: `python app.py`.

## ☁️ Producción

Este repositorio está listo para ser desplegado en **Render**. 
Utiliza el `Procfile` y `render.yaml` adjuntos para gestionar el servidor web y el scheduler de tareas en segundo plano.

---
*Proyecto de Grado - Ingeniería de Sistemas*