# ai_scheduler.py
import logging
import time
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

# cargar variables del archivo api.env
load_dotenv("api.env")

import predictor

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("ai_scheduler")

def job_daily():
    lat = float(os.environ.get("LAT", "10.426593484472422"))
    lon = float(os.environ.get("LON", "-75.53768518832231"))
    key = os.environ.get("OPENWEATHER_KEY")

    LOG.info(f"[Scheduler] usando OpenWeather lat={lat}, lon={lon}, key={'OK' if key else 'MISSING'}")

    pred_days = predictor.predict_next_days(
        num_days=7,
        lat=lat,
        lon=lon,
        owm_key=key
    )

def start_scheduler(demo=False):
    sched = BackgroundScheduler()
    if demo:
        sched.add_job(job_daily, 'interval', minutes=1, next_run_time=datetime.now())
    else:
        sched.add_job(job_daily, 'interval', hours=24, next_run_time=datetime.now())

    sched.start()
    LOG.info("Scheduler started (demo=%s)", demo)

    try:
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()

if __name__ == "__main__":
    start_scheduler(demo=False)
