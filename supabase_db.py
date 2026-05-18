# supabase_db.py
"""
Conexión a Supabase PostgreSQL para reemplazar MySQL local
"""
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    print("⚠️ supabase-py no instalado. Ejecuta: pip install supabase")
    raise

load_dotenv("api.env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL o SUPABASE_KEY no configurados en api.env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

LOG = logging.getLogger("supabase_db")

# ========================
# LECTURAS (Datos IoT)
# ========================

def insert_lectura(temperatura: float, humedad: float) -> Dict:
    """Guardar lectura de sensores"""
    try:
        response = supabase.table("lecturas").insert({
            "temperatura": temperatura,
            "humedad": humedad,
            "fecha_hora": datetime.now().isoformat()
        }).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        LOG.error(f"Error al guardar lectura: {e}")
        raise

def get_last_lectura() -> Optional[Dict]:
    """Obtener última lectura"""
    try:
        response = supabase.table("lecturas").select("*").order("fecha_hora", desc=True).limit(1).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        LOG.error(f"Error al obtener última lectura: {e}")
        return None

def get_historial(limit: int = 50, offset: int = 0) -> List[Dict]:
    """Obtener histórico de lecturas"""
    try:
        response = (
            supabase.table("lecturas")
            .select("*")
            .order("fecha_hora", desc=True)
            .range(offset, offset + limit)
            .execute()
        )
        return response.data if response.data else []
    except Exception as e:
        LOG.error(f"Error al obtener historial: {e}")
        return []

def get_lecturas_rango(fecha_inicio: str, fecha_fin: str) -> List[Dict]:
    """Obtener lecturas en un rango de fechas"""
    try:
        response = (
            supabase.table("lecturas")
            .select("*")
            .gte("fecha_hora", fecha_inicio)
            .lte("fecha_hora", fecha_fin)
            .order("fecha_hora", desc=False)
            .execute()
        )
        return response.data if response.data else []
    except Exception as e:
        LOG.error(f"Error al obtener lecturas en rango: {e}")
        return []

# ========================
# PRONÓSTICOS (IA)
# ========================

def save_pronostico(detalle_json: dict, resumen: dict = None) -> Optional[Dict]:
    """Guardar pronóstico generado por IA"""
    try:
        response = supabase.table("pronosticos").insert({
            "fecha_prediccion": datetime.now().isoformat(),
            "detalle_json": detalle_json,
            "resumen": resumen or {},
            "created_at": datetime.now().isoformat()
        }).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        LOG.error(f"Error al guardar pronóstico: {e}")
        raise

def get_last_pronostico() -> Optional[Dict]:
    """Obtener último pronóstico"""
    try:
        response = supabase.table("pronosticos").select("*").order("fecha_prediccion", desc=True).limit(1).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        LOG.error(f"Error al obtener pronóstico: {e}")
        return None

# ========================
# TAREAS (Mantenimiento)
# ========================

def insert_tarea(
    tarea: str,
    inicio: datetime,
    fin: Optional[datetime] = None,
    riesgo: str = "bajo",
    pronostico_id: Optional[int] = None
) -> Optional[Dict]:
    """Guardar tarea de mantenimiento preventivo"""
    try:
        response = supabase.table("tareas").insert({
            "tarea": tarea,
            "inicio": inicio.isoformat() if isinstance(inicio, datetime) else inicio,
            "fin": fin.isoformat() if isinstance(fin, datetime) else fin,
            "riesgo": riesgo,
            "pronostico_id": pronostico_id,
            "estado": "pendiente",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        LOG.error(f"Error al guardar tarea: {e}")
        raise

def get_tareas(estado: str = None, limit: int = 50) -> List[Dict]:
    """Obtener tareas de mantenimiento"""
    try:
        query = supabase.table("tareas").select("*")
        if estado:
            query = query.eq("estado", estado)
        response = query.order("created_at", desc=True).limit(limit).execute()
        return response.data if response.data else []
    except Exception as e:
        LOG.error(f"Error al obtener tareas: {e}")
        return []

def update_tarea(tarea_id: int, estado: str) -> Optional[Dict]:
    """Actualizar estado de tarea"""
    try:
        response = supabase.table("tareas").update({
            "estado": estado,
            "updated_at": datetime.now().isoformat()
        }).eq("id", tarea_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        LOG.error(f"Error al actualizar tarea: {e}")
        raise

def delete_tarea(tarea_id: int):
    """Eliminar tarea"""
    try:
        supabase.table("tareas").delete().eq("id", tarea_id).execute()
    except Exception as e:
        LOG.error(f"Error al eliminar tarea: {e}")
        raise

# ========================
# LOCKING (Mutex)
# ========================

class TaskGenerationLock:
    """Sistema de mutex para prevenir generación simultánea de tareas"""

    LOCK_NAME = "task_generation"
    LOCK_TIMEOUT_SECONDS = 300

    @staticmethod
    def acquire_lock(process_id: str) -> bool:
        """Intenta adquirir el lock"""
        try:
            # Obtener lock actual
            response = supabase.table("task_generation_lock").select("*").eq("id", 1).execute()
            lock = response.data[0] if response.data else None

            if not lock:
                # Si no existe la fila inicial con id=1, intentamos crearla para adquirir el lock
                try:
                    supabase.table("task_generation_lock").insert({
                        "id": 1, "process_id": process_id, "locked_at": datetime.now().isoformat(), "running": True
                    }).execute()
                    return True
                except Exception:
                    return False

            if lock and lock.get("process_id"):
                # Hay un lock activo, verificar si expiró
                locked_time_str = lock.get("locked_at")
                if not locked_time_str:
                    # Si no hay fecha registrada, permitimos tomar el control
                    pass
                else:
                    locked_time = datetime.fromisoformat(locked_time_str)
                    elapsed = (datetime.now() - locked_time).total_seconds()
                    if elapsed < TaskGenerationLock.LOCK_TIMEOUT_SECONDS:
                        return False  # Lock aún válido

            # Adquirir lock
            res = supabase.table("task_generation_lock").update({
                "process_id": process_id,
                "locked_at": datetime.now().isoformat(),
                "running": True
            }).eq("id", 1).execute()

            # Retorna True solo si se actualizó la fila correctamente
            return len(res.data) > 0
            
        except Exception as e:
            LOG.error(f"Error al adquirir lock: {e}")
            return False

    @staticmethod
    def release_lock(process_id: str):
        """Liberar lock"""
        try:
            supabase.table("task_generation_lock").update({
                "process_id": None,
                "locked_at": None,
                "running": False
            }).eq("id", 1).execute()
        except Exception as e:
            LOG.error(f"Error al liberar lock: {e}")

# ========================
# COMPATIBILITY LAYER (para no romper código existente)
# ========================

def get_connection():
    """Dummy para compatibilidad"""
    return None

# Test de conexión
if __name__ == "__main__":
    print("🔍 Probando conexión a Supabase...")
    try:
        last = get_last_lectura()
        print(f"✅ Conexión OK. Última lectura: {last}")
    except Exception as e:
        print(f"❌ Error: {e}")
