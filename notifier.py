# notifier.py
import supabase_db
import logging

LOG = logging.getLogger("notifier")

def get_pending_tasks(limit=50):
    """Obtiene tareas pendientes de Supabase"""
    return supabase_db.get_tareas(estado="pendiente", limit=limit)

def mark_task_done(task_id):
    """Marca una tarea como completada en Supabase"""
    try:
        return supabase_db.update_tarea(task_id, estado="completada")
    except Exception as e:
        LOG.exception("mark_task_done failed: %s", e)
        return False
