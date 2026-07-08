from __future__ import annotations

from config import settings


class StatusViewModel:
    def get_status(self) -> dict[str, str]:
        # TODO: Check model file, database, and log directory readiness.
        return {
            "model_path": str(settings.YOLO_MODEL_PATH),
            "database_path": str(settings.DATABASE_PATH),
            "log_path": str(settings.ERROR_LOG_PATH),
        }
