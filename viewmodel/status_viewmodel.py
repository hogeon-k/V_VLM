from __future__ import annotations

import shutil

from config import settings
from repository.db_manager import DBManager


class StatusViewModel:
    def __init__(self, db_manager: DBManager | None = None) -> None:
        self.db_manager = db_manager or DBManager()

    def get_status(self) -> dict[str, str]:
        try:
            self.db_manager.initialize()
            db_status = "Ready"
        except Exception as exc:
            db_status = f"Error: {exc}"

        return {
            "model_path": str(settings.YOLO_MODEL_PATH),
            "yolo_model": "Ready" if settings.YOLO_MODEL_PATH.exists() else "Missing",
            "vlm": "Configured for local Ollama",
            "database_path": str(settings.DATABASE_PATH),
            "database": db_status,
            "gpu": "CUDA check requires Torch runtime" if shutil.which("nvidia-smi") is None else "NVIDIA driver visible",
            "log_path": str(settings.ERROR_LOG_PATH),
            "logs": "Ready" if settings.LOG_DIR.exists() else "Missing",
        }
