from __future__ import annotations

import shutil

from config import settings
from repository.db_manager import DBManager
from service.ollama_status_service import OllamaStatus, OllamaStatusService


class StatusViewModel:
    def __init__(
        self,
        db_manager: DBManager | None = None,
        ollama_status_service: OllamaStatusService | None = None,
    ) -> None:
        self.db_manager = db_manager or DBManager()
        self.ollama_status_service = ollama_status_service or OllamaStatusService()

    def get_status(self) -> dict[str, str]:
        try:
            self.db_manager.initialize()
            db_status = "Ready"
        except Exception as exc:
            db_status = f"Error: {exc}"
        recent_logs = self.get_recent_logs()

        return {
            "model_path": str(settings.YOLO_MODEL_PATH),
            "yolo_model": "Ready" if settings.YOLO_MODEL_PATH.exists() else "Missing",
            "vlm": "확인 중...",
            "vlm_detail": (
                f"Host: {self.ollama_status_service.host} | "
                f"Model: {self.ollama_status_service.model_name}"
            ),
            "database_path": str(settings.DATABASE_PATH),
            "database": db_status,
            "gpu": "CUDA check requires Torch runtime" if shutil.which("nvidia-smi") is None else "NVIDIA driver visible",
            "log_path": str(settings.ERROR_LOG_PATH),
            "logs": "Ready" if settings.LOG_DIR.exists() else "Missing",
            "recent_logs": recent_logs,
        }

    def check_vlm_status(self) -> OllamaStatus:
        return self.ollama_status_service.check_status()

    def get_recent_logs(self, max_lines: int = 80) -> str:
        log_path = settings.ERROR_LOG_PATH
        if not log_path.exists():
            return "아직 기록된 실행 로그가 없습니다."
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"로그 파일을 읽을 수 없습니다: {exc}"
        if not lines:
            return "로그 파일이 비어 있습니다."
        return "\n".join(lines[-max_lines:])
