from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_IMAGE_DIR = PROJECT_ROOT / "data" / "input_images"
RESULT_IMAGE_DIR = PROJECT_ROOT / "data" / "result_images"

MODEL_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_PATH = MODEL_DIR / "best.pt"

DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "inspection_results.sqlite3"

LOG_DIR = PROJECT_ROOT / "logs"
ERROR_LOG_PATH = LOG_DIR / "error.log"
