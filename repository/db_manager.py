from __future__ import annotations

import sqlite3
from pathlib import Path

from config.settings import DATABASE_PATH


class DBManager:
    def __init__(self, database_path: Path = DATABASE_PATH) -> None:
        self.database_path = database_path

    def get_connection(self) -> sqlite3.Connection:
        """Return a SQLite connection for repository classes."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
