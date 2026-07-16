from __future__ import annotations

import sqlite3
from pathlib import Path

from config.settings import DATABASE_PATH


class DBManager:
    def __init__(self, database_path: Path = DATABASE_PATH, schema_path: Path | None = None) -> None:
        self.database_path = Path(database_path)
        self.schema_path = schema_path or Path(__file__).with_name("schema.sql")

    def get_connection(self) -> sqlite3.Connection:
        """Return a SQLite connection for repository classes."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        """Create the application schema when it does not exist yet."""
        with self.get_connection() as connection:
            connection.executescript(self.schema_path.read_text(encoding="utf-8"))
