from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3

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

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a transactional connection and always close it."""
        connection = self.get_connection()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the application schema when it does not exist yet."""
        with self.connection() as connection:
            connection.executescript(self.schema_path.read_text(encoding="utf-8"))
            _ensure_inspection_vlm_columns(connection)
            _ensure_defect_class_id_column(connection)


def _ensure_inspection_vlm_columns(connection: sqlite3.Connection) -> None:
    existing_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(inspections)").fetchall()
    }
    migrations = {
        "vlm_status": "ALTER TABLE inspections ADD COLUMN vlm_status TEXT NOT NULL DEFAULT 'NOT_REQUESTED'",
        "vlm_error_message": "ALTER TABLE inspections ADD COLUMN vlm_error_message TEXT",
        "vlm_updated_at": "ALTER TABLE inspections ADD COLUMN vlm_updated_at TEXT",
    }
    added_vlm_status = "vlm_status" not in existing_columns
    for column_name, sql in migrations.items():
        if column_name not in existing_columns:
            connection.execute(sql)
    if added_vlm_status:
        connection.execute(
            """
            UPDATE inspections
            SET vlm_status = CASE
                WHEN vlm_description IS NOT NULL AND TRIM(vlm_description) != '' THEN 'COMPLETED'
                ELSE 'NOT_REQUESTED'
            END
            """
        )
    else:
        connection.execute(
            """
            UPDATE inspections
            SET vlm_status = CASE
                WHEN vlm_description IS NOT NULL AND TRIM(vlm_description) != '' THEN 'COMPLETED'
                ELSE 'NOT_REQUESTED'
            END
            WHERE vlm_status IS NULL OR vlm_status = ''
            """
        )


def _ensure_defect_class_id_column(connection: sqlite3.Connection) -> None:
    existing_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(defects)").fetchall()
    }
    if "class_id" not in existing_columns:
        connection.execute(
            "ALTER TABLE defects ADD COLUMN class_id INTEGER NOT NULL DEFAULT -1"
        )
