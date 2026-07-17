from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from model.inspection_result import InspectionResult
from repository.defect_repository import DefectRepository
from repository.db_manager import DBManager


class InspectionRepository:
    def __init__(
        self,
        db_manager: DBManager | None = None,
        defect_repository: DefectRepository | None = None,
    ) -> None:
        self.db_manager = db_manager or DBManager()
        self.defect_repository = defect_repository or DefectRepository(self.db_manager)

    def save(self, inspection_result: InspectionResult) -> int:
        """Persist an inspection and all defects in one transaction."""
        self.db_manager.initialize()
        inspected_at = inspection_result.inspected_at or datetime.now()
        with self.db_manager.get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO inspections (
                    image_name, original_image_path, result_image_path,
                    status, defect_count, vlm_description, inspected_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inspection_result.image_name,
                    str(inspection_result.original_image_path),
                    str(inspection_result.result_image_path) if inspection_result.result_image_path else None,
                    inspection_result.status,
                    inspection_result.defect_count,
                    inspection_result.vlm_description,
                    inspected_at.isoformat(timespec="seconds"),
                ),
            )
            inspection_id = int(cursor.lastrowid)
            self.defect_repository.save_many(
                inspection_id,
                inspection_result.defects,
                connection=connection,
            )
            inspection_result.id = inspection_id
            return inspection_id

    def find_by_id(self, inspection_id: int) -> InspectionResult | None:
        self.db_manager.initialize()
        with self.db_manager.get_connection() as connection:
            row = connection.execute(
                "SELECT * FROM inspections WHERE id = ?",
                (inspection_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_result(row)

    def count(self) -> int:
        self.db_manager.initialize()
        with self.db_manager.get_connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM inspections").fetchone()
        return int(row["count"])

    def delete_by_id(self, inspection_id: int) -> int:
        """Delete one inspection row and its child defects in one transaction."""
        self.db_manager.initialize()
        with self.db_manager.get_connection() as connection:
            try:
                connection.execute("BEGIN")
                cursor = connection.execute(
                    "DELETE FROM inspections WHERE id = ?",
                    (inspection_id,),
                )
                deleted_count = int(cursor.rowcount)
                connection.commit()
                return deleted_count
            except Exception:
                connection.rollback()
                raise

    def delete_all(self) -> int:
        """Delete all inspection rows and their child defects in one transaction."""
        self.db_manager.initialize()
        with self.db_manager.get_connection() as connection:
            try:
                connection.execute("BEGIN")
                cursor = connection.execute("DELETE FROM inspections")
                deleted_count = int(cursor.rowcount)
                connection.commit()
                return deleted_count
            except Exception:
                connection.rollback()
                raise

    def find_recent(self, limit: int = 100) -> list[InspectionResult]:
        return self.search(limit=limit)

    def search(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        status: str | None = None,
        defect_type: str | None = None,
        limit: int = 500,
    ) -> list[InspectionResult]:
        self.db_manager.initialize()
        where: list[str] = []
        params: list[Any] = []

        if start_date:
            where.append("i.inspected_at >= ?")
            params.append(_normalize_start_date(start_date))
        if end_date:
            where.append("i.inspected_at <= ?")
            params.append(_normalize_end_date(end_date))
        if status and status != "ALL":
            where.append("i.status = ?")
            params.append(status)
        if defect_type:
            where.append(
                "EXISTS (SELECT 1 FROM defects d WHERE d.inspection_id = i.id AND d.defect_type = ?)"
            )
            params.append(defect_type)

        query = "SELECT i.* FROM inspections i"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY i.inspected_at DESC, i.id DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self.db_manager.get_connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_result(row) for row in rows]

    def list_defect_types(self) -> list[str]:
        self.db_manager.initialize()
        with self.db_manager.get_connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT defect_type FROM defects ORDER BY defect_type"
            ).fetchall()
        return [str(row["defect_type"]) for row in rows]

    def _row_to_result(self, row: Any) -> InspectionResult:
        result = InspectionResult(
            source_image_path=Path(row["original_image_path"]),
            id=int(row["id"]),
            result_image_path=Path(row["result_image_path"]) if row["result_image_path"] else None,
            status=row["status"],
            detections=self.defect_repository.find_by_inspection_id(int(row["id"])),
            vlm_explanation=row["vlm_description"],
            inspected_at=_parse_datetime(row["inspected_at"]),
        )
        return result


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalize_start_date(value: str) -> str:
    return f"{value}T00:00:00" if len(value) == 10 else value


def _normalize_end_date(value: str) -> str:
    return f"{value}T23:59:59" if len(value) == 10 else value
