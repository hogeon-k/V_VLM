from __future__ import annotations

from model.statistics_result import StatisticsResult
from repository.db_manager import DBManager


class StatisticsService:
    def __init__(self, db_manager: DBManager | None = None) -> None:
        self.db_manager = db_manager or DBManager()

    def build_summary(self) -> StatisticsResult:
        self.db_manager.initialize()
        with self.db_manager.get_connection() as connection:
            status_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_count,
                    SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) AS ok_count,
                    SUM(CASE WHEN status = 'NG' THEN 1 ELSE 0 END) AS ng_count
                FROM inspections
                """
            ).fetchone()
            defect_rows = connection.execute(
                """
                SELECT defect_type, COUNT(*) AS count
                FROM defects
                GROUP BY defect_type
                ORDER BY count DESC, defect_type
                """
            ).fetchall()

        return StatisticsResult(
            total_count=int(status_row["total_count"] or 0),
            ok_count=int(status_row["ok_count"] or 0),
            ng_count=int(status_row["ng_count"] or 0),
            defect_type_counts={
                str(row["defect_type"]): int(row["count"])
                for row in defect_rows
            },
        )
