from __future__ import annotations

import sqlite3

from model.defect_info import Detection
from repository.db_manager import DBManager


class DefectRepository:
    def __init__(self, db_manager: DBManager | None = None) -> None:
        self.db_manager = db_manager or DBManager()

    def save_many(
        self,
        inspection_id: int,
        defects: list[object],
        connection: sqlite3.Connection | None = None,
    ) -> None:
        """Persist defects linked to one inspection."""
        if not defects:
            return

        rows = [
            (
                inspection_id,
                str(getattr(defect, "defect_type", getattr(defect, "class_name", ""))),
                float(getattr(defect, "confidence")),
                int(getattr(defect, "bbox_x1", getattr(defect, "x1"))),
                int(getattr(defect, "bbox_y1", getattr(defect, "y1"))),
                int(getattr(defect, "bbox_x2", getattr(defect, "x2"))),
                int(getattr(defect, "bbox_y2", getattr(defect, "y2"))),
                getattr(defect, "vlm_description", None),
            )
            for defect in defects
        ]

        sql = """
            INSERT INTO defects (
                inspection_id, defect_type, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, vlm_description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        if connection is not None:
            connection.executemany(sql, rows)
            return

        self.db_manager.initialize()
        with self.db_manager.get_connection() as local_connection:
            local_connection.executemany(sql, rows)

    def find_by_inspection_id(self, inspection_id: int) -> list[Detection]:
        self.db_manager.initialize()
        with self.db_manager.get_connection() as connection:
            rows = connection.execute(
                """
                SELECT defect_type, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, vlm_description
                FROM defects
                WHERE inspection_id = ?
                ORDER BY id
                """,
                (inspection_id,),
            ).fetchall()

        return [
            Detection(
                class_id=-1,
                class_name=row["defect_type"],
                confidence=row["confidence"],
                x1=row["bbox_x1"],
                y1=row["bbox_y1"],
                x2=row["bbox_x2"],
                y2=row["bbox_y2"],
                vlm_description=row["vlm_description"],
            )
            for row in rows
        ]
