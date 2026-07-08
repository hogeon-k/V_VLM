from __future__ import annotations

from repository.db_manager import DBManager


class InspectionRepository:
    def __init__(self, db_manager: DBManager | None = None) -> None:
        self.db_manager = db_manager or DBManager()

    def save(self, inspection_result: object) -> int:
        # TODO: Persist an inspection row and return its generated id.
        raise NotImplementedError

    def find_by_id(self, inspection_id: int) -> object | None:
        # TODO: Load one inspection result by id.
        raise NotImplementedError

    def find_recent(self, limit: int = 100) -> list[object]:
        # TODO: Load recent inspection rows.
        raise NotImplementedError
