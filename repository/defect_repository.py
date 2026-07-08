from __future__ import annotations

from repository.db_manager import DBManager


class DefectRepository:
    def __init__(self, db_manager: DBManager | None = None) -> None:
        self.db_manager = db_manager or DBManager()

    def save_many(self, inspection_id: int, defects: list[object]) -> None:
        # TODO: Persist defects linked to the given inspection id.
        raise NotImplementedError

    def find_by_inspection_id(self, inspection_id: int) -> list[object]:
        # TODO: Load all defects for one inspection result.
        raise NotImplementedError
