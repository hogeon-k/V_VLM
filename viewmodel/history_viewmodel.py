from __future__ import annotations

from repository.inspection_repository import InspectionRepository


class HistoryViewModel:
    def __init__(self, inspection_repository: InspectionRepository | None = None) -> None:
        self.inspection_repository = inspection_repository or InspectionRepository()

    def load_recent(self) -> list[object]:
        # TODO: Load inspection history through the repository layer.
        return []
