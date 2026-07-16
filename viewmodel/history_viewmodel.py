from __future__ import annotations

from model.inspection_result import InspectionResult
from repository.inspection_repository import InspectionRepository


class HistoryViewModel:
    def __init__(self, inspection_repository: InspectionRepository | None = None) -> None:
        self.inspection_repository = inspection_repository or InspectionRepository()

    def load_recent(self) -> list[InspectionResult]:
        return self.inspection_repository.find_recent()

    def search(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        status: str | None = None,
        defect_type: str | None = None,
    ) -> list[InspectionResult]:
        return self.inspection_repository.search(
            start_date=start_date,
            end_date=end_date,
            status=status,
            defect_type=defect_type,
        )

    def load_detail(self, inspection_id: int) -> InspectionResult | None:
        return self.inspection_repository.find_by_id(inspection_id)

    def defect_types(self) -> list[str]:
        return self.inspection_repository.list_defect_types()
