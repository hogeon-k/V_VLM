from __future__ import annotations

from model.inspection_result import InspectionResult
from repository.inspection_repository import InspectionRepository
from service.inspection_service import InspectionService
from service.inspection_history_service import DeletionReport, InspectionHistoryService


class HistoryViewModel:
    def __init__(
        self,
        inspection_repository: InspectionRepository | None = None,
        history_service: InspectionHistoryService | None = None,
        inspection_service: InspectionService | None = None,
    ) -> None:
        self.inspection_repository = inspection_repository or InspectionRepository()
        self.history_service = history_service or InspectionHistoryService(self.inspection_repository)
        self.inspection_service = inspection_service or InspectionService(
            inspection_repository=self.inspection_repository,
        )

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

    def run_vlm_for_inspection(self, inspection_id: int) -> InspectionResult:
        return self.inspection_service.run_vlm_for_inspection(inspection_id)

    def defect_types(self) -> list[str]:
        return self.inspection_repository.list_defect_types()

    def history_count(self) -> int:
        return self.history_service.count()

    def delete_history(self, inspection_id: int) -> DeletionReport:
        return self.history_service.delete_history(inspection_id)

    def delete_all_history(self) -> DeletionReport:
        return self.history_service.delete_all_history()
