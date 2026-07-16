from model.defect_info import Detection
from model.inspection_result import InspectionResult
from repository.db_manager import DBManager
from repository.inspection_repository import InspectionRepository
from service.statistics_service import StatisticsService


def test_statistics_service_returns_empty_summary(tmp_path) -> None:
    summary = StatisticsService(DBManager(tmp_path / "inspection.sqlite3")).build_summary()

    assert summary.total_count == 0
    assert summary.ok_count == 0
    assert summary.ng_count == 0
    assert summary.defect_type_counts == {}


def test_statistics_service_aggregates_from_sql(tmp_path) -> None:
    manager = DBManager(tmp_path / "inspection.sqlite3")
    repository = InspectionRepository(manager)
    repository.save(InspectionResult(source_image_path=tmp_path / "ok.png", status="OK"))
    repository.save(
        InspectionResult(
            source_image_path=tmp_path / "ng.png",
            status="NG",
            detections=[
                Detection(0, "short", 0.9, 1, 2, 3, 4),
                Detection(1, "open_circuit", 0.8, 5, 6, 7, 8),
            ],
        )
    )

    summary = StatisticsService(manager).build_summary()

    assert summary.total_count == 2
    assert summary.ok_count == 1
    assert summary.ng_count == 1
    assert summary.defect_type_counts == {"open_circuit": 1, "short": 1}
