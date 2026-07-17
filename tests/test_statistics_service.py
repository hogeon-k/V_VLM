from datetime import datetime

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
    assert summary.ng_rate == 0.0


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


def test_statistics_dashboard_filters_by_date_and_sorts_defects(tmp_path) -> None:
    manager = DBManager(tmp_path / "inspection.sqlite3")
    repository = InspectionRepository(manager)
    repository.save(
        InspectionResult(
            source_image_path=tmp_path / "old.png",
            status="NG",
            detections=[Detection(0, "old", 0.9, 1, 2, 3, 4)],
            inspected_at=datetime(2026, 6, 30, 12, 0, 0),
        )
    )
    repository.save(
        InspectionResult(
            source_image_path=tmp_path / "ng1.png",
            status="NG",
            detections=[
                Detection(0, "short", 0.9, 1, 2, 3, 4),
                Detection(1, "short", 0.8, 5, 6, 7, 8),
                Detection(2, "open_circuit", 0.7, 9, 10, 11, 12),
            ],
            inspected_at=datetime(2026, 7, 1, 9, 0, 0),
        )
    )
    repository.save(
        InspectionResult(
            source_image_path=tmp_path / "ok.png",
            status="OK",
            inspected_at=datetime(2026, 7, 2, 9, 0, 0),
        )
    )

    dashboard = StatisticsService(manager).build_dashboard(
        start_date="2026-07-01",
        end_date="2026-07-02",
    )

    assert dashboard.summary.total_count == 2
    assert dashboard.summary.ok_count == 1
    assert dashboard.summary.ng_count == 1
    assert dashboard.summary.ng_rate == 50.0
    assert [(item.defect_type, item.count) for item in dashboard.defect_counts] == [
        ("short", 2),
        ("open_circuit", 1),
    ]
    assert [(point.period_label, point.count) for point in dashboard.ng_trend] == [
        ("2026-07-01", 1),
        ("2026-07-02", 0),
    ]
    assert [(item.defect_type, item.count) for item in dashboard.top_defect_types] == [("short", 2)]


def test_statistics_dashboard_handles_top_defect_ties_and_status_aliases(tmp_path) -> None:
    manager = DBManager(tmp_path / "inspection.sqlite3")
    repository = InspectionRepository(manager)
    repository.save(
        InspectionResult(
            source_image_path=tmp_path / "pass.png",
            status="PASS",
            inspected_at=datetime(2026, 7, 1, 8, 0, 0),
        )
    )
    repository.save(
        InspectionResult(
            source_image_path=tmp_path / "fail.png",
            status="FAIL",
            detections=[
                Detection(0, "short", 0.9, 1, 2, 3, 4),
                Detection(1, "open_circuit", 0.8, 5, 6, 7, 8),
            ],
            inspected_at=datetime(2026, 7, 1, 9, 0, 0),
        )
    )

    dashboard = StatisticsService(manager).build_dashboard()

    assert dashboard.summary.ok_count == 1
    assert dashboard.summary.ng_count == 1
    assert [(item.defect_type, item.count) for item in dashboard.top_defect_types] == [
        ("open_circuit", 1),
        ("short", 1),
    ]
