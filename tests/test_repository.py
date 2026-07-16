from repository.db_manager import DBManager
from repository.inspection_repository import InspectionRepository
from model.defect_info import Detection
from model.inspection_result import InspectionResult


def test_db_manager_uses_configured_path(tmp_path) -> None:
    db_path = tmp_path / "inspection.sqlite3"
    manager = DBManager(db_path)

    with manager.get_connection() as connection:
        enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert db_path.exists()
    assert enabled == 1


def test_repository_saves_and_loads_inspection_with_defects(tmp_path) -> None:
    manager = DBManager(tmp_path / "inspection.sqlite3")
    repository = InspectionRepository(manager)
    result = InspectionResult(
        source_image_path=tmp_path / "input.png",
        result_image_path=tmp_path / "result.png",
        status="NG",
        detections=[Detection(0, "short", 0.95, 1, 2, 3, 4)],
        vlm_explanation="short near top-left",
    )

    inspection_id = repository.save(result)
    loaded = repository.find_by_id(inspection_id)

    assert loaded is not None
    assert loaded.id == inspection_id
    assert loaded.status == "NG"
    assert loaded.defect_count == 1
    assert loaded.defects[0].defect_type == "short"
    assert loaded.vlm_description == "short near top-left"


def test_repository_filters_by_status_and_defect_type(tmp_path) -> None:
    manager = DBManager(tmp_path / "inspection.sqlite3")
    repository = InspectionRepository(manager)
    repository.save(InspectionResult(source_image_path=tmp_path / "ok.png", status="OK"))
    repository.save(
        InspectionResult(
            source_image_path=tmp_path / "ng.png",
            status="NG",
            detections=[Detection(0, "missing_hole", 0.8, 10, 20, 30, 40)],
        )
    )

    assert [result.status for result in repository.search(status="OK")] == ["OK"]
    filtered = repository.search(status="NG", defect_type="missing_hole")
    assert len(filtered) == 1
    assert filtered[0].defects[0].defect_type == "missing_hole"
