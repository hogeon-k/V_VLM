from repository.db_manager import DBManager
from repository.inspection_repository import InspectionRepository
from model.defect_info import Detection
from model.inspection_result import InspectionResult
from model.inspection_result import (
    VLM_STATUS_COMPLETED,
    VLM_STATUS_FAILED,
    VLM_STATUS_NOT_REQUESTED,
)


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
    assert loaded.vlm_status == VLM_STATUS_COMPLETED


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


def test_repository_updates_vlm_result_on_existing_inspection(tmp_path) -> None:
    manager = DBManager(tmp_path / "inspection.sqlite3")
    repository = InspectionRepository(manager)
    inspection_id = repository.save(
        InspectionResult(
            source_image_path=tmp_path / "ng.png",
            status="NG",
            detections=[Detection(0, "short", 0.8, 1, 2, 3, 4)],
        )
    )

    assert repository.update_vlm_result(
        inspection_id,
        VLM_STATUS_COMPLETED,
        "VLM explanation",
        None,
    )

    loaded = repository.find_by_id(inspection_id)
    assert loaded is not None
    assert loaded.vlm_status == VLM_STATUS_COMPLETED
    assert loaded.vlm_description == "VLM explanation"
    assert loaded.vlm_error_message is None
    assert loaded.vlm_updated_at is not None
    assert loaded.defect_count == 1


def test_repository_records_failed_vlm_without_removing_yolo_data(tmp_path) -> None:
    manager = DBManager(tmp_path / "inspection.sqlite3")
    repository = InspectionRepository(manager)
    inspection_id = repository.save(
        InspectionResult(
            source_image_path=tmp_path / "ng.png",
            result_image_path=tmp_path / "result.png",
            status="NG",
            detections=[Detection(0, "short", 0.8, 1, 2, 3, 4)],
        )
    )

    assert repository.update_vlm_result(inspection_id, VLM_STATUS_FAILED, None, "boom")

    loaded = repository.find_by_id(inspection_id)
    assert loaded is not None
    assert loaded.status == "NG"
    assert loaded.result_image_path == tmp_path / "result.png"
    assert loaded.defect_count == 1
    assert loaded.vlm_status == VLM_STATUS_FAILED
    assert loaded.vlm_error_message == "boom"


def test_db_manager_adds_vlm_columns_to_existing_schema(tmp_path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    manager = DBManager(db_path)
    with manager.get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                original_image_path TEXT NOT NULL,
                result_image_path TEXT,
                status TEXT NOT NULL,
                defect_count INTEGER NOT NULL DEFAULT 0,
                vlm_description TEXT,
                inspected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER NOT NULL,
                defect_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x1 INTEGER NOT NULL,
                bbox_y1 INTEGER NOT NULL,
                bbox_x2 INTEGER NOT NULL,
                bbox_y2 INTEGER NOT NULL,
                vlm_description TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO inspections (
                image_name, original_image_path, status, defect_count, vlm_description
            )
            VALUES ('old.png', 'old.png', 'NG', 1, 'old explanation')
            """
        )

    manager.initialize()
    repository = InspectionRepository(manager)
    loaded = repository.find_by_id(1)

    assert loaded is not None
    assert loaded.vlm_status == VLM_STATUS_COMPLETED
    assert loaded.vlm_description == "old explanation"
    assert repository.save(InspectionResult(source_image_path=tmp_path / "new.png", status="OK"))
    new_loaded = repository.find_by_id(2)
    assert new_loaded is not None
    assert new_loaded.vlm_status == VLM_STATUS_NOT_REQUESTED
