from pathlib import Path

from model.defect_info import Detection
from model.inspection_result import (
    InspectionResult,
    VLM_STATUS_COMPLETED,
    VLM_STATUS_FAILED,
    VLM_STATUS_NOT_REQUESTED,
    VLM_STATUS_PROCESSING,
)
from model.yolo_result import YoloResult
from repository.db_manager import DBManager
from repository.inspection_repository import InspectionRepository
from service.inspection_service import InspectionService
from service.auto_inspection_service import AutoInspectionService


class PassthroughImageService:
    def prepare_image(self, image_path: Path) -> Path:
        return image_path


class FakeInspectionRepository:
    def save(self, inspection_result: object) -> int:
        return 1


def test_inspection_service_returns_ok_without_vlm() -> None:
    class FakeYoloService:
        def detect(self, image_path: Path) -> YoloResult:
            return YoloResult(image_path=image_path, annotated_image_path=Path("result.jpg"))

    class FailingVlmService:
        def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str:
            raise AssertionError("VLM should not be called for OK images.")

    result = InspectionService(
        image_service=PassthroughImageService(),
        yolo_service=FakeYoloService(),
        vlm_service=FailingVlmService(),
        inspection_repository=FakeInspectionRepository(),
    ).inspect_image(Path("sample.png"))

    assert result.image_name == "sample.png"
    assert result.defect_count == 0
    assert result.status == "OK"
    assert result.vlm_explanation is None


def test_inspection_service_returns_ng_with_vlm_description() -> None:
    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)

    class FakeYoloService:
        def detect(self, image_path: Path) -> YoloResult:
            return YoloResult(
                image_path=image_path,
                detections=[detection],
                annotated_image_path=Path("result.jpg"),
            )

    class FakeVlmService:
        def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str:
            assert image_path == Path("result.jpg")
            return "VLM explanation"

    result = InspectionService(
        image_service=PassthroughImageService(),
        yolo_service=FakeYoloService(),
        vlm_service=FakeVlmService(),
        inspection_repository=FakeInspectionRepository(),
    ).inspect(Path("sample.png"))

    assert result.status == "NG"
    assert result.defect_count == 1
    assert result.vlm_explanation == "VLM explanation"


def test_inspect_yolo_only_returns_ng_without_calling_vlm(tmp_path) -> None:
    source = tmp_path / "sample.png"
    result_image = tmp_path / "sample_yolo.png"
    source.write_bytes(b"source")
    result_image.write_bytes(b"result")
    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)

    class FakeYoloService:
        def detect(self, image_path: Path) -> YoloResult:
            return YoloResult(
                image_path=image_path,
                detections=[detection],
                annotated_image_path=result_image,
            )

    class FailingVlmService:
        def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str:
            raise AssertionError("VLM should not run during YOLO-only inspection.")

    repository = InspectionRepository(DBManager(tmp_path / "inspection.sqlite3"))
    result = InspectionService(
        image_service=PassthroughImageService(),
        yolo_service=FakeYoloService(),
        vlm_service=FailingVlmService(),
        inspection_repository=repository,
    ).inspect_yolo_only(source)

    loaded = repository.find_by_id(result.id or -1)
    assert result.status == "NG"
    assert result.vlm_description is None
    assert result.vlm_status == VLM_STATUS_NOT_REQUESTED
    assert loaded is not None
    assert loaded.defect_count == 1
    assert loaded.vlm_status == VLM_STATUS_NOT_REQUESTED


def test_run_vlm_for_saved_ng_inspection_updates_existing_row(tmp_path) -> None:
    source = tmp_path / "sample.png"
    result_image = tmp_path / "sample_yolo.png"
    source.write_bytes(b"source")
    result_image.write_bytes(b"result")
    repository = InspectionRepository(DBManager(tmp_path / "inspection.sqlite3"))
    inspection_id = repository.save(
        InspectionResult(
            source_image_path=source,
            result_image_path=result_image,
            status="NG",
            detections=[Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)],
        )
    )

    class FakeVlmService:
        def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str:
            assert image_path == result_image
            assert yolo_result.defect_count == 1
            return "generated VLM explanation"

    service = InspectionService(
        image_service=PassthroughImageService(),
        vlm_service=FakeVlmService(),
        inspection_repository=repository,
    )

    result = service.run_vlm_for_inspection(inspection_id)

    assert result.id == inspection_id
    assert result.vlm_status == VLM_STATUS_COMPLETED
    assert result.vlm_description == "generated VLM explanation"
    assert repository.count() == 1


def test_run_vlm_failure_marks_failed_and_preserves_yolo_result(tmp_path) -> None:
    source = tmp_path / "sample.png"
    result_image = tmp_path / "sample_yolo.png"
    source.write_bytes(b"source")
    result_image.write_bytes(b"result")
    repository = InspectionRepository(DBManager(tmp_path / "inspection.sqlite3"))
    inspection_id = repository.save(
        InspectionResult(
            source_image_path=source,
            result_image_path=result_image,
            status="NG",
            detections=[Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)],
        )
    )

    class FailingVlmService:
        def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str:
            raise RuntimeError("vlm failed")

    service = InspectionService(
        image_service=PassthroughImageService(),
        vlm_service=FailingVlmService(),
        inspection_repository=repository,
    )

    try:
        service.run_vlm_for_inspection(inspection_id)
    except RuntimeError:
        pass

    loaded = repository.find_by_id(inspection_id)
    assert loaded is not None
    assert loaded.status == "NG"
    assert loaded.result_image_path == result_image
    assert loaded.defect_count == 1
    assert loaded.vlm_status == VLM_STATUS_FAILED
    assert loaded.vlm_error_message == "vlm failed"


def test_run_vlm_rejects_ok_and_processing_inspections(tmp_path) -> None:
    source = tmp_path / "sample.png"
    source.write_bytes(b"source")
    repository = InspectionRepository(DBManager(tmp_path / "inspection.sqlite3"))
    ok_id = repository.save(InspectionResult(source_image_path=source, status="OK"))
    ng_id = repository.save(
        InspectionResult(
            source_image_path=source,
            result_image_path=source,
            status="NG",
            detections=[Detection(0, "short", 0.8, 1, 2, 3, 4)],
            vlm_status=VLM_STATUS_PROCESSING,
        )
    )
    service = InspectionService(
        image_service=PassthroughImageService(),
        inspection_repository=repository,
    )

    try:
        service.run_vlm_for_inspection(ok_id)
    except ValueError as exc:
        assert "NG" in str(exc)
    else:
        raise AssertionError("OK inspections should not run VLM.")

    try:
        service.run_vlm_for_inspection(ng_id)
    except RuntimeError as exc:
        assert "processing" in str(exc)
    else:
        raise AssertionError("PROCESSING inspections should not run duplicate VLM.")


def test_auto_inspection_service_lists_nested_images(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    image = nested / "sample.JFIF"
    image.write_bytes(b"fake image bytes")
    (tmp_path / "note.txt").write_text("ignore", encoding="utf-8")

    images = AutoInspectionService().list_images(tmp_path)

    assert images == [image]
