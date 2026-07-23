from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
import sys

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QDate
from PySide6.QtWidgets import QApplication, QMessageBox

from model.defect_info import Detection
from model.inspection_result import InspectionResult
from model.inspection_result import VLM_STATUS_COMPLETED, VLM_STATUS_FAILED, VLM_STATUS_NOT_REQUESTED
from repository.db_manager import DBManager
from repository.inspection_repository import InspectionRepository
from service.inspection_history_service import (
    DeletionReport,
    InspectionHistoryNotFoundError,
    InspectionHistoryService,
)
from view.history_view import HistoryView
from viewmodel.history_viewmodel import HistoryViewModel


DETAIL_PLACEHOLDER = "검사 이력을 선택하면 상세 결과가 표시됩니다."


def _repository(tmp_path: Path) -> InspectionRepository:
    return InspectionRepository(DBManager(tmp_path / "inspection.sqlite3"))


def _ng_result(source_path: Path, result_path: Path | None = None) -> InspectionResult:
    return InspectionResult(
        source_image_path=source_path,
        result_image_path=result_path,
        status="NG",
        detections=[Detection(0, "short", 0.95, 1, 2, 3, 4, vlm_description="vlm detail")],
        vlm_explanation="VLM explanation",
    )


def test_delete_selected_inspection_removes_row_child_and_managed_result_file(tmp_path) -> None:
    repository = _repository(tmp_path)
    result_dir = tmp_path / "data" / "result_images"
    input_dir = tmp_path / "data" / "input_images"
    result_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    result_image = result_dir / "result.png"
    result_image.write_bytes(b"result")
    external_source = tmp_path / "external_dataset" / "source.png"
    external_source.parent.mkdir()
    external_source.write_bytes(b"source")

    inspection_id = repository.save(_ng_result(external_source, result_image))
    report = InspectionHistoryService(
        repository,
        result_image_dir=result_dir,
        input_image_dir=input_dir,
    ).delete_history(inspection_id)

    assert repository.find_by_id(inspection_id) is None
    assert repository.count() == 0
    assert not result_image.exists()
    assert external_source.exists()
    assert report.deleted_db_records == 1
    assert result_image in report.deleted_files
    assert external_source in report.skipped_external_originals
    with repository.db_manager.get_connection() as connection:
        defect_count = connection.execute("SELECT COUNT(*) FROM defects").fetchone()[0]
    assert defect_count == 0


def test_delete_missing_inspection_id_raises_not_found(tmp_path) -> None:
    service = InspectionHistoryService(_repository(tmp_path))

    with pytest.raises(InspectionHistoryNotFoundError):
        service.delete_history(999)


def test_result_image_outside_allowed_folder_is_not_deleted(tmp_path) -> None:
    repository = _repository(tmp_path)
    result_dir = tmp_path / "managed_results"
    input_dir = tmp_path / "managed_inputs"
    result_dir.mkdir()
    input_dir.mkdir()
    outside_result = tmp_path / "outside" / "result.png"
    outside_result.parent.mkdir()
    outside_result.write_bytes(b"result")
    source = tmp_path / "dataset" / "source.png"
    source.parent.mkdir()
    source.write_bytes(b"source")

    inspection_id = repository.save(_ng_result(source, outside_result))
    report = InspectionHistoryService(
        repository,
        result_image_dir=result_dir,
        input_image_dir=input_dir,
    ).delete_history(inspection_id)

    assert repository.find_by_id(inspection_id) is None
    assert outside_result.exists()
    assert len(report.blocked_files) == 1
    assert report.blocked_files[0].path == outside_result


def test_missing_result_file_does_not_block_db_delete(tmp_path) -> None:
    repository = _repository(tmp_path)
    result_dir = tmp_path / "data" / "result_images"
    input_dir = tmp_path / "data" / "input_images"
    result_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    missing_result = result_dir / "missing.png"
    source = tmp_path / "dataset" / "source.png"
    source.parent.mkdir()
    source.write_bytes(b"source")

    inspection_id = repository.save(_ng_result(source, missing_result))
    report = InspectionHistoryService(
        repository,
        result_image_dir=result_dir,
        input_image_dir=input_dir,
    ).delete_history(inspection_id)

    assert repository.find_by_id(inspection_id) is None
    assert missing_result in report.missing_files


def test_repository_delete_rolls_back_when_db_delete_fails(tmp_path) -> None:
    repository = _repository(tmp_path)
    inspection_id = repository.save(_ng_result(tmp_path / "source.png", tmp_path / "result.png"))
    with repository.db_manager.get_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_inspection_delete
            BEFORE DELETE ON inspections
            BEGIN
                SELECT RAISE(ABORT, 'forced delete failure');
            END;
            """
        )

    with pytest.raises(sqlite3.DatabaseError):
        repository.delete_by_id(inspection_id)

    assert repository.find_by_id(inspection_id) is not None
    with repository.db_manager.get_connection() as connection:
        defect_count = connection.execute(
            "SELECT COUNT(*) FROM defects WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchone()[0]
    assert defect_count == 1


def test_delete_all_history_deletes_all_rows_and_managed_files(tmp_path) -> None:
    repository = _repository(tmp_path)
    result_dir = tmp_path / "data" / "result_images"
    input_dir = tmp_path / "data" / "input_images"
    result_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    first_result = result_dir / "first.png"
    second_result = result_dir / "second.png"
    first_result.write_bytes(b"first")
    second_result.write_bytes(b"second")
    repository.save(_ng_result(tmp_path / "dataset" / "first.png", first_result))
    repository.save(_ng_result(tmp_path / "dataset" / "second.png", second_result))

    report = InspectionHistoryService(
        repository,
        result_image_dir=result_dir,
        input_image_dir=input_dir,
    ).delete_all_history()

    assert repository.count() == 0
    assert report.deleted_db_records == 2
    assert not first_result.exists()
    assert not second_result.exists()


class FakeHistoryViewModel(HistoryViewModel):
    def __init__(self, results: list[InspectionResult]) -> None:
        self._results = results
        self.deleted_ids: list[int] = []
        self.delete_all_called = False
        self.search_calls: list[dict[str, object]] = []

    def search(self, **kwargs: object) -> list[InspectionResult]:
        self.search_calls.append(kwargs)
        return list(self._results)

    def load_detail(self, inspection_id: int) -> InspectionResult | None:
        return next((result for result in self._results if result.id == inspection_id), None)

    def defect_types(self) -> list[str]:
        return ["short"]

    def history_count(self) -> int:
        return len(self._results)

    def delete_history(self, inspection_id: int) -> DeletionReport:
        self.deleted_ids.append(inspection_id)
        self._results = [result for result in self._results if result.id != inspection_id]
        return DeletionReport(deleted_db_records=1)

    def delete_all_history(self) -> DeletionReport:
        self.delete_all_called = True
        deleted_count = len(self._results)
        self._results = []
        return DeletionReport(deleted_db_records=deleted_count)


class FakeFilteredHistoryViewModel(FakeHistoryViewModel):
    def __init__(self, all_count: int) -> None:
        super().__init__([])
        self._all_count = all_count

    def history_count(self) -> int:
        return self._all_count


def _app(monkeypatch: pytest.MonkeyPatch) -> QApplication:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication(sys.argv)


def test_delete_selected_without_selection_shows_message(monkeypatch) -> None:
    _app(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(text),
    )
    view = HistoryView(FakeHistoryViewModel([]))

    view._delete_selected()

    assert messages == ["삭제할 검사 기록을 선택해주세요."]


def test_selected_row_loads_detail_from_id(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    source = tmp_path / "source.png"
    result_image = tmp_path / "result.png"
    source.write_bytes(b"not a real image")
    result_image.write_bytes(b"not a real image")
    result = _ng_result(source, result_image)
    result.id = 10
    view = HistoryView(FakeHistoryViewModel([result]))

    view.table.selectRow(0)

    assert view.status_value.text() == "불량"
    assert view.defect_value.text() == "short"
    assert view.confidence_value.text() == "95.0%"
    assert view.detail.toPlainText() == "VLM explanation"


def test_history_detail_enables_vlm_generation_for_ng_without_description(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    result = InspectionResult(
        source_image_path=tmp_path / "source.png",
        result_image_path=tmp_path / "result.png",
        id=10,
        status="NG",
        detections=[Detection(0, "short", 0.95, 1, 2, 3, 4)],
        vlm_status=VLM_STATUS_NOT_REQUESTED,
    )
    view = HistoryView(FakeHistoryViewModel([result]))

    view.table.selectRow(0)

    assert view.generate_vlm_button.isEnabled()
    assert view.generate_vlm_button.text() == "VLM 설명 생성"
    assert "VLM 설명은 아직 생성되지 않았습니다." in view.detail.toPlainText()


def test_history_vlm_completion_does_not_replace_different_selected_detail(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    first = InspectionResult(
        source_image_path=tmp_path / "first.png",
        id=10,
        status="NG",
        detections=[Detection(0, "short", 0.95, 1, 2, 3, 4)],
    )
    second = InspectionResult(
        source_image_path=tmp_path / "second.png",
        id=20,
        status="NG",
        detections=[Detection(0, "open", 0.9, 5, 6, 7, 8)],
    )
    view = HistoryView(FakeHistoryViewModel([first, second]))
    view.table.selectRow(0)

    updated_second = InspectionResult(
        source_image_path=tmp_path / "second.png",
        id=20,
        status="NG",
        detections=[Detection(0, "open", 0.9, 5, 6, 7, 8)],
        vlm_explanation="second VLM",
        vlm_status=VLM_STATUS_COMPLETED,
    )
    view._on_vlm_finished(20, updated_second)

    assert "second VLM" not in view.detail.toPlainText()
    assert view._selected_inspection_id() == 10


def test_history_vlm_failure_refreshes_selected_detail(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    result = InspectionResult(
        source_image_path=tmp_path / "source.png",
        id=10,
        status="NG",
        detections=[Detection(0, "short", 0.95, 1, 2, 3, 4)],
        vlm_status=VLM_STATUS_FAILED,
        vlm_error_message="timeout",
    )
    messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, text: messages.append(text),
    )
    view = HistoryView(FakeHistoryViewModel([result]))
    view.table.selectRow(0)

    view._on_vlm_failed(10, "timeout")

    assert "timeout" in view.detail.toPlainText()
    assert messages == ["VLM 분석 실패: timeout"]


def test_history_table_shows_sequential_number_but_keeps_database_id(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    first = InspectionResult(
        source_image_path=tmp_path / "first.png",
        id=10,
        status="OK",
        inspected_at=datetime(2026, 7, 17, 9, 0, 0),
    )
    second = InspectionResult(
        source_image_path=tmp_path / "second.png",
        id=65,
        status="NG",
        inspected_at=datetime(2026, 7, 17, 10, 0, 0),
    )
    view = HistoryView(FakeHistoryViewModel([second, first]))

    assert view.table.horizontalHeaderItem(0).text() == "번호"
    assert view.table.item(0, 0).text() == "1"
    assert view.table.item(1, 0).text() == "2"

    view.table.selectRow(0)

    assert view._selected_inspection_id() == 10


def test_delete_selected_button_removes_selected_row_and_clears_detail(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    result = InspectionResult(source_image_path=tmp_path / "source.png", id=10, status="OK")
    viewmodel = FakeHistoryViewModel([result])
    info_messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, text: info_messages.append(text),
    )
    view = HistoryView(viewmodel)
    view.table.selectRow(0)

    view._delete_selected()

    assert viewmodel.deleted_ids == [10]
    assert view.table.rowCount() == 0
    assert view.detail.toPlainText() == DETAIL_PLACEHOLDER
    assert not view.delete_selected_button.isEnabled()
    assert not view.delete_all_button.isEnabled()
    assert info_messages == ["선택한 검사 기록이 삭제되었습니다."]


def test_delete_all_cancel_does_not_delete(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    result = InspectionResult(source_image_path=tmp_path / "source.png", id=10, status="OK")
    viewmodel = FakeHistoryViewModel([result])
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )
    view = HistoryView(viewmodel)

    view._delete_all()

    assert not viewmodel.delete_all_called
    assert view.table.rowCount() == 1


def test_delete_all_button_uses_database_count_not_filtered_rows(monkeypatch) -> None:
    _app(monkeypatch)
    viewmodel = FakeFilteredHistoryViewModel(all_count=1)
    view = HistoryView(viewmodel)

    assert view.table.rowCount() == 0
    assert view.delete_all_button.isEnabled()


def test_delete_all_success_clears_table_detail_and_buttons(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    result = InspectionResult(source_image_path=tmp_path / "source.png", id=10, status="OK")
    viewmodel = FakeHistoryViewModel([result])
    info_messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        "view.history_view.QInputDialog.getText",
        lambda *_args, **_kwargs: ("전체 삭제", True),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, text: info_messages.append(text),
    )
    view = HistoryView(viewmodel)
    view.table.selectRow(0)

    view._delete_all()

    assert viewmodel.delete_all_called
    assert view.table.rowCount() == 0
    assert view.detail.toPlainText() == DETAIL_PLACEHOLDER
    assert not view.delete_selected_button.isEnabled()
    assert not view.delete_all_button.isEnabled()
    assert info_messages == ["전체 검사 기록이 삭제되었습니다."]


def test_filter_search_passes_combined_conditions(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    result = InspectionResult(source_image_path=tmp_path / "source.png", id=10, status="OK")
    viewmodel = FakeHistoryViewModel([result])
    view = HistoryView(viewmodel)
    view.all_dates_checkbox.setChecked(False)
    view.start_date.setDate(QDate.fromString("2026-07-01", "yyyy-MM-dd"))
    view.end_date.setDate(QDate.fromString("2026-07-17", "yyyy-MM-dd"))
    view.status_filter.setCurrentText("불량")
    view.defect_filter.setCurrentText("short")

    view.reload()

    assert viewmodel.search_calls[-1] == {
        "start_date": "2026-07-01",
        "end_date": "2026-07-17",
        "status": "NG",
        "defect_type": "short",
    }
