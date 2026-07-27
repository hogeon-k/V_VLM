from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication

from model.inspection_result import InspectionResult
from view import main_window as main_window_module
from view.inspection_view import InspectionView
from view.main_window import MainWindow
from viewmodel.inspection_viewmodel import InspectionViewModel, InspectionWorker


def application() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


class FakeInspectionViewModel(QObject):
    started = Signal(int)
    image_started = Signal(str, int, int)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.image_paths = [Path("one.jpg")]
        self.running = False
        self.stop_calls = 0

    def backend_display_text(self) -> str:
        return "PyTorch"

    def is_running(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.stop_calls += 1

    def pause(self) -> None:
        return None

    def resume(self) -> None:
        return None


def test_worker_stop_during_inference_suppresses_stale_result() -> None:
    worker_holder: dict[str, InspectionWorker] = {}

    class StoppingService:
        def inspect_image(self, image_path: Path) -> InspectionResult:
            worker_holder["worker"].stop()
            return InspectionResult(source_image_path=image_path, status="OK")

    worker = InspectionWorker([Path("one.jpg")], StoppingService())
    worker_holder["worker"] = worker
    results: list[object] = []
    finished: list[bool] = []
    worker.result_ready.connect(results.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert results == []
    assert finished == [True]


def test_worker_runs_service_off_main_thread_and_delivers_result_on_main_thread() -> None:
    app = application()

    class RecordingService:
        worker_thread: QThread | None = None

        def inspect_image(self, image_path: Path) -> InspectionResult:
            self.worker_thread = QThread.currentThread()
            return InspectionResult(source_image_path=image_path, status="OK")

    class Receiver(QObject):
        def __init__(self) -> None:
            super().__init__()
            self.result_thread: QThread | None = None

        @Slot(object)
        def receive(self, result: object) -> None:
            self.result_thread = QThread.currentThread()

    service = RecordingService()
    receiver = Receiver()
    worker = InspectionWorker([Path("one.jpg")], service)
    thread = QThread()
    loop = QEventLoop()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.result_ready.connect(receiver.receive)
    worker.finished.connect(thread.quit)
    thread.finished.connect(loop.quit)
    thread.start()
    QTimer.singleShot(3000, loop.quit)

    loop.exec()
    thread.wait(3000)
    app.processEvents()

    assert thread.isRunning() is False
    assert service.worker_thread is not None
    assert service.worker_thread is not app.thread()
    assert receiver.result_thread is app.thread()


def test_worker_error_is_not_overwritten_as_completed(monkeypatch) -> None:
    application()
    monkeypatch.setattr(
        "view.inspection_view.QMessageBox.warning",
        lambda *args, **kwargs: None,
    )
    viewmodel = FakeInspectionViewModel()
    view = InspectionView(viewmodel)

    view._on_started(3)
    view._current_index = 1
    view.progress_bar.setValue(1)
    view._on_worker_error("inference failed")
    view._on_finished()

    assert view.state_label.text() == "현재 상태: 오류"
    assert view.progress_bar.value() == 1
    assert view.start_button.isEnabled() is True
    view.close()


def test_stop_keeps_controls_disabled_until_worker_finishes() -> None:
    application()
    viewmodel = FakeInspectionViewModel()
    viewmodel.running = True
    view = InspectionView(viewmodel)

    view._stop()

    assert viewmodel.stop_calls == 1
    assert view.state_label.text() == "현재 상태: 정지 중"
    assert view.start_button.isEnabled() is False
    assert view.choose_button.isEnabled() is False

    viewmodel.running = False
    view._on_finished()

    assert view.state_label.text() == "현재 상태: 대기"
    assert view.start_button.isEnabled() is True
    view.close()


def test_viewmodel_shutdown_does_not_close_service_while_worker_runs() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class FakeWorker:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    service = FakeService()
    worker = FakeWorker()
    viewmodel = InspectionViewModel(
        inspection_service=service,
        auto_inspection_service=SimpleNamespace(),
    )
    viewmodel._thread = SimpleNamespace()
    viewmodel._worker = worker

    assert viewmodel.shutdown() is False
    assert worker.stop_calls == 1
    assert service.close_calls == 0

    viewmodel._thread = None
    viewmodel._worker = None
    assert viewmodel.shutdown() is True
    assert service.close_calls == 1


def test_main_window_close_is_ignored_while_inspection_runs(monkeypatch) -> None:
    stop_calls: list[bool] = []
    ignored: list[bool] = []
    fake_window = SimpleNamespace(
        inspection_view=SimpleNamespace(
            viewmodel=SimpleNamespace(
                is_running=lambda: True,
                stop=lambda: stop_calls.append(True),
            )
        ),
        history_view=SimpleNamespace(is_vlm_running=lambda: False),
    )
    event = SimpleNamespace(ignore=lambda: ignored.append(True))
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda *args, **kwargs: None,
    )

    MainWindow.closeEvent(fake_window, event)

    assert stop_calls == [True]
    assert ignored == [True]
