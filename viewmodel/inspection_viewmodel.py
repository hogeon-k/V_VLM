from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from model.inspection_result import InspectionResult
from service.auto_inspection_service import AutoInspectionService
from service.inspection_service import InspectionService


class InspectionWorker(QObject):
    started = Signal(int)
    image_started = Signal(str, int, int)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, image_paths: list[Path], inspection_service: InspectionService) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.inspection_service = inspection_service
        self._pause_requested = False
        self._stop_requested = False

    def run(self) -> None:
        self.started.emit(len(self.image_paths))
        try:
            for index, image_path in enumerate(self.image_paths, start=1):
                if self._stop_requested:
                    break
                while self._pause_requested and not self._stop_requested:
                    QThread.msleep(100)
                if self._stop_requested:
                    break

                self.image_started.emit(str(image_path), index, len(self.image_paths))
                result = self.inspection_service.inspect_image(image_path)
                self.result_ready.emit(result)

                for _ in range(20):
                    if self._stop_requested or self._pause_requested:
                        break
                    QThread.msleep(100)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def pause(self) -> None:
        self._pause_requested = True

    def resume(self) -> None:
        self._pause_requested = False

    def stop(self) -> None:
        self._stop_requested = True
        self._pause_requested = False


class InspectionViewModel(QObject):
    started = Signal(int)
    image_started = Signal(str, int, int)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        inspection_service: InspectionService | None = None,
        auto_inspection_service: AutoInspectionService | None = None,
    ) -> None:
        super().__init__()
        self.inspection_service = inspection_service or InspectionService()
        self.auto_inspection_service = auto_inspection_service or AutoInspectionService(self.inspection_service)
        self.current_result: InspectionResult | None = None
        self.selected_folder: Path | None = None
        self.image_paths: list[Path] = []
        self._thread: QThread | None = None
        self._worker: InspectionWorker | None = None

    def inspect_image(self, image_path: Path) -> InspectionResult:
        self.current_result = self.inspection_service.inspect_image(image_path)
        return self.current_result

    def select_folder(self, folder: str | Path) -> list[Path]:
        self.selected_folder = Path(folder)
        self.image_paths = self.auto_inspection_service.list_images(self.selected_folder)
        if not self.image_paths:
            raise ValueError("선택한 폴더에 이미지 파일이 없습니다.")
        return self.image_paths

    def start_auto_inspection(self) -> None:
        if not self.image_paths:
            raise ValueError("이미지 폴더를 먼저 선택하세요.")
        if self._thread is not None:
            raise RuntimeError("자동 검사가 이미 실행 중입니다.")

        self._thread = QThread()
        self._worker = InspectionWorker(self.image_paths, self.inspection_service)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.started.connect(self.started)
        self._worker.image_started.connect(self.image_started)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.error.connect(self.error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self.finished)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def pause(self) -> None:
        if self._worker is not None:
            self._worker.pause()

    def resume(self) -> None:
        if self._worker is not None:
            self._worker.resume()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def is_running(self) -> bool:
        return self._thread is not None

    def _on_result_ready(self, result: InspectionResult) -> None:
        self.current_result = result
        self.result_ready.emit(result)

    def _cleanup_thread(self) -> None:
        self._thread = None
        self._worker = None
