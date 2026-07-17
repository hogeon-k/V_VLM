from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path

from config.settings import ERROR_LOG_PATH, INPUT_IMAGE_DIR, RESULT_IMAGE_DIR
from model.inspection_result import InspectionResult
from repository.inspection_repository import InspectionRepository

logger = logging.getLogger(__name__)


class InspectionHistoryError(Exception):
    """Base error for inspection history deletion."""


class InspectionHistoryNotFoundError(ValueError):
    """Raised when the requested inspection row does not exist."""


@dataclass(slots=True)
class FileCleanupIssue:
    path: Path
    reason: str


@dataclass(slots=True)
class DeletionReport:
    deleted_db_records: int = 0
    deleted_files: list[Path] = field(default_factory=list)
    missing_files: list[Path] = field(default_factory=list)
    skipped_external_originals: list[Path] = field(default_factory=list)
    blocked_files: list[FileCleanupIssue] = field(default_factory=list)
    failed_files: list[FileCleanupIssue] = field(default_factory=list)

    @property
    def has_file_warnings(self) -> bool:
        return bool(self.blocked_files or self.failed_files)


class InspectionHistoryService:
    def __init__(
        self,
        inspection_repository: InspectionRepository | None = None,
        *,
        result_image_dir: Path = RESULT_IMAGE_DIR,
        input_image_dir: Path = INPUT_IMAGE_DIR,
    ) -> None:
        self.inspection_repository = inspection_repository or InspectionRepository()
        self.result_image_dir = Path(result_image_dir)
        self.input_image_dir = Path(input_image_dir)
        _configure_history_logging()

    def count(self) -> int:
        return self.inspection_repository.count()

    def delete_history(self, inspection_id: int) -> DeletionReport:
        logger.info("Inspection history delete requested: inspection_id=%s", inspection_id)
        history = self.inspection_repository.find_by_id(inspection_id)
        if history is None:
            logger.warning("Inspection history delete failed: inspection_id=%s not found", inspection_id)
            raise InspectionHistoryNotFoundError("삭제할 검사 기록을 찾을 수 없습니다.")

        deleted_count = self.inspection_repository.delete_by_id(inspection_id)
        logger.info(
            "Inspection history DB rows deleted: inspection_id=%s deleted_db_records=%s",
            inspection_id,
            deleted_count,
        )
        if deleted_count <= 0:
            raise InspectionHistoryNotFoundError("삭제할 검사 기록을 찾을 수 없습니다.")

        report = DeletionReport(deleted_db_records=deleted_count)
        self._delete_files_for_history(history, report)
        return report

    def delete_all_history(self) -> DeletionReport:
        logger.info("All inspection history delete requested")
        histories = self.inspection_repository.search(limit=max(1, self.inspection_repository.count()))
        deleted_count = self.inspection_repository.delete_all()
        logger.info("All inspection history DB rows deleted: deleted_db_records=%s", deleted_count)

        report = DeletionReport(deleted_db_records=deleted_count)
        for history in histories:
            self._delete_files_for_history(history, report)
        return report

    def _delete_files_for_history(self, history: InspectionResult, report: DeletionReport) -> None:
        self._delete_managed_file(
            getattr(history, "result_image_path", None),
            allowed_root=self.result_image_dir,
            report=report,
            role="result_image",
            skip_outside_allowed=False,
        )
        self._delete_managed_file(
            getattr(history, "original_image_path", None),
            allowed_root=self.input_image_dir,
            report=report,
            role="original_image",
            skip_outside_allowed=True,
        )

    def _delete_managed_file(
        self,
        path: Path | None,
        *,
        allowed_root: Path,
        report: DeletionReport,
        role: str,
        skip_outside_allowed: bool,
    ) -> None:
        if path is None:
            return

        target_path = Path(path)
        logger.info("Inspection history file cleanup target: role=%s path=%s", role, target_path)
        if not _is_path_inside(target_path, allowed_root):
            if skip_outside_allowed:
                report.skipped_external_originals.append(target_path)
                logger.info("Skipped external original image cleanup: path=%s", target_path)
                return
            issue = FileCleanupIssue(target_path, f"허용된 저장 폴더 밖의 파일입니다: {allowed_root}")
            report.blocked_files.append(issue)
            logger.warning("Blocked file cleanup outside allowed root: role=%s path=%s", role, target_path)
            return

        try:
            resolved_path = target_path.resolve()
            if not resolved_path.exists():
                report.missing_files.append(target_path)
                logger.info("File cleanup skipped because file is already missing: path=%s", target_path)
                return
            if not resolved_path.is_file():
                issue = FileCleanupIssue(target_path, "삭제 대상이 파일이 아닙니다.")
                report.blocked_files.append(issue)
                logger.warning("Blocked non-file cleanup target: path=%s", target_path)
                return
            resolved_path.unlink()
            report.deleted_files.append(target_path)
            logger.info("File cleanup succeeded: path=%s", target_path)
        except Exception as exc:
            issue = FileCleanupIssue(target_path, str(exc))
            report.failed_files.append(issue)
            logger.exception("File cleanup failed: path=%s", target_path)


def _is_path_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _configure_history_logging() -> None:
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == ERROR_LOG_PATH
        for handler in logger.handlers
    ):
        return
    handler = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
