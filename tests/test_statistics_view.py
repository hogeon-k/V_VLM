from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QDate
from PySide6.QtWidgets import QApplication, QMessageBox

from model.statistics_result import (
    DefectTypeCount,
    NgTrendPoint,
    StatisticsDashboardData,
    StatisticsResult,
)
from view.statistics_view import StatisticsView


class FakeStatisticsViewModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.data = StatisticsDashboardData(
            summary=StatisticsResult(total_count=10, ok_count=7, ng_count=3),
            defect_counts=[DefectTypeCount("short", 2), DefectTypeCount("open_circuit", 1)],
            ng_trend=[NgTrendPoint("2026-07-01", 1), NgTrendPoint("2026-07-02", 2)],
            top_defect_types=[DefectTypeCount("short", 2)],
        )

    def load_dashboard(self, **kwargs: object) -> StatisticsDashboardData:
        self.calls.append(kwargs)
        return self.data


def _app(monkeypatch: pytest.MonkeyPatch) -> QApplication:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication(sys.argv)


def test_statistics_view_renders_cards_from_dashboard(monkeypatch) -> None:
    _app(monkeypatch)
    viewmodel = FakeStatisticsViewModel()
    view = StatisticsView(viewmodel)  # type: ignore[arg-type]

    assert view.total_card.value_label.text() == "10"
    assert view.ok_card.value_label.text() == "7"
    assert view.ng_card.value_label.text() == "3"
    assert view.ng_rate_card.value_label.text() == "30.0%"
    assert view.defect_chart.items[0].defect_type == "short"
    assert view.trend_chart.points[-1].count == 2
    assert view.top_defect_card.name_label.text() == "short"


def test_statistics_view_passes_date_filter(monkeypatch) -> None:
    _app(monkeypatch)
    viewmodel = FakeStatisticsViewModel()
    view = StatisticsView(viewmodel)  # type: ignore[arg-type]
    view.all_dates_checkbox.setChecked(False)
    view.start_date.setDate(QDate.fromString("2026-07-01", "yyyy-MM-dd"))
    view.end_date.setDate(QDate.fromString("2026-07-17", "yyyy-MM-dd"))

    view.reload()

    assert viewmodel.calls[-1] == {
        "start_date": "2026-07-01",
        "end_date": "2026-07-17",
    }


def test_statistics_view_rejects_invalid_date_range(monkeypatch) -> None:
    _app(monkeypatch)
    viewmodel = FakeStatisticsViewModel()
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    view = StatisticsView(viewmodel)  # type: ignore[arg-type]
    call_count = len(viewmodel.calls)
    view.all_dates_checkbox.setChecked(False)
    view.start_date.setDate(QDate.fromString("2026-07-17", "yyyy-MM-dd"))
    view.end_date.setDate(QDate.fromString("2026-07-01", "yyyy-MM-dd"))

    view.reload()

    assert len(viewmodel.calls) == call_count
    assert warnings == ["시작일은 종료일보다 늦을 수 없습니다."]


def test_statistics_view_reset_uses_all_dates(monkeypatch) -> None:
    _app(monkeypatch)
    viewmodel = FakeStatisticsViewModel()
    view = StatisticsView(viewmodel)  # type: ignore[arg-type]
    view.all_dates_checkbox.setChecked(False)

    view.reset_filters()

    assert view.all_dates_checkbox.isChecked()
    assert viewmodel.calls[-1] == {"start_date": None, "end_date": None}
