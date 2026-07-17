from __future__ import annotations

from PySide6.QtCore import QDate, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDateEdit,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from model.statistics_result import DefectTypeCount, NgTrendPoint, StatisticsDashboardData
from viewmodel.statistics_viewmodel import StatisticsViewModel


class StatisticsView(QWidget):
    go_main_requested = Signal()

    def __init__(self, viewmodel: StatisticsViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or StatisticsViewModel()
        self._is_loading = False

        self.all_dates_checkbox = QCheckBox("전체 기간")
        self.all_dates_checkbox.setChecked(True)
        self.all_dates_checkbox.toggled.connect(self._sync_date_filter_state)
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        self.start_date.setDate(QDate.currentDate().addDays(-30))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        self.end_date.setDate(QDate.currentDate())

        self.search_button = QPushButton("조회")
        self.search_button.clicked.connect(self.reload)
        self.reset_button = QPushButton("초기화")
        self.reset_button.clicked.connect(self.reset_filters)
        self.main_button = QPushButton("메인 검사 화면")
        self.main_button.clicked.connect(self.go_main_requested.emit)

        self.total_card = StatCard("전체 검사 수")
        self.ok_card = StatCard("OK 수", accent="ok")
        self.ng_card = StatCard("NG 수", accent="ng")
        self.ng_rate_card = StatCard("NG 비율", accent="ng")

        self.defect_chart = BarChartWidget("불량 유형별 발생 건수")
        self.trend_chart = LineChartWidget("기간별 NG 추이")
        self.top_defect_card = TopDefectCard()
        self.state_label = QLabel("")
        self.state_label.setObjectName("MutedText")

        self._build_layout()
        self.setStyleSheet(_statistics_stylesheet())
        self._sync_date_filter_state()
        self.reload()

    def _build_layout(self) -> None:
        filters = QHBoxLayout()
        filters.setSpacing(8)
        filters.addWidget(self.all_dates_checkbox)
        filters.addWidget(QLabel("시작일"))
        filters.addWidget(self.start_date)
        filters.addWidget(QLabel("종료일"))
        filters.addWidget(self.end_date)
        filters.addWidget(self.search_button)
        filters.addWidget(self.reset_button)
        filters.addStretch(1)
        filters.addWidget(self.main_button)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)
        for col, card in enumerate((self.total_card, self.ok_card, self.ng_card, self.ng_rate_card)):
            cards.addWidget(card, 0, col)
            cards.setColumnStretch(col, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.defect_chart)
        splitter.addWidget(self.trend_chart)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 560])

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 8, 24, 16)
        layout.setSpacing(12)
        layout.addLayout(filters)
        layout.addWidget(self.state_label)
        layout.addLayout(cards)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.top_defect_card)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    def reload(self) -> None:
        if self._is_loading:
            return
        if not self.all_dates_checkbox.isChecked() and self.start_date.date() > self.end_date.date():
            QMessageBox.warning(self, "통계 조회", "시작일은 종료일보다 늦을 수 없습니다.")
            return

        self._is_loading = True
        self.search_button.setEnabled(False)
        self.state_label.setText("조회 중...")
        try:
            data = self.viewmodel.load_dashboard(
                start_date=None if self.all_dates_checkbox.isChecked() else self.start_date.date().toString("yyyy-MM-dd"),
                end_date=None if self.all_dates_checkbox.isChecked() else self.end_date.date().toString("yyyy-MM-dd"),
            )
        except Exception as exc:
            self._clear_dashboard()
            QMessageBox.critical(self, "통계 조회", f"통계 데이터를 불러오는 중 오류가 발생했습니다.\n{exc}")
            self.state_label.setText("통계 조회 오류")
            return
        finally:
            self._is_loading = False
            self.search_button.setEnabled(True)

        self._render_dashboard(data)

    def reset_filters(self) -> None:
        self.all_dates_checkbox.setChecked(True)
        self.start_date.setDate(QDate.currentDate().addDays(-30))
        self.end_date.setDate(QDate.currentDate())
        self.reload()

    def _render_dashboard(self, data: StatisticsDashboardData) -> None:
        summary = data.summary
        self.total_card.set_value(str(summary.total_count))
        self.ok_card.set_value(str(summary.ok_count))
        self.ng_card.set_value(str(summary.ng_count))
        self.ng_rate_card.set_value(f"{summary.ng_rate:.1f}%")
        self.defect_chart.set_items(data.defect_counts)
        self.trend_chart.set_points(data.ng_trend)
        self.top_defect_card.set_items(data.top_defect_types, summary.ng_count)
        self.state_label.setText("선택한 기간에 검사 데이터가 없습니다." if summary.total_count == 0 else "조회 완료")

    def _clear_dashboard(self) -> None:
        for card in (self.total_card, self.ok_card, self.ng_card, self.ng_rate_card):
            card.set_value("0")
        self.ng_rate_card.set_value("0.0%")
        self.defect_chart.set_items([])
        self.trend_chart.set_points([])
        self.top_defect_card.set_items([], 0)

    def _sync_date_filter_state(self) -> None:
        enabled = not self.all_dates_checkbox.isChecked()
        self.start_date.setEnabled(enabled)
        self.end_date.setEnabled(enabled)


class StatCard(QFrame):
    def __init__(self, title: str, *, accent: str = "default") -> None:
        super().__init__()
        self.setObjectName("StatCard")
        self.setProperty("accent", accent)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")
        self.value_label = QLabel("0")
        self.value_label.setObjectName("CardValue")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class TopDefectCard(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("TopDefectCard")
        self.title_label = QLabel("최다 불량 유형")
        self.title_label.setObjectName("CardTitle")
        self.name_label = QLabel("없음")
        self.name_label.setObjectName("TopDefectName")
        self.detail_label = QLabel("발생 건수: 0건 | 전체 NG 대비 0.0%")
        self.detail_label.setObjectName("MutedText")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        layout.addWidget(self.title_label)
        layout.addWidget(self.name_label)
        layout.addWidget(self.detail_label)

    def set_items(self, items: list[DefectTypeCount], total_ng: int) -> None:
        if not items or total_ng <= 0:
            self.name_label.setText("없음")
            self.detail_label.setText("발생 건수: 0건 | 전체 NG 대비 0.0%")
            return
        names = ", ".join(item.defect_type for item in items)
        count = items[0].count
        rate = count / total_ng * 100.0 if total_ng else 0.0
        prefix = "공동 1위: " if len(items) > 1 else ""
        self.name_label.setText(f"{prefix}{names}")
        self.detail_label.setText(f"발생 건수: {count}건 | 전체 NG 대비 {rate:.1f}%")


class BarChartWidget(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("ChartPanel")
        self.title = title
        self.items: list[DefectTypeCount] = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_items(self, items: list[DefectTypeCount]) -> None:
        self.items = items[:10]
        self.update()

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _draw_panel_title(painter, self.rect(), self.title)
        chart_rect = self.rect().adjusted(18, 46, -18, -18)
        if not self.items:
            _draw_empty_text(painter, chart_rect, "선택한 기간에 불량 데이터가 없습니다.")
            return
        max_count = max(item.count for item in self.items) or 1
        row_height = max(24, chart_rect.height() // max(1, len(self.items)))
        label_width = min(160, max(90, chart_rect.width() // 3))
        for index, item in enumerate(self.items):
            y = chart_rect.top() + index * row_height
            painter.setPen(QColor("#263241"))
            painter.drawText(QRectF(chart_rect.left(), y, label_width - 8, row_height), Qt.AlignmentFlag.AlignVCenter, item.defect_type)
            bar_left = chart_rect.left() + label_width
            bar_width = int((chart_rect.width() - label_width - 44) * item.count / max_count)
            bar_rect = QRectF(bar_left, y + 5, max(2, bar_width), row_height - 10)
            painter.fillRect(bar_rect, QColor("#dc2626"))
            painter.setPen(QColor("#17202a"))
            painter.drawText(QRectF(bar_rect.right() + 6, y, 40, row_height), Qt.AlignmentFlag.AlignVCenter, str(item.count))


class LineChartWidget(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("ChartPanel")
        self.title = title
        self.points: list[NgTrendPoint] = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_points(self, points: list[NgTrendPoint]) -> None:
        self.points = points
        self.update()

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _draw_panel_title(painter, self.rect(), self.title)
        chart_rect = self.rect().adjusted(42, 48, -20, -42)
        nonzero = [point for point in self.points if point.count > 0]
        if not nonzero:
            _draw_empty_text(painter, chart_rect, "선택한 기간에 NG 데이터가 없습니다.")
            return
        max_count = max(point.count for point in self.points) or 1
        painter.setPen(QPen(QColor("#cfd7e3"), 1))
        painter.drawLine(chart_rect.bottomLeft(), chart_rect.bottomRight())
        painter.drawLine(chart_rect.bottomLeft(), chart_rect.topLeft())
        if len(self.points) == 1:
            x_step = 0
        else:
            x_step = chart_rect.width() / (len(self.points) - 1)
        coordinates: list[QPointF] = []
        for index, point in enumerate(self.points):
            x = chart_rect.left() + index * x_step
            y = chart_rect.bottom() - (chart_rect.height() * point.count / max_count)
            coordinates.append(QPointF(x, y))
        painter.setPen(QPen(QColor("#2563eb"), 2))
        for start, end in zip(coordinates, coordinates[1:], strict=False):
            painter.drawLine(start, end)
        painter.setBrush(QColor("#2563eb"))
        for coordinate, point in zip(coordinates, self.points, strict=True):
            painter.drawEllipse(coordinate, 3, 3)
            if point.count:
                painter.drawText(QRectF(coordinate.x() - 12, coordinate.y() - 22, 24, 16), Qt.AlignmentFlag.AlignCenter, str(point.count))
        painter.setPen(QColor("#667085"))
        if self.points:
            painter.drawText(QRectF(chart_rect.left() - 20, chart_rect.bottom() + 8, 80, 20), Qt.AlignmentFlag.AlignLeft, self.points[0].period_label)
            painter.drawText(QRectF(chart_rect.right() - 80, chart_rect.bottom() + 8, 100, 20), Qt.AlignmentFlag.AlignRight, self.points[-1].period_label)


def _draw_panel_title(painter: QPainter, rect: QRectF, title: str) -> None:
    painter.setPen(QColor("#17202a"))
    font = QFont()
    font.setBold(True)
    font.setPointSize(10)
    painter.setFont(font)
    painter.drawText(rect.adjusted(14, 10, -14, -10), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, title)


def _draw_empty_text(painter: QPainter, rect: QRectF, text: str) -> None:
    painter.setPen(QColor("#7a8594"))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


def _statistics_stylesheet() -> str:
    return """
    QLabel#MutedText {
        color: #667085;
    }
    QFrame#StatCard, QFrame#TopDefectCard, QFrame#ChartPanel {
        background: #ffffff;
        border: 1px solid #d6dde8;
        border-radius: 6px;
    }
    QLabel#CardTitle {
        color: #667085;
        font-size: 12px;
        font-weight: 700;
    }
    QLabel#CardValue {
        color: #17202a;
        font-size: 30px;
        font-weight: 900;
    }
    QFrame#StatCard[accent="ok"] QLabel#CardValue {
        color: #147a42;
    }
    QFrame#StatCard[accent="ng"] QLabel#CardValue {
        color: #b4232d;
    }
    QLabel#TopDefectName {
        color: #17202a;
        font-size: 22px;
        font-weight: 900;
    }
    """
