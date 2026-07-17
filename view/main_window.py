from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from view.history_view import HistoryView
from view.inspection_view import InspectionView
from view.statistics_view import StatisticsView
from view.status_view import StatusView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PCB Vision Inspection")
        self.setMinimumSize(1180, 760)

        self.stack = QStackedWidget()
        self.inspection_view = InspectionView()
        self.history_view = HistoryView()
        self.statistics_view = StatisticsView()
        self.status_view = StatusView()
        self.stack.addWidget(self.inspection_view)
        self.stack.addWidget(self.history_view)
        self.stack.addWidget(self.statistics_view)
        self.stack.addWidget(self.status_view)

        self.inspection_tab = _tab_button("메인 검사 화면")
        self.history_tab = _tab_button("검사 이력 화면")
        self.statistics_tab = _tab_button("통계 화면")
        self.status_tab = _tab_button("시스템 화면")
        self.inspection_tab.clicked.connect(lambda: self._show_page(0))
        self.history_tab.clicked.connect(lambda: self._show_page(1))
        self.statistics_tab.clicked.connect(lambda: self._show_page(2))
        self.status_tab.clicked.connect(lambda: self._show_page(3))

        title = QLabel("PCB 자동 검사")
        title.setObjectName("AppTitle")
        subtitle = QLabel("실시간 검사 대시보드")
        subtitle.setObjectName("AppSubtitle")

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        tabs = QHBoxLayout()
        tabs.setSpacing(8)
        tabs.addWidget(self.inspection_tab)
        tabs.addWidget(self.history_tab)
        tabs.addWidget(self.statistics_tab)
        tabs.addWidget(self.status_tab)

        header = QHBoxLayout()
        header.setContentsMargins(24, 14, 24, 8)
        header.addLayout(title_block, 1)
        header.addLayout(tabs)
        header.setAlignment(tabs, Qt.AlignRight | Qt.AlignVCenter)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        self._show_page(0)
        self.setStyleSheet(_main_stylesheet())

    def _show_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for tab_index, button in (
            (0, self.inspection_tab),
            (1, self.history_tab),
            (2, self.statistics_tab),
            (3, self.status_tab),
        ):
            button.setProperty("selected", tab_index == index)
            button.style().unpolish(button)
            button.style().polish(button)

    def closeEvent(self, event: object) -> None:
        self.status_view.stop_vlm_status_check()
        super().closeEvent(event)


def _tab_button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setCheckable(False)
    button.setCursor(Qt.PointingHandCursor)
    button.setMinimumHeight(38)
    return button


def _main_stylesheet() -> str:
    return """
    QMainWindow, QWidget {
        background: #f4f6f9;
        color: #17202a;
        font-family: "Malgun Gothic", "Segoe UI", sans-serif;
        font-size: 13px;
    }
    QLabel#AppTitle {
        font-size: 24px;
        font-weight: 800;
        color: #101820;
    }
    QLabel#AppSubtitle {
        color: #687381;
        font-size: 12px;
    }
    QPushButton {
        background: #ffffff;
        color: #263241;
        border: 1px solid #ccd4df;
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 600;
    }
    QPushButton:hover {
        background: #eef4ff;
        border-color: #8fb5ff;
    }
    QPushButton:disabled {
        background: #eef1f5;
        color: #9aa5b1;
        border-color: #d8dee7;
    }
    QPushButton[selected="true"] {
        background: #e8f1ff;
        color: #1250b5;
        border-color: #9dbcf5;
        border-bottom: 3px solid #2563eb;
    }
    """
