from __future__ import annotations

from model.statistics_result import StatisticsResult
from service.statistics_service import StatisticsService


class StatisticsViewModel:
    def __init__(self, statistics_service: StatisticsService | None = None) -> None:
        self.statistics_service = statistics_service or StatisticsService()

    def load_summary(self) -> StatisticsResult:
        return self.statistics_service.build_summary()
