from __future__ import annotations

from model.statistics_result import StatisticsDashboardData, StatisticsResult
from service.statistics_service import StatisticsService


class StatisticsViewModel:
    def __init__(self, statistics_service: StatisticsService | None = None) -> None:
        self.statistics_service = statistics_service or StatisticsService()

    def load_summary(self) -> StatisticsResult:
        return self.statistics_service.build_summary()

    def load_dashboard(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> StatisticsDashboardData:
        return self.statistics_service.build_dashboard(
            start_date=start_date,
            end_date=end_date,
        )
