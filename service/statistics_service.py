from __future__ import annotations

from model.statistics_result import StatisticsResult


class StatisticsService:
    def build_summary(self) -> StatisticsResult:
        # TODO: Query repositories and aggregate inspection statistics.
        return StatisticsResult()
