from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DefectTypeCount:
    defect_type: str
    count: int


@dataclass(frozen=True, slots=True)
class NgTrendPoint:
    period_label: str
    count: int


@dataclass(slots=True)
class StatisticsResult:
    total_count: int = 0
    ok_count: int = 0
    ng_count: int = 0
    defect_type_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ng_rate(self) -> float:
        return (self.ng_count / self.total_count * 100.0) if self.total_count else 0.0


@dataclass(frozen=True, slots=True)
class StatisticsDashboardData:
    summary: StatisticsResult
    defect_counts: list[DefectTypeCount] = field(default_factory=list)
    ng_trend: list[NgTrendPoint] = field(default_factory=list)
    top_defect_types: list[DefectTypeCount] = field(default_factory=list)
