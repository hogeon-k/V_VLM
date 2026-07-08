from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StatisticsResult:
    total_count: int = 0
    ok_count: int = 0
    ng_count: int = 0
    defect_type_counts: dict[str, int] = field(default_factory=dict)
