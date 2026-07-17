from __future__ import annotations

from datetime import date, datetime, timedelta

from model.statistics_result import (
    DefectTypeCount,
    NgTrendPoint,
    StatisticsDashboardData,
    StatisticsResult,
)
from repository.db_manager import DBManager

OK_STATUSES = ("OK", "정상", "PASS", "pass")
NG_STATUSES = ("NG", "불량", "FAIL", "fail")


class StatisticsService:
    def __init__(self, db_manager: DBManager | None = None) -> None:
        self.db_manager = db_manager or DBManager()

    def build_summary(self) -> StatisticsResult:
        return self.build_dashboard().summary

    def build_dashboard(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> StatisticsDashboardData:
        self.db_manager.initialize()
        where_sql, params = _date_filter(start_date, end_date)
        ok_placeholders = ",".join("?" for _ in OK_STATUSES)
        ng_placeholders = ",".join("?" for _ in NG_STATUSES)

        with self.db_manager.get_connection() as connection:
            status_row = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS total_count,
                    SUM(CASE WHEN status IN ({ok_placeholders}) THEN 1 ELSE 0 END) AS ok_count,
                    SUM(CASE WHEN status IN ({ng_placeholders}) THEN 1 ELSE 0 END) AS ng_count
                FROM inspections i
                {where_sql}
                """,
                (*OK_STATUSES, *NG_STATUSES, *params),
            ).fetchone()
            defect_rows = connection.execute(
                f"""
                SELECT d.defect_type, COUNT(*) AS count
                FROM defects d
                JOIN inspections i ON i.id = d.inspection_id
                {where_sql}
                {"AND" if where_sql else "WHERE"} i.status IN ({ng_placeholders})
                GROUP BY d.defect_type
                ORDER BY count DESC, d.defect_type
                """,
                (*params, *NG_STATUSES),
            ).fetchall()
            trend_rows = connection.execute(
                f"""
                SELECT date(i.inspected_at) AS period_label, COUNT(*) AS count
                FROM inspections i
                {where_sql}
                {"AND" if where_sql else "WHERE"} i.status IN ({ng_placeholders})
                GROUP BY date(i.inspected_at)
                ORDER BY date(i.inspected_at)
                """,
                (*params, *NG_STATUSES),
            ).fetchall()

        summary = StatisticsResult(
            total_count=int(status_row["total_count"] or 0),
            ok_count=int(status_row["ok_count"] or 0),
            ng_count=int(status_row["ng_count"] or 0),
            defect_type_counts={
                str(row["defect_type"]): int(row["count"])
                for row in defect_rows
            },
        )
        defect_counts = [
            DefectTypeCount(str(row["defect_type"]), int(row["count"]))
            for row in defect_rows
        ]
        ng_trend = _fill_daily_trend(
            [NgTrendPoint(str(row["period_label"]), int(row["count"])) for row in trend_rows],
            start_date=start_date,
            end_date=end_date,
        )
        return StatisticsDashboardData(
            summary=summary,
            defect_counts=defect_counts,
            ng_trend=ng_trend,
            top_defect_types=_top_defect_types(defect_counts),
        )


def _date_filter(start_date: str | None, end_date: str | None) -> tuple[str, tuple[str, ...]]:
    clauses: list[str] = []
    params: list[str] = []
    if start_date:
        clauses.append("i.inspected_at >= ?")
        params.append(_start_datetime(start_date))
    if end_date:
        clauses.append("i.inspected_at <= ?")
        params.append(_end_datetime(end_date))
    if not clauses:
        return "", ()
    return "WHERE " + " AND ".join(clauses), tuple(params)


def _start_datetime(value: str) -> str:
    return f"{value}T00:00:00" if len(value) == 10 else value


def _end_datetime(value: str) -> str:
    return f"{value}T23:59:59" if len(value) == 10 else value


def _fill_daily_trend(
    points: list[NgTrendPoint],
    *,
    start_date: str | None,
    end_date: str | None,
) -> list[NgTrendPoint]:
    if not start_date or not end_date:
        return points

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start is None or end is None or start > end:
        return points
    if (end - start).days > 370:
        return points

    by_label = {point.period_label: point.count for point in points}
    filled: list[NgTrendPoint] = []
    current = start
    while current <= end:
        label = current.isoformat()
        filled.append(NgTrendPoint(label, by_label.get(label, 0)))
        current += timedelta(days=1)
    return filled


def _parse_date(value: str) -> date | None:
    try:
        return datetime.fromisoformat(value[:10]).date()
    except ValueError:
        return None


def _top_defect_types(items: list[DefectTypeCount]) -> list[DefectTypeCount]:
    if not items:
        return []
    max_count = items[0].count
    return [item for item in items if item.count == max_count]
