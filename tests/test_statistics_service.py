from service.statistics_service import StatisticsService


def test_statistics_service_returns_empty_summary() -> None:
    summary = StatisticsService().build_summary()

    assert summary.total_count == 0
    assert summary.ok_count == 0
    assert summary.ng_count == 0
    assert summary.defect_type_counts == {}
