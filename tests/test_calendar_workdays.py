from __future__ import annotations

import pandas as pd

from src.calendar_utils import US_HOLIDAY_CALENDAR, business_days_for_dates


def test_business_days_exclude_us_federal_holidays():
    dates = pd.DatetimeIndex([pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-01")])
    result = business_days_for_dates(dates)
    assert len(result) == 2

    for idx, month_start in enumerate(dates):
        month_end = month_start + pd.offsets.MonthEnd(1)
        baseline = len(pd.bdate_range(start=month_start, end=month_end))
        holidays = US_HOLIDAY_CALENDAR.holidays(start=month_start, end=month_end)
        weekday_holidays = sum(1 for h in holidays if h.weekday() < 5)
        assert int(result[idx]) == baseline - weekday_holidays

