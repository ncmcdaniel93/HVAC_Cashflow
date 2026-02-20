"""Calendar and working-day utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


US_HOLIDAY_CALENDAR = USFederalHolidayCalendar()


@dataclass(frozen=True)
class WorkdaySeries:
    """Derived monthly workday counts and labor hours."""

    workdays: np.ndarray
    tech_hours: np.ndarray
    sales_hours: np.ndarray


def _month_business_days(month_start: pd.Timestamp) -> int:
    month_end = month_start + pd.offsets.MonthEnd(1)
    holidays = US_HOLIDAY_CALENDAR.holidays(start=month_start, end=month_end)
    holiday_list = [h.to_pydatetime() for h in holidays]
    cbd = CustomBusinessDay(holidays=holiday_list)
    bdays = pd.date_range(start=month_start, end=month_end, freq=cbd)
    return int(len(bdays))


def business_days_for_dates(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return Mon-Fri business days per month, excluding US federal holidays."""
    return np.array([_month_business_days(pd.Timestamp(d)) for d in dates], dtype=float)


def derive_workdays_and_hours(
    dates: pd.DatetimeIndex,
    techs: np.ndarray,
    sales_staff: np.ndarray,
    tech_hours_per_day: float,
    sales_hours_per_day: float,
) -> WorkdaySeries:
    workdays = business_days_for_dates(dates)
    tech_hours = techs * workdays * float(tech_hours_per_day)
    sales_hours = sales_staff * workdays * float(sales_hours_per_day)
    return WorkdaySeries(workdays=workdays, tech_hours=tech_hours, sales_hours=sales_hours)
