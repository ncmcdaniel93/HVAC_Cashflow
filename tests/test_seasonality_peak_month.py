from __future__ import annotations

import numpy as np

from src.model import _seasonality_multiplier


def test_seasonality_peaks_on_selected_month():
    months = np.arange(1, 13)
    peak_month = 8
    amplitude = 0.2
    mult = _seasonality_multiplier(months, amplitude, peak_month)

    assert int(months[int(np.argmax(mult))]) == peak_month
    opposite_month = ((peak_month + 6 - 1) % 12) + 1
    assert int(months[int(np.argmin(mult))]) == opposite_month

