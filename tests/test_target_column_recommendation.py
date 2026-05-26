from __future__ import annotations

import polars as pl

from scripts.streamlit_single_agent_app import _recommend_target_column


def test_recommend_target_column_accepts_numeric_strings() -> None:
    df = pl.DataFrame(
        {
            "date": [
                "2016-01-11 17:00:00",
                "2016-01-11 17:10:00",
                "2016-01-11 17:20:00",
                "2016-01-11 17:30:00",
                "2016-01-11 17:40:00",
                "2016-01-11 17:50:00",
            ],
            "T1": [19.8, 20.0, 20.1, 20.4, 20.6, 20.8],
            "Visibility": [63.0, 59.1, 55.3, 54.0, 57.0, 58.0],
            "Appliances": ["  60", "  60", "  50", "  70", "  80", "  90"],
            "rv2": ["13.2", "18.6", "28.6", "45.2", "31.3", "26.1"],
        }
    )

    assert _recommend_target_column(df) == "Appliances"


def test_recommend_target_column_prefers_measure_over_calendar_fields() -> None:
    df = pl.DataFrame(
        {
            "Month": [1, 1, 1, 1, 1, 1],
            "Day": [1, 2, 3, 4, 5, 6],
            "Year": [1995, 1995, 1995, 1995, 1995, 1995],
            "AvgTemperature": [64.2, 49.4, 48.8, 46.4, 47.9, 48.7],
        }
    )

    assert _recommend_target_column(df) == "AvgTemperature"


def test_recommend_target_column_prefers_price_over_volume_in_ohlcv_data() -> None:
    df = pl.DataFrame(
        {
            "Date": [
                "2000-01-03",
                "2000-01-04",
                "2000-01-05",
                "2000-01-06",
                "2000-01-07",
                "2000-01-10",
            ],
            "Open": [5.90, 5.60, 5.50, 5.50, 5.50, 5.70],
            "High": [5.88, 5.55, 5.49, 5.47, 5.50, 5.74],
            "Low": [5.88, 5.55, 5.49, 5.47, 5.50, 5.70],
            "Close": [5.88, 5.55, 5.49, 5.47, 5.50, 5.72],
            "Volume": [35389440000, 28861440000, 43033600000, 34055680000, 20912640000, 25405440000],
        }
    )

    assert _recommend_target_column(df) == "Close"