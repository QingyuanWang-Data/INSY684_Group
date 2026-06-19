from __future__ import annotations

import numpy as np
import pandas as pd

from homecredit_service.features import aggregate_categorical_indicators, filter_by_temporal_cutoff


def test_filter_by_temporal_cutoff_keeps_only_non_future_rows() -> None:
    df = pd.DataFrame(
        {
            "MONTHS_BALANCE": [-3, 0, 2, np.nan],
            "value": [1, 2, 3, 4],
        }
    )

    filtered = filter_by_temporal_cutoff(df, ("MONTHS_BALANCE",))

    assert filtered["value"].tolist() == [1, 2, 4]


def test_filter_by_temporal_cutoff_noop_when_columns_missing() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    filtered = filter_by_temporal_cutoff(df, ("DAYS_DECISION",))

    assert filtered.equals(df)


def test_aggregate_categorical_indicators_adds_count_and_share_features() -> None:
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "STATUS": ["Active", "Closed", "Active", "Closed"],
        }
    )

    aggregated = aggregate_categorical_indicators(
        df,
        group_key="SK_ID_CURR",
        prefix="BUREAU_CAT",
        columns=("STATUS",),
        top_n=2,
    )

    assert aggregated is not None
    assert aggregated.loc[1, "BUREAU_CAT_STATUS_ACTIVE_COUNT"] == 2
    assert aggregated.loc[1, "BUREAU_CAT_STATUS_ACTIVE_SHARE"] == 2 / 3
    assert aggregated.loc[2, "BUREAU_CAT_STATUS_CLOSED_SHARE"] == 1.0
