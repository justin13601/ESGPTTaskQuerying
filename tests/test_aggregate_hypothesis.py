import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from datetime import datetime, timedelta

import polars as pl
import polars.selectors as cs
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_series_equal
from polars.testing.parametric import column, dataframes

from aces.aggregate import aggregate_temporal_window
from aces.types import TemporalWindowBounds

datetime_st = st.datetimes(min_value=datetime(1989, 12, 1), max_value=datetime(1999, 12, 31))

N_PREDICATES = 5
PREDICATE_DATAFRAMES = dataframes(
    cols=[
        column("subject_id", allow_null=False, dtype=pl.UInt32),
        column("timestamp", allow_null=False, dtype=pl.Datetime("ms"), strategy=datetime_st),
        *[column(f"predicate_{i}", allow_null=False, dtype=pl.UInt8) for i in range(1, N_PREDICATES + 1)],
    ],
    min_size=1,
    max_size=50,
)


@given(
    df=PREDICATE_DATAFRAMES,
    left_inclusive=st.booleans(),
    right_inclusive=st.booleans(),
    window_size=st.timedeltas(min_value=timedelta(days=1), max_value=timedelta(days=365 * 5)),
    offset=st.timedeltas(min_value=timedelta(days=0), max_value=timedelta(days=365)),
)
@settings(max_examples=50)
def test_aggregate_temporal_window(
    df: pl.DataFrame, left_inclusive: bool, right_inclusive: bool, window_size: timedelta, offset: timedelta
):
    """Tests whether calling the `aggregate_temporal_window` function works produces a consistent output."""

    max_N_subjects = 3
    df = df.with_columns(
        (pl.col("subject_id") % max_N_subjects).alias("subject_id"),
        cs.starts_with("predicate_").cast(pl.Int32).name.keep(),
    ).sort("subject_id", "timestamp")

    endpoint_expr = TemporalWindowBounds(
        left_inclusive=left_inclusive, right_inclusive=right_inclusive, window_size=window_size, offset=offset
    )

    # Should run:
    agg_df = aggregate_temporal_window(df.lazy(), endpoint_expr)
    assert agg_df is not None
    agg_df = agg_df.collect()

    # This will return something of the below form:
    #
    # shape: (6, 7)
    # ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
    # │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
    # │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
    # │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
    # ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
    # │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1    ┆ 1    ┆ 2    │
    # │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-03 05:17:00 ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 1    ┆ 1    │
    # │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-03 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
    # │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-07 11:00:00 ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
    # │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-02 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
    # │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
    # └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
    #
    # We're going to validate this by asserting that the sums of the predicate columns between the rows
    # for a given subject are consistent.

    assert set(df.columns).issubset(set(agg_df.columns))
    assert len(agg_df.columns) == len(df.columns) + 2
    assert "timestamp_at_start" in agg_df.columns
    assert "timestamp_at_end" in agg_df.columns
    assert_series_equal(agg_df["subject_id"], df["subject_id"])
    assert_series_equal(agg_df["timestamp"], df["timestamp"])

    # Now we're going to validate the sums of the predicate columns between the rows for a given subject are
    # consistent.
    for subject_id in range(max_N_subjects):
        if subject_id not in df["subject_id"]:
            assert subject_id not in agg_df["subject_id"]
            continue

        raw_subj = df.filter(pl.col("subject_id") == subject_id)
        agg_subj = agg_df.filter(pl.col("subject_id") == subject_id)

        for row in agg_subj.iter_rows(named=True):
            start = row["timestamp_at_start"]
            end = row["timestamp_at_end"]

            if left_inclusive:
                st_filter = pl.col("timestamp") >= start
            else:
                st_filter = pl.col("timestamp") > start

            if right_inclusive:
                et_filter = pl.col("timestamp") <= end
            else:
                et_filter = pl.col("timestamp") < end

            raw_filtered = raw_subj.filter(st_filter & et_filter)
            if len(raw_filtered) == 0:
                for i in range(1, N_PREDICATES + 1):
                    # TODO: Is this right? Or should it always be one or the other?
                    assert (row[f"predicate_{i}"] is None) or (row[f"predicate_{i}"] == 0)
            else:
                raw_sums = raw_filtered.select(cs.starts_with("predicate_")).sum()
                for i in range(1, N_PREDICATES + 1):
                    assert raw_sums[f"predicate_{i}"].item() == row[f"predicate_{i}"]
