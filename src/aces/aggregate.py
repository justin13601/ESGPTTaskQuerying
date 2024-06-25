"""This module contains the functions for aggregating windows over conditional row bounds."""

from datetime import timedelta

import polars as pl

from .types import (
    EVENT_INDEX_COLUMN,
    PRED_CNT_TYPE,
    TemporalWindowBounds,
    ToEventWindowBounds,
)


def aggregate_temporal_window(
    predicates_df: pl.DataFrame,
    endpoint_expr: TemporalWindowBounds | tuple[bool, timedelta, bool, timedelta | None],
) -> pl.DataFrame:
    """Aggregates the predicates dataframe into the specified temporal buckets.

    # TODO: Use https://hypothesis.readthedocs.io/en/latest/quickstart.html to add extra tests.

    Args:
        predicates_df: The dataframe containing the predicates. The input must be sorted in ascending order by
            timestamp within each subject group. It must contain the following columns:
              - A column ``subject_id`` which contains the subject ID.
              - A column ``timestamp`` which contains the timestamp at which the event contained in any given
                row occurred.
              - A set of "predicate" columns that contain counts of the number of times a given predicate
                is satisfied in the event contained in any given row.
        endpoint_expr: The expression defining the temporal window endpoints. Can be specified as a tuple or a
            TemporalWindowBounds object, which is just a named tuple of the expected form. Said expected form
            is as follows:
              - The first element is a boolean indicating whether the start of the window is inclusive.
              - The second element is a timedelta object indicating the size of the window.
              - The third element is a boolean indicating whether the end of the window is inclusive.
              - The fourth element is a timedelta object indicating the offset from the timestamp of the row
                to the start of the window.

    Returns:
        The dataframe that has been aggregated temporally according to the specified ``endpoint_expr``. This
        dataframe will contain the same number of rows and be in the same order (in terms of ``subject_id``
        and ``timestamp``) as the input dataframe. The column names will also be the same as the input
        dataframe, but the values in the predicate columns of the output dataframe will be the sum of the
        values in the predicate columns of the input dataframe from the timestamp of the event in the row of
        the output dataframe spanning the specified temporal window.

        The dataframe that has been aggregated in a temporal manner according to the specified
        ``endpoint_expr``. This aggregation means the following:
          - The output dataframe will contain the same number of rows and be in the same order (in terms of
            ``subject_id`` and ``timestamp``) as the input dataframe.
          - The column names will also be the same as the input dataframe, plus there will be two additional
            column, ``timestamp_at_end`` that contains the timestamp of the end of the aggregation window for
            each row (or null if no such aggregation window exists) and ``timestamp_at_start`` that contains
            the timestamp at the start of the aggregation window corresponding to the row.
          - The values in the predicate columns of the output dataframe will contain the sum of the values in
            the predicate columns of the input dataframe from the timestamp of the event in any given row plus
            the specified offset (``endpoint_expr[3]``) to the timestamp of the given row plus the offset plus
            the window size (``endpoint_expr[1]``), less either the start or end of the window pending the
            ``left_inclusive`` (``endpoint_expr[0]``) and ``right_inclusive`` (``endpoint_expr[2]``) values.
            The sum of the predicate columns over the rows in the original dataframe spanning each row's
            ``timestamp + offset`` to each row's ``timestamp + offset + window_size`` should be exactly equal
            to the values in the predicate columns of the output dataframe (again, less the left or right
            inclusive values as specified).

    Examples:
        >>> import polars as pl
        >>> _ = pl.Config.set_tbl_width_chars(150)
        >>> from datetime import datetime, timedelta
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 2, 2],
        ...     "timestamp": [
        ...         # Subject 1
        ...         datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=2, hour=5,  minute=17),
        ...         datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=6, hour=11, minute=0),
        ...         # Subject 2
        ...         datetime(year=1989, month=12, day=1, hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=3, hour=15, minute=17),
        ...     ],
        ...     "is_A": [1, 0, 1, 0, 0, 0],
        ...     "is_B": [0, 1, 0, 1, 1, 0],
        ...     "is_C": [1, 1, 0, 0, 1, 0],
        ... })
        >>> aggregate_temporal_window(df, TemporalWindowBounds(
        ... True, timedelta(days=7), True, None)).drop("_EVENT_INDEX")
        shape: (6, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-08 12:03:00 ┆ 2    ┆ 2    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-02 05:17:00 ┆ 1989-12-09 05:17:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-09 12:03:00 ┆ 1    ┆ 1    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-06 11:00:00 ┆ 1989-12-13 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 1989-12-08 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 1989-12-10 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (
        ... True, timedelta(days=1), True, timedelta(days=0))).drop("_EVENT_INDEX")
        shape: (6, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 2    ┆ 1    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-02 05:17:00 ┆ 1989-12-03 05:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-03 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-06 11:00:00 ┆ 1989-12-07 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 1989-12-02 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (
        ... True, timedelta(days=1), False, timedelta(days=0))).drop("_EVENT_INDEX")
        shape: (6, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1    ┆ 1    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-02 05:17:00 ┆ 1989-12-03 05:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-03 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-06 11:00:00 ┆ 1989-12-07 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 1989-12-02 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (
        ... False, timedelta(days=1), False, timedelta(days=0))).drop("_EVENT_INDEX")
        shape: (6, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-02 05:17:00 ┆ 1989-12-03 05:17:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-03 12:03:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-06 11:00:00 ┆ 1989-12-07 11:00:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 1989-12-02 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (
        ... False, timedelta(days=-1), False, timedelta(days=0))).drop("_EVENT_INDEX")
        shape: (6, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-11-30 12:03:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-02 05:17:00 ┆ 1989-12-01 05:17:00 ┆ 1    ┆ 0    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-06 11:00:00 ┆ 1989-12-05 11:00:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 1989-11-30 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 1989-12-02 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (
        ... False, timedelta(hours=12), False, timedelta(hours=12))).drop("_EVENT_INDEX")
        shape: (6, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-02 00:03:00 ┆ 1989-12-02 12:03:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-02 17:17:00 ┆ 1989-12-03 05:17:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-03 00:03:00 ┆ 1989-12-03 12:03:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-06 23:00:00 ┆ 1989-12-07 11:00:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-02 01:14:00 ┆ 1989-12-02 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 03:17:00 ┆ 1989-12-04 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> # Note that left_inclusive and right_inclusive are relative to the temporal ordering of the window
        >>> # and not the timestamp of the row. E.g., if left_inclusive is False, the window will not include
        >>> # the earliest event in the aggregation window, regardless of whether that is earlier than the
        >>> # timestamp of the row.
        >>> aggregate_temporal_window(df, (
        ... False, timedelta(days=-1), True, timedelta(days=1))).drop("_EVENT_INDEX")
        shape: (6, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-03 05:17:00 ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-03 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-07 11:00:00 ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-02 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (
        ... True, timedelta(days=-1), False, timedelta(days=1))).drop("_EVENT_INDEX")
        shape: (6, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1    ┆ 1    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-03 05:17:00 ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-03 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-07 11:00:00 ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-02 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
    """
    if not isinstance(endpoint_expr, TemporalWindowBounds):
        endpoint_expr = TemporalWindowBounds(*endpoint_expr)

    predicate_cols = [
        c for c in predicates_df.columns if c not in {"subject_id", "timestamp", EVENT_INDEX_COLUMN}
    ]

    return (
        predicates_df.rolling(
            index_column="timestamp",
            group_by="subject_id",
            **endpoint_expr.polars_gp_rolling_kwargs,
        )
        .agg(
            *[pl.col(c).sum().cast(PRED_CNT_TYPE).alias(c) for c in predicate_cols],
            pl.col(EVENT_INDEX_COLUMN).max(),
        )
        .sort(by=["subject_id", "timestamp"])
        .select(
            "subject_id",
            "timestamp",
            (pl.col("timestamp") + endpoint_expr.offset).alias("timestamp_at_start"),
            (pl.col("timestamp") + endpoint_expr.offset + endpoint_expr.window_size).alias(
                "timestamp_at_end"
            ),
            *predicate_cols,
            EVENT_INDEX_COLUMN,
        )
    )


def aggregate_event_bound_window(
    predicates_df: pl.DataFrame,
    endpoint_expr: ToEventWindowBounds | tuple[bool, str, bool, timedelta | None],
) -> pl.DataFrame:
    """Aggregates ``predicates_df`` between each row plus an offset and the next per-subject matching event.

    # TODO: Use https://hypothesis.readthedocs.io/en/latest/quickstart.html to test this function.

    See the testing for ``boolean_expr_bound_sum`` for more comprehensive examples and test cases for the
    underlying API here, and the testing for ``ToEventWindowBounds`` for how the bounding syntax is converted
    into arguments for the ``boolean_expr_bound_sum`` function.

    Args:
        predicates_df: The dataframe containing the predicates. The input must be sorted in ascending order by
            timestamp within each subject group. It must contain the following columns:
              - A column ``subject_id`` which contains the subject ID.
              - A column ``timestamp`` which contains the timestamp at which the event contained in any given
                row occurred.
              - A set of "predicate" columns that contain counts of the number of times a given predicate
                is satisfied in the event contained in any given row.
        endpoint_expr: The expression defining the event bound window endpoints. Can be specified as a tuple
            or a ToEventWindowBounds object, which is just a named tuple of the expected form. Said expected
            form is as follows:
              - The first element is a boolean indicating whether the start of the window is inclusive.
              - The second element is a string indicating the name of the column in which non-zero counts
                indicate the event is a valid "end event" of the window.
              - The third element is a boolean indicating whether the end of the window is inclusive.
              - The fourth element is a timedelta object indicating the offset from the timestamp of the row
                to the start of the window. The offset here can _only_ be positive.

    Returns:
        The dataframe that has been aggregated in an event bound manner according to the specified
        ``endpoint_expr``. This aggregation means the following:
          - The output dataframe will contain the same number of rows and be in the same order (in terms of
            ``subject_id`` and ``timestamp``) as the input dataframe.
          - The column names will also be the same as the input dataframe, plus there will be two additional
            column, ``timestamp_at_end`` that contains the timestamp of the end of the aggregation window for
            each row (or null if no such aggregation window exists) and ``timestamp_at_start`` that contains
            the timestamp at the start of the aggregation window corresponding to the row.
          - The values in the predicate columns of the output dataframe will contain the sum of the values in
            the predicate columns of the input dataframe from the timestamp of the event in any given row plus
            the specified offset (``endpoint_expr[3]``) to the timestamp of the next row for that subject in
            the input dataframe such that the specified event predicate (``endpoint_expr[1]``) has a value
            greater than zero for that patient, less either the start or end of the window pending the
            ``left_inclusive`` (``endpoint_expr[0]``) and ``right_inclusive`` (``endpoint_expr[2]``) values.
            If there is no valid "next row" for a given event, the values in the predicate columns of the
            output dataframe will be 0, as the sum of an empty set is 0.
            The sum of the predicate columns over the rows in the original dataframe spanning each row's
            ``timestamp + offset`` to each row's ``timestamp_at_end`` should be exactly equal to the values in
            the predicate columns of the output dataframe.

    Raises:
        ValueError: If the offset is negative.

    Examples:
        >>> import polars as pl
        >>> _ = pl.Config.set_tbl_width_chars(150)
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 2, 2, 2],
        ...     "timestamp": [
        ...         # Subject 1
        ...         datetime(year=1989, month=12, day=1,  hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=3,  hour=13, minute=14), # HAS EVENT BOUND
        ...         datetime(year=1989, month=12, day=5,  hour=15, minute=17),
        ...         # Subject 2
        ...         datetime(year=1989, month=12, day=2,  hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=4,  hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=6,  hour=15, minute=17), # HAS EVENT BOUND
        ...         datetime(year=1989, month=12, day=8,  hour=16, minute=22),
        ...         datetime(year=1989, month=12, day=10, hour=3,  minute=7),  # HAS EVENT BOUND
        ...     ],
        ...     "_EVENT_INDEX": [0, 1, 2, 0, 1, 2, 3, 4],
        ...     "is_A": [1, 0, 1, 1, 1, 1, 0, 0],
        ...     "is_B": [0, 1, 0, 1, 0, 1, 1, 1],
        ...     "is_C": [0, 1, 0, 0, 0, 1, 0, 1],
        ... })
        >>> aggregate_event_bound_window(df, ToEventWindowBounds(True, "is_C", True, None)).drop(
        ... "_EVENT_INDEX")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-06 15:17:00 ┆ 3    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 2    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_event_bound_window(df, ToEventWindowBounds(True, "is_C", False, None)).drop(
        ... "_EVENT_INDEX")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-06 15:17:00 ┆ 2    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_event_bound_window(df, ToEventWindowBounds(False, "is_C", True, None)).drop(
        ... "_EVENT_INDEX")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-06 15:17:00 ┆ 2    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_event_bound_window(df, ToEventWindowBounds(
        ... True, "is_C", True, timedelta(days=3))).drop("_EVENT_INDEX")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-05 12:03:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-07 13:14:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-09 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
    """
    if not isinstance(endpoint_expr, ToEventWindowBounds):
        endpoint_expr = ToEventWindowBounds(*endpoint_expr)

    return boolean_expr_bound_sum(predicates_df, **endpoint_expr.boolean_expr_bound_sum_kwargs)


def boolean_expr_bound_sum(
    df: pl.DataFrame,
    boundary_expr: pl.Expr,
    mode: str,
    closed: str,
    offset: timedelta = timedelta(0),
) -> pl.DataFrame:
    """Sums all columns of ``df`` between each row plus an offset and the next per-subject satisfying event.

    # TODO: Use https://hypothesis.readthedocs.io/en/latest/quickstart.html to test this function.

    Performs a boolean-expression-bounded summation over the columns of ``df``. The logic of this is as
    follows.
      - If ``mode`` is ``bound_to_row``, then that means that the _left_ endpoint of the
        window must correspond to an instance where ``boundary_expr`` is `True` and the _right_ endpoint will
        correspond to a row in the dataframe.
      - If ``mode`` is ``row_to_bound``, then that means that the _right_ endpoint of the
        window must correspond to an instance where ``boundary_expr`` is `True` and the _left_ endpoint will
        correspond to a row in the dataframe.
      - If ``closed`` is ``'none'``, then neither the associated left row endpoint nor the associated right
        row endpoint will be included in the summation.
      - If ``closed`` is ``'both'``, then both the associated left row endpoint and the associated right
        row endpoint will be included in the summation.
      - If ``closed`` is ``'left'``, then only the associated left row endpoint will be included in the
        summation.
      - If ``closed`` is ``'right'``, then only the associated right row endpoint will be included in the
        summation.
      - The output dataframe will have the same number of rows and order as the input dataframe. Each row will
        correspond to the aggregation over the matching window that uses that row as the "row" side of the
        terminating aggregation (regardless of whether that corresponds to the left or the right endpoint, and
        regardless of whether that endpoint would actually be included in the calculation based on
        ``closed``). The associated ``boundary_expr`` side of the endpoint will be the nearest possible
        ``boundary_expr`` endpoint that produces a non-empty set of rows over which to aggregate. Note that
        this may depend on the value of ``closed`` -- e.g., a row with ``boundary_expr`` being `True` can
        produce a one-element set of rows to aggregate if ``closed`` is ``'both'``, but this is not possible
        if ``closed`` is something else, and in that case that same row may instead rely on a different row to
        fill its ``boundary_expr`` endpoint when evaluated as a "row endpoint" during calculation.
      - Offset further modifies this logic by applying a temporal offset of fixed size to the "row" endpoint.
        All other logic stays the same.


    In particular, suppose that we have following rows and boolean boundary expression evaluations (for a
    single subject):

    ```markdown
    Rows:                  [0,      1,      2,      3,      4,    5,      6]
    Boundary Expression:   [False,  True,   False,  True,   True, False,  False]
    ```

    Then, we would aggregate the following rows under the following specified conditions:

    ```markdown
    mode         | closed | aggregate_groups
    -------------|--------|------------------------------------------------------
    bound_to_row |  both  | [],     [1],    [1, 2], [3],    [4],  [4, 5], [4, 5, 6]
    bound_to_row |  left  | [],     [],     [1],    [1, 2], [3],  [4],    [4, 5]
    bound_to_row |  right | [],     [],     [2],    [2, 3], [4],  [5],    [5, 6]
    bound_to_row |  none  | [],     [],     [],     [2],    [],   [],     [5]
    row_to_bound |  both  | [0, 1], [1],    [2, 3], [3],    [4],  [],     []
    row_to_bound |  left  | [0],    [],     [2],    [],     [],   [],     []
    row_to_bound |  right | [1],    [2, 3], [3],    [4],    [],   [],     []
    row_to_bound |  none  | [],     [2],    [],     [],     [],   [],     []
    ```

    How to think about this? the ``closed`` parameter controls where we put the endpoints to merge for our
    bounds. Consider the case accented with ** above, where we have the row being row 1 (indexing from 0), we
    are in a ``row_to_bound`` setting, and we have ``closed = "none"``. For this, as the left endpoint (here
    the row, as we are in ``row_to_bound`` is not included, so our boundary on the left is between row 2 and
    3. Our boundary on the right is just to the left of the next row which has the True boolean value. In this
    case, that means between row 2 and 3 (as row 1 is to the left of our left endpoint). In contrast, if
    closed were ``"left"``, then row 1 would be a possible bound row to use on the right, and thus by virtue
    of our right endpoint being open, we would include no rows.

    Args:
        df: The dataframe to be aggregated. The input must be sorted in ascending order by
            timestamp within each subject group. It must contain the following columns:
              - A column ``subject_id`` which contains the subject ID.
              - A column ``timestamp`` which contains the timestamp at which the event contained in any given
                row occurred.
              - A set of other columns that will be summed over the specified windows.
        boundary_expr: A boolean expression which can be evaluated over the passed dataframe, and defines rows
            that can serve as valid left or right boundaries for the aggregation step. The precise window over
            which the dataframe will be aggregated will depend on ``mode`` and ``closed``. See above for an
            explanation of the various configurations possible.
        mode:
        closed:
        offset:

    Returns:
        The dataframe that has been aggregated in an event bound manner according to the specified
        ``endpoint_expr``. This aggregation means the following:
          - The output dataframe will contain the same number of rows and be in the same order (in terms of
            ``subject_id`` and ``timestamp``) as the input dataframe.
          - The column names will also be the same as the input dataframe, plus there will be two additional
            column, ``timestamp_at_end`` that contains the timestamp of the end of the aggregation window for
            each row (or null if no such aggregation window exists) and ``timestamp_at_start`` that contains
            the timestamp at the start of the aggregation window corresponding to the row.
          - The values in the predicate columns of the output dataframe will contain the sum of the values
            over the permissible row ranges given the input parameters. See above for an explanation.

    Examples:
        >>> import polars as pl
        >>> _ = pl.Config.set_tbl_width_chars(150)
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 2, 2, 2],
        ...     "timestamp": [
        ...         # Subject 1
        ...         datetime(year=1989, month=12, day=1,  hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=3,  hour=13, minute=14), # HAS EVENT BOUND
        ...         datetime(year=1989, month=12, day=5,  hour=15, minute=17),
        ...         # Subject 2
        ...         datetime(year=1989, month=12, day=2,  hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=4,  hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=6,  hour=15, minute=17), # HAS EVENT BOUND
        ...         datetime(year=1989, month=12, day=8,  hour=16, minute=22),
        ...         datetime(year=1989, month=12, day=10, hour=3,  minute=7),  # HAS EVENT BOUND
        ...     ],
        ...     "_EVENT_INDEX":  [0, 1, 2, 0, 1, 2, 3, 4],
        ...     "idx":  [0, 1, 2, 3, 4, 5, 6, 7],
        ...     "is_A": [1, 0, 1, 1, 1, 1, 0, 0],
        ...     "is_B": [0, 1, 0, 1, 0, 1, 1, 1],
        ...     "is_C": [0, 1, 0, 0, 0, 1, 0, 1],
        ... })
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "bound_to_row",
        ...     "both",
        ... ).drop("idx")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-05 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 2    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-08 16:22:00 ┆ 2    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "bound_to_row",
        ...     "none",
        ... ).drop("idx")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-05 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-08 16:22:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 2    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "bound_to_row",
        ...     "left",
        ... ).drop("idx")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-05 15:17:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-08 16:22:00 ┆ 2    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "bound_to_row",
        ...     "right",
        ... ).drop("idx")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-05 15:17:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-08 16:22:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 3    ┆ 2    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "row_to_bound",
        ...     "both",
        ... ).drop("idx")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-04 13:14:00 ┆ 2    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 3    ┆ 2    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "row_to_bound",
        ...     "none",
        ... ).drop("idx")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-04 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "row_to_bound",
        ...     "left",
        ... ).drop("idx")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-04 13:14:00 ┆ 1    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-10 03:07:00 ┆ 2    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "row_to_bound",
        ...     "right",
        ... ).drop("idx")
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 1989-12-04 13:14:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> #### WITH OFFSET ####
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "bound_to_row",
        ...     "both",
        ...     offset = timedelta(days=3),
        ... ).drop(["idx", "_EVENT_INDEX"])
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-04 12:03:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-06 13:14:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-08 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-05 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-07 13:14:00 ┆ 2    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-09 15:17:00 ┆ 2    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-11 16:22:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-13 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "bound_to_row",
        ...     "left",
        ...     offset = timedelta(days=3),
        ... ).drop(["idx", "_EVENT_INDEX"])
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-04 12:03:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-06 13:14:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ 1989-12-03 13:14:00 ┆ 1989-12-08 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-05 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-07 13:14:00 ┆ 2    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-09 15:17:00 ┆ 2    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-11 16:22:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 1989-12-13 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "bound_to_row",
        ...     "none",
        ...     timedelta(days=-3),
        ... ).drop(["idx", "_EVENT_INDEX"])
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-05 16:22:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-07 03:07:00 ┆ 1    ┆ 1    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "bound_to_row",
        ...     "right",
        ...     offset = timedelta(days=-3),
        ... ).drop(["idx", "_EVENT_INDEX"])
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-05 16:22:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-04 13:14:00 ┆ 1989-12-07 03:07:00 ┆ 1    ┆ 1    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "row_to_bound",
        ...     "both",
        ...     offset = timedelta(days=3),
        ... ).drop(["idx", "_EVENT_INDEX"])
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-05 12:03:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 3    ┆ 2    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-07 13:14:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-09 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "row_to_bound",
        ...     "left",
        ...     offset = timedelta(days=3),
        ... ).drop(["idx", "_EVENT_INDEX"])
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-05 12:03:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-07 13:14:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-09 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "row_to_bound",
        ...     "none",
        ...     offset = timedelta(days=-3),
        ... ).drop(["idx", "_EVENT_INDEX"])
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-11-28 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-11-30 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ 1989-12-02 15:17:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-11-29 12:03:00 ┆ 1989-12-04 13:14:00 ┆ 1    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 1    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-05 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-07 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> boolean_expr_bound_sum(
        ...     df,
        ...     pl.col("idx").is_in([1, 4, 7]),
        ...     "row_to_bound",
        ...     "right",
        ...     offset = timedelta(days=-3),
        ... ).drop(["idx", "_EVENT_INDEX"])
        shape: (8, 7)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_start  ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-11-28 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-11-30 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ 1989-12-02 15:17:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-11-29 12:03:00 ┆ 1989-12-04 13:14:00 ┆ 2    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 1989-12-04 13:14:00 ┆ 2    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 13:14:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-05 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 3    ┆ 2    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ 1989-12-07 03:07:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 2    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
    """
    if mode not in ("bound_to_row", "row_to_bound"):
        raise ValueError(f"Mode '{mode}' invalid!")
    if closed not in ("both", "none", "left", "right"):
        raise ValueError(f"Closed '{closed}' invalid!")

    if offset != timedelta(0):
        if offset > timedelta(0):
            left_inclusive = False
            if mode == "row_to_bound":
                # Here, we'll be taking cumsum_at_bound - cumsum_at_row - aggd_over_offset
                if closed in ("left", "both"):
                    right_inclusive = False
                else:
                    right_inclusive = True
            else:
                # Here, we'll be taking cumsum_at_row - cumsum_at_bound + aggd_over_offset
                if closed in ("right", "both"):
                    right_inclusive = True
                else:
                    right_inclusive = False
        else:
            right_inclusive = False
            if mode == "row_to_bound":
                # Here, we'll be taking cumsum_at_bound - cumsum_at_row + aggd_over_offset
                if closed in ("left", "both"):
                    left_inclusive = True
                else:
                    left_inclusive = False
            else:
                # Here, we'll be taking cumsum_at_row - cumsum_at_bound - aggd_over_offset
                if closed in ("right", "both"):
                    left_inclusive = False
                else:
                    left_inclusive = True

        aggd_over_offset = aggregate_temporal_window(
            df,
            TemporalWindowBounds(
                left_inclusive=left_inclusive,
                window_size=offset,
                right_inclusive=right_inclusive,
                offset=None,
            ),
        )

    cols = [c for c in df.columns if c not in {"subject_id", "timestamp", EVENT_INDEX_COLUMN}]

    cumsum_cols = {c: pl.col(c).cum_sum().over("subject_id").alias(f"{c}_cumsum_at_row") for c in cols}
    df = df.with_columns(*cumsum_cols.values())

    cumsum_at_boundary = {c: pl.col(f"{c}_cumsum_at_row").alias(f"{c}_cumsum_at_boundary") for c in cols}

    # We need to adjust `cumsum_at_boundary` to appropriately include or exclude the boundary event.
    if mode == "bound_to_row" and closed in ("left", "both"):
        cumsum_at_boundary = {
            c: (expr - pl.col(c)).alias(f"{c}_cumsum_at_boundary") for c, expr in cumsum_at_boundary.items()
        }
    elif mode == "row_to_bound" and closed not in ("right", "both"):
        cumsum_at_boundary = {
            c: (expr - pl.col(c)).alias(f"{c}_cumsum_at_boundary") for c, expr in cumsum_at_boundary.items()
        }

    timestamp_offset = pl.col("timestamp") - offset
    if mode == "bound_to_row":
        if closed in ("left", "both"):
            timestamp_offset -= timedelta(seconds=1e-6)
        else:
            timestamp_offset += timedelta(seconds=1e-6)

        fill_strategy = "forward"
        sum_exprs = {
            c: (
                pl.col(f"{c}_cumsum_at_row")
                - pl.col(f"{c}_cumsum_at_boundary").fill_null(strategy=fill_strategy).over("subject_id")
            ).alias(c)
            for c in cols
        }
        if (closed in ("left", "none") and offset <= timedelta(0)) or offset < timedelta(0):
            # If we either don't include the right endpoint due to the closed value and the lack of a positive
            # offset or we have a negative offset, we want to remove the right endpoint's counts from the
            # cumsum difference.
            sum_exprs = {c: expr - pl.col(c) for c, expr in sum_exprs.items()}
    else:
        if closed in ("right", "both"):
            timestamp_offset += timedelta(seconds=1e-6)
        else:
            timestamp_offset -= timedelta(seconds=1e-6)

        fill_strategy = "backward"
        sum_exprs = {
            c: (
                pl.col(f"{c}_cumsum_at_boundary").fill_null(strategy=fill_strategy).over("subject_id")
                - pl.col(f"{c}_cumsum_at_row")
            ).alias(c)
            for c in cols
        }
        if (closed in ("left", "both") and offset <= timedelta(0)) or offset < timedelta(0):
            # If we either do include the left endpoint due to the closed value and the lack of a positive
            # offset or we have a negative offset, we want to add in the left endpoint's (the row's) counts
            # As they will always be included.
            sum_exprs = {c: expr + pl.col(c) for c, expr in sum_exprs.items()}

    at_boundary_df = df.filter(boundary_expr).select(
        "subject_id",
        pl.col("timestamp").alias("timestamp_at_boundary"),
        timestamp_offset.alias("timestamp"),
        *cumsum_at_boundary.values(),
        pl.lit(False).alias("is_real"),
    )

    with_at_boundary_events = (
        pl.concat([df.with_columns(pl.lit(True).alias("is_real")), at_boundary_df], how="diagonal")
        .sort(by=["subject_id", "timestamp"])
        .select(
            "subject_id",
            "timestamp",
            pl.col("timestamp_at_boundary").fill_null(strategy=fill_strategy).over("subject_id"),
            *sum_exprs.values(),
            "is_real",
        )
        .filter("is_real")
        .drop("is_real")
    )

    if mode == "bound_to_row":
        st_timestamp_expr = pl.col("timestamp_at_boundary")
        end_timestamp_expr = pl.when(pl.col("timestamp_at_boundary").is_not_null()).then(
            pl.col("timestamp") + offset
        )
    else:
        st_timestamp_expr = pl.when(pl.col("timestamp_at_boundary").is_not_null()).then(
            pl.col("timestamp") + offset
        )
        end_timestamp_expr = pl.col("timestamp_at_boundary")

    if offset == timedelta(0):
        return with_at_boundary_events.select(
            "subject_id",
            "timestamp",
            st_timestamp_expr.alias("timestamp_at_start"),
            end_timestamp_expr.alias("timestamp_at_end"),
            *(pl.col(c).cast(PRED_CNT_TYPE).fill_null(0).alias(c) for c in cols),
        )

    if mode == "bound_to_row" and offset > timedelta(0):

        def agg_offset_fn(c: str) -> pl.Expr:
            return pl.col(c) + pl.col(f"{c}_in_offset_period")

    elif mode == "bound_to_row" and offset < timedelta(0):

        def agg_offset_fn(c: str) -> pl.Expr:
            return pl.col(c) - pl.col(f"{c}_in_offset_period")

    elif mode == "row_to_bound" and offset > timedelta(0):

        def agg_offset_fn(c: str) -> pl.Expr:
            return pl.col(c) - pl.col(f"{c}_in_offset_period")

    elif mode == "row_to_bound" and offset < timedelta(0):

        def agg_offset_fn(c: str) -> pl.Expr:
            return pl.col(c) + pl.col(f"{c}_in_offset_period")

    else:
        raise ValueError(f"Mode '{mode}' and offset '{offset}' invalid!")

    return with_at_boundary_events.join(
        aggd_over_offset,
        on=["subject_id", "timestamp"],
        how="left",
        suffix="_in_offset_period",
    ).select(
        "subject_id",
        "timestamp",
        st_timestamp_expr.alias("timestamp_at_start"),
        end_timestamp_expr.alias("timestamp_at_end"),
        *(agg_offset_fn(c).cast(PRED_CNT_TYPE, strict=False).fill_null(0).alias(c) for c in cols),
        EVENT_INDEX_COLUMN,
    )
