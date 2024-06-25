"""This module contains types defined by this package.

These are all simple types using named tuples so can be safely ignored by downstream users provided data
fields are passed in the correct order.
"""

import dataclasses
from datetime import timedelta

import polars as pl

# The type used for final aggregate counts of predicates.
PRED_CNT_TYPE = pl.Int64

# The type used for event indexing
EVENT_INDEX_TYPE = pl.Int64

# The key used in the endpoint expression to indicate the window should be aggregated to the record start.
START_OF_RECORD_KEY = "_RECORD_START"
END_OF_RECORD_KEY = "_RECORD_END"

# The key used to capture the count of events of any kind that occur in a window.
ANY_EVENT_COLUMN = "_ANY_EVENT"

# The key used for the event index
EVENT_INDEX_COLUMN = "_EVENT_INDEX"
LAST_EVENT_INDEX_COLUMN = "_LAST_EVENT_INDEX"


@dataclasses.dataclass(order=True)
class TemporalWindowBounds:
    """Named tuple to represent temporal window bounds.

    Attributes:
        left_inclusive: The start of the window, inclusive.
        window_size: The size of the window.
        right_inclusive: The end of the window, inclusive.
        offset: The offset from the start of the window to the end of the window.

    Example:
        >>> bounds = TemporalWindowBounds(
        ...     left_inclusive=True,
        ...     window_size=timedelta(days=1),
        ...     right_inclusive=False,
        ...     offset=timedelta(hours=1)
        ... )
        >>> bounds # doctest: +NORMALIZE_WHITESPACE
        TemporalWindowBounds(left_inclusive=True,
                             window_size=datetime.timedelta(days=1),
                             right_inclusive=False,
                             offset=datetime.timedelta(seconds=3600))
        >>> left_inclusive, window_size, right_inclusive, offset = bounds
        >>> bounds.left_inclusive
        True
        >>> window_size
        datetime.timedelta(days=1)
        >>> right_inclusive
        False
        >>> offset
        datetime.timedelta(seconds=3600)
    """

    left_inclusive: bool
    window_size: timedelta
    right_inclusive: bool
    offset: timedelta | None = None

    # Needed to make it accessible like a tuple.
    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))

    # Needed to make it scriptable.
    def __getitem__(self, key):
        return tuple(getattr(self, field.name) for field in dataclasses.fields(self))[key]

    def __post_init__(self):
        if self.offset is None:
            self.offset = timedelta(0)

    @property
    def polars_gp_rolling_kwargs(self) -> dict[str, str | timedelta]:
        """Return the parameters for a group_by rolling operation in Polars.

        Examples:
            >>> TemporalWindowBounds(
            ...     left_inclusive=True,
            ...     window_size=timedelta(days=1),
            ...     right_inclusive=True,
            ...     offset=None
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=1),
             'offset': datetime.timedelta(0),
             'closed': 'both'}
            >>> TemporalWindowBounds(
            ...     left_inclusive=True,
            ...     window_size=timedelta(days=1),
            ...     right_inclusive=True,
            ...     offset=timedelta(hours=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=1),
             'offset': datetime.timedelta(seconds=3600),
             'closed': 'both'}
            >>> TemporalWindowBounds(
            ...     left_inclusive=False,
            ...     window_size=timedelta(days=2),
            ...     right_inclusive=False,
            ...     offset=timedelta(minutes=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=2),
             'offset': datetime.timedelta(seconds=60),
             'closed': 'none'}
            >>> TemporalWindowBounds(
            ...     left_inclusive=True,
            ...     window_size=timedelta(days=2),
            ...     right_inclusive=False,
            ...     offset=timedelta(minutes=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=2),
             'offset': datetime.timedelta(seconds=60),
             'closed': 'left'}
            >>> TemporalWindowBounds(
            ...     left_inclusive=False,
            ...     window_size=timedelta(days=2),
            ...     right_inclusive=True,
            ...     offset=timedelta(minutes=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=2),
             'offset': datetime.timedelta(seconds=60),
             'closed': 'right'}
        """
        if self.left_inclusive and self.right_inclusive:
            closed = "both"
        elif self.left_inclusive:
            closed = "left"
        elif self.right_inclusive:
            closed = "right"
        else:
            closed = "none"

        # set parameters for group_by rolling window
        if self.window_size < timedelta(days=0):
            period = -self.window_size
            offset = -period + self.offset
        else:
            period = self.window_size
            offset = self.offset

        return {"period": period, "offset": offset, "closed": closed}


@dataclasses.dataclass(order=True)
class ToEventWindowBounds:
    """Named tuple to represent temporal window bounds.

    Attributes:
        left_inclusive: The start of the window, inclusive.
        end_event: The string name of the event that bounds the end of this window. Operationally, this is
            interpreted as the string name of the column which contains a positive value if the row
            corresponds to the end event of this window and a zero otherwise.
        right_inclusive: The end of the window, inclusive.
        offset: The offset from the start of the window to the end of the window.

    Raises:
        ValueError: If `end_event` is an empty string.
        ValueError: If `offset` is negative.

    Example:
        >>> bounds = ToEventWindowBounds(
        ...     left_inclusive=True,
        ...     end_event="foo",
        ...     right_inclusive=False,
        ...     offset=timedelta(hours=1)
        ... )
        >>> bounds # doctest: +NORMALIZE_WHITESPACE
        ToEventWindowBounds(left_inclusive=True,
                            end_event='foo',
                            right_inclusive=False,
                            offset=datetime.timedelta(seconds=3600))
        >>> left_inclusive, end_event, right_inclusive, offset = bounds
        >>> left_inclusive
        True
        >>> end_event
        'foo'
        >>> right_inclusive
        False
        >>> offset
        datetime.timedelta(seconds=3600)
        >>> bounds = ToEventWindowBounds(
        ...     left_inclusive=True,
        ...     end_event="",
        ...     right_inclusive=False,
        ...     offset=timedelta(hours=1)
        ... )
        Traceback (most recent call last):
            ...
        ValueError: The 'end_event' must be a non-empty string.
        >>> bounds = ToEventWindowBounds(
        ...     left_inclusive=True,
        ...     end_event="_RECORD_START",
        ...     right_inclusive=False,
        ...     offset=timedelta(hours=1)
        ... ) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: It doesn't make sense to have the start of the record _RECORD_START be an end event. Did
        you mean to make that be the start event (which should result in the `end_event` parameter being
        '-_RECORD_START'?
    """

    left_inclusive: bool
    end_event: str
    right_inclusive: bool
    offset: timedelta | None = None

    def __post_init__(self):
        if self.end_event == "":
            raise ValueError("The 'end_event' must be a non-empty string.")

        if self.end_event == START_OF_RECORD_KEY:
            raise ValueError(
                f"It doesn't make sense to have the start of the record {START_OF_RECORD_KEY} be an end "
                "event. Did you mean to make that be the start event (which should result in the `end_event` "
                f"parameter being '-{START_OF_RECORD_KEY}'?"
            )
        elif self.end_event == f"-{END_OF_RECORD_KEY}":
            raise ValueError(
                f"It doesn't make sense to have the end of the record {END_OF_RECORD_KEY} be a start "
                "event. Did you mean to make that be the end event (which should result in the `end_event` "
                f"parameter being '{END_OF_RECORD_KEY}'?"
            )

        if self.offset is None:
            self.offset = timedelta(0)

    # Needed to make it accessible like a tuple.
    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))

    # Needed to make it scriptable.
    def __getitem__(self, key):
        return tuple(getattr(self, field.name) for field in dataclasses.fields(self))[key]

    @property
    def boolean_expr_bound_sum_kwargs(self) -> dict[str, str | timedelta | pl.Expr]:
        """Return the parameters for a group_by rolling operation in Polars.

        Examples:
            >>> def print_kwargs(kwargs: dict):
            ...     for key, value in kwargs.items():
            ...         print(f"{key}: {value}")
            >>> print_kwargs(ToEventWindowBounds(
            ...     left_inclusive=True, end_event="is_A", right_inclusive=False, offset=None
            ... ).boolean_expr_bound_sum_kwargs) # doctest: +NORMALIZE_WHITESPACE
            boundary_expr: [(col("is_A")) > (dyn int: 0)]
            mode: row_to_bound
            closed: left
            offset: 0:00:00
            >>> print_kwargs(ToEventWindowBounds(
            ...     left_inclusive=False, end_event="-is_B", right_inclusive=True, offset=None
            ... ).boolean_expr_bound_sum_kwargs) # doctest: +NORMALIZE_WHITESPACE
            boundary_expr: [(col("is_B")) > (dyn int: 0)]
            mode: bound_to_row
            closed: right
            offset: 0:00:00
            >>> print_kwargs(ToEventWindowBounds(
            ...     left_inclusive=False, end_event="is_B", right_inclusive=False, offset=timedelta(hours=-3)
            ... ).boolean_expr_bound_sum_kwargs) # doctest: +NORMALIZE_WHITESPACE
            boundary_expr: [(col("is_B")) > (dyn int: 0)]
            mode: row_to_bound
            closed: none
            offset: -1 day, 21:00:00
            >>> print_kwargs(ToEventWindowBounds(
            ...     left_inclusive=True,
            ...     end_event="-_RECORD_START",
            ...     right_inclusive=True,
            ...     offset=timedelta(days=2),
            ... ).boolean_expr_bound_sum_kwargs) # doctest: +NORMALIZE_WHITESPACE
            boundary_expr: [(col("timestamp")) == (col("timestamp").min().over([col("subject_id")]))]
            mode: bound_to_row
            closed: both
            offset: 2 days, 0:00:00
            >>> print_kwargs(ToEventWindowBounds(
            ...     left_inclusive=False,
            ...     end_event="_RECORD_END",
            ...     right_inclusive=True,
            ...     offset=timedelta(days=1),
            ... ).boolean_expr_bound_sum_kwargs) # doctest: +NORMALIZE_WHITESPACE
            boundary_expr: [(col("timestamp")) == (col("timestamp").max().over([col("subject_id")]))]
            mode: row_to_bound
            closed: right
            offset: 1 day, 0:00:00
        """

        if self.left_inclusive and self.right_inclusive:
            closed = "both"
        elif (not self.left_inclusive) and (not self.right_inclusive):
            closed = "none"
        elif self.left_inclusive:
            closed = "left"
        elif self.right_inclusive:
            closed = "right"

        mode = "bound_to_row" if self.end_event.startswith("-") else "row_to_bound"

        if mode == "bound_to_row":
            end_event = self.end_event[1:]
        else:
            end_event = self.end_event

        if end_event == START_OF_RECORD_KEY:
            boundary_expr = pl.col("timestamp") == pl.col("timestamp").min().over("subject_id")
        elif end_event == END_OF_RECORD_KEY:
            boundary_expr = pl.col("timestamp") == pl.col("timestamp").max().over("subject_id")
        else:
            boundary_expr = pl.col(end_event) > 0

        return {
            "boundary_expr": boundary_expr,
            "mode": mode,
            "closed": closed,
            "offset": self.offset,
        }
