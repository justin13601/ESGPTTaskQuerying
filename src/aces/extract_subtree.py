"""This module contains the functions for extracting constraint hierarchy subtrees."""

from datetime import timedelta

import polars as pl
from bigtree import Node
from loguru import logger

from .aggregate import aggregate_event_bound_window, aggregate_temporal_window
from .constraints import check_constraints
from .types import EVENT_INDEX_COLUMN, LAST_EVENT_INDEX_COLUMN


def extract_subtree(
    subtree: Node,
    subtree_anchor_realizations: pl.DataFrame,
    predicates_df: pl.DataFrame,
    subtree_root_offset: timedelta = timedelta(0),
) -> pl.DataFrame:
    """The main algorithmic recursive call to identify valid realizations of a subtree.

    This function takes in a global ``predicates_df``, a subtree of constraints, and the temporal offset that
    any realization the root timestamp of the subtree would have relative to the corresponding subtree anchor.
    It will use this information to recurse through the subtree and identify any valid realizations of this
    subtree, returning them in a dataframe keyed by the subtree anchor event timestamps and with a series of
    columns containing subtree edge start and end timestamps and contained predicate counts.

    Args:
        subtree: The subtree to extract realizations from. This is specified through a `BigTree.Node` object.
            This ``Node`` object can have zero or more children, each of which must have the following:
              - ``name``: The name of the subtree root.
              - ``constraints``: The constraints associated with the subtree root, structured as a dictionary
                from predicate column name to a tuple containing the valid (inclusive) minimum and maximum
                values the predicate counts can take on (use `None` for no constraint).
              - ``endpoint_expr``: A tuple containing the endpoint expression for the subtree root. This
                should be either a `ToEventWindowBounds` or a `TemporalWindowBounds` formatted tuple object,
                less the offset parameter, as that is something determined by the structure of the tree, not
                pre-set in the configuration.
        subtree_anchor_realizations: The dataframe containing the anchor to subtree root mapping. This
            dataframe will have the following columns:
              - ``"subject_id"``: The ID of the subject. All analyses will be performed within ``subject_id``
                groups.
              - ``subtree_anchor_timestamp``: The timestamp of all possible prospective subtree anchor
                realizations. These will all correspond to extant events (``subject_id``, ``timestamp`` pairs
                in ``predicates_df``).
        predicates_df: The dataframe containing the predicates to summarize. This dataframe will have the
            following mandatory columns:
        subtree_root_offset: The temporal offset of the subtree root relative to the subtree anchor.

    Returns:
        pl.DataFrame: The result of the subtree extraction, containing subjects who satisfy the conditions
        defined in the subtree. Timestamps for the start/end boundaries of each window specified in the
        subtree configuration, as well as predicate counts for each window, are provided.

    Examples:
        >>> from bigtree import Node
        >>> from datetime import datetime
        >>> # We'll use an example for in-hospital mortality prediction. Our root event of the tree will be
        >>> # an admission event.
        >>> root = Node("admission")
        >>> #
        >>> #### BRANCH 1 ####
        >>> # Our first branch off of admission will be checking a gap window, then our target window.
        >>> # Node 1 will represent our gap window. We say that in the 24 hours after the admission, there
        >>> # should be no discharges, deaths, or covid events.
        >>> gap_node = Node("gap") # This sets the node's name.
        >>> gap_node.endpoint_expr = (True, timedelta(days=2), True)
        >>> gap_node.constraints = {
        ...     "is_discharge": (None, 0), "is_death": (None, 0), "is_covid_dx": (None, 0)
        ... }
        >>> gap_node.parent = root
        >>> # Node 2 will start our target window and span until the next discharge or death event.
        >>> # There should be no covid events.
        >>> target_node = Node("target") # This sets the node's name.
        >>> target_node.endpoint_expr = (True, "is_discharge", True)
        >>> target_node.constraints = {"is_covid_dx": (None, 0)}
        >>> target_node.parent = gap_node
        >>> #
        >>> #### BRANCH 2 ####
        >>> # Finally, for our second branch, we will impose no constraints but track the input time range,
        >>> # which will span from the beginning of the record to 24 hours after admission.
        >>> input_end_node = Node("input_end")
        >>> input_end_node.endpoint_expr = (True, timedelta(days=1), True)
        >>> input_end_node.constraints = {}
        >>> input_end_node.parent = root
        >>> input_start_node = Node("input_start")
        >>> input_start_node.endpoint_expr = (True, "-_RECORD_START", True)
        >>> input_start_node.constraints = {}
        >>> input_start_node.parent = root
        >>> #
        >>> #### BRANCH 3 ####
        >>> # For our last branch, we will validate that the patient has sufficient historical data, asserting
        >>> # that they should have at least 1 event of any kind at least 1 year prior to the trigger event.
        >>> # This will be expressed through two windows, one spanning back a year, and the other looking
        >>> # prior to that year.
        >>> pre_node_1yr = Node("pre_node_1yr")
        >>> pre_node_1yr.endpoint_expr = (False, timedelta(days=-365), False)
        >>> pre_node_1yr.constraints = {}
        >>> pre_node_1yr.parent = root
        >>> pre_node_total = Node("pre_node_total")
        >>> pre_node_total.endpoint_expr = (False, "-_RECORD_START", False)
        >>> pre_node_total.constraints = {"*": (1, None)}
        >>> pre_node_total.parent = pre_node_1yr
        >>> #
        >>> #### PREDICATES_DF ####
        >>> # We'll have the following patient data:
        >>> #  - subject 1 will have an admission that won't count because they'll have a covid diagnosis,
        >>> #    then an admission that won't count because there will be no associated discharge.
        >>> #  - subject 2 will have an admission that won't count because they'll have too little data before
        >>> #    it, then a second admission that will count.
        >>> #  - subject 3 will have an admission that will be too short.
        >>> #
        >>> predicates_df = pl.DataFrame({
        ...     "subject_id": [
        ...         1, 1, 1, 1, 1, # Pre-event, Admission, Covid, Discharge, Admission.
        ...         2, 2, 2, 2, 2, # Pre-event-too-close, Admission, Discharge, Admission, Death & Discharge.
        ...         3, 3, 3,       # Pre-event, Admission, Death
        ...     ],
        ...     "timestamp": [
        ...         # Subject 1
        ...         datetime(year=1980, month=12, day=1,  hour=12, minute=3),  # Pre-event
        ...         datetime(year=1989, month=12, day=3,  hour=13, minute=14), # Admission
        ...         datetime(year=1989, month=12, day=5,  hour=15, minute=17), # Covid
        ...         datetime(year=1989, month=12, day=7,  hour=11, minute=4),  # Discharge
        ...         datetime(year=1989, month=12, day=23, hour=3,  minute=12), # Admission
        ...         # Subject 2
        ...         datetime(year=1983, month=12, day=1,  hour=22, minute=2),  # Pre-event-too-close
        ...         datetime(year=1983, month=12, day=2,  hour=12, minute=3),  # Admission
        ...         datetime(year=1983, month=12, day=8,  hour=13, minute=14), # Discharge
        ...         datetime(year=1989, month=12, day=6,  hour=15, minute=17), # Valid Admission
        ...         datetime(year=1989, month=12, day=10, hour=16, minute=22), # Death & Discharge
        ...         # Subject 3
        ...         datetime(year=1982, month=2,  day=13, hour=10, minute=44), # Pre-event
        ...         datetime(year=1999, month=12, day=6,  hour=15, minute=17), # Admission
        ...         datetime(year=1999, month=12, day=6,  hour=16, minute=22), # Discharge
        ...     ],
        ...     "_EVENT_INDEX": [0, 1, 2, 3, 4,   0, 1, 2, 3, 4,   0, 1, 2],
        ...     "is_admission": [0, 1, 0, 0, 1,   0, 1, 0, 1, 0,   0, 1, 0],
        ...     "is_discharge": [0, 0, 0, 1, 0,   0, 0, 1, 0, 1,   0, 0, 1],
        ...     "is_death":     [0, 0, 0, 0, 0,   0, 0, 0, 0, 1,   0, 0, 0],
        ...     "is_covid_dx":  [0, 0, 1, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0],
        ...     "_ANY_EVENT":   [1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1],
        ... })
        >>> subtreee_anchor_realizations = (
        ...     predicates_df.filter(pl.col("is_admission") > 0)
        ...     .rename({"timestamp": "subtree_anchor_timestamp"})
        ... ).select("subject_id", "subtree_anchor_timestamp")
        >>> print(subtreee_anchor_realizations)
        shape: (5, 3)
        ┌────────────┬──────────────────────────┬──────────────┐
        │ subject_id ┆ subtree_anchor_timestamp ┆ _EVENT_INDEX │
        │ ---        ┆ ---                      ┆ ---          │
        │ i64        ┆ datetime[μs]             ┆ i64          │
        ╞════════════╪══════════════════════════╪══════════════╡
        │ 1          ┆ 1989-12-03 13:14:00      ┆ 1            │
        │ 1          ┆ 1989-12-23 03:12:00      ┆ 4            │
        │ 2          ┆ 1983-12-02 12:03:00      ┆ 1            │
        │ 2          ┆ 1989-12-06 15:17:00      ┆ 3            │
        │ 3          ┆ 1999-12-06 15:17:00      ┆ 1            │
        └────────────┴──────────────────────────┴──────────────┘
        >>> out = extract_subtree(root, subtreee_anchor_realizations, predicates_df, timedelta(0))
        >>> out.select(
        ...     "subject_id",
        ...     "subtree_anchor_timestamp",
        ... )
        shape: (1, 2)
        ┌────────────┬──────────────────────────┐
        │ subject_id ┆ subtree_anchor_timestamp │
        │ ---        ┆ ---                      │
        │ i64        ┆ datetime[μs]             │
        ╞════════════╪══════════════════════════╡
        │ 2          ┆ 1989-12-06 15:17:00      │
        └────────────┴──────────────────────────┘
        >>> out.columns # doctest: +NORMALIZE_WHITESPACE
        ['subject_id',
         'target_summary',
         'subtree_anchor_timestamp',
         'gap_summary',
         'input_end_summary',
         'input_start_summary',
         'pre_node_total_summary',
         'pre_node_1yr_summary']
        >>> def print_window(name: str, do_drop_any_events: bool = True):
        ...     drop_cols = ["window_name", "subject_id", "subtree_anchor_timestamp"]
        ...     if do_drop_any_events:
        ...         drop_cols.append("_ANY_EVENT")
        ...     return (
        ...         out.select("subject_id", "subtree_anchor_timestamp", name)
        ...         .unnest(name)
        ...         .drop(*drop_cols)
        ...     )
        >>> print_window("gap_summary")
        shape: (1, 6)
        ┌─────────────────────┬─────────────────────┬──────────────┬──────────────┬──────────┬─────────────┐
        │ timestamp_at_start  ┆ timestamp_at_end    ┆ is_admission ┆ is_discharge ┆ is_death ┆ is_covid_dx │
        │ ---                 ┆ ---                 ┆ ---          ┆ ---          ┆ ---      ┆ ---         │
        │ datetime[μs]        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64      ┆ i64         │
        ╞═════════════════════╪═════════════════════╪══════════════╪══════════════╪══════════╪═════════════╡
        │ 1989-12-06 15:17:00 ┆ 1989-12-08 15:17:00 ┆ 1            ┆ 0            ┆ 0        ┆ 0           │
        └─────────────────────┴─────────────────────┴──────────────┴──────────────┴──────────┴─────────────┘
        >>> print_window("target_summary")
        shape: (1, 6)
        ┌─────────────────────┬─────────────────────┬──────────────┬──────────────┬──────────┬─────────────┐
        │ timestamp_at_start  ┆ timestamp_at_end    ┆ is_admission ┆ is_discharge ┆ is_death ┆ is_covid_dx │
        │ ---                 ┆ ---                 ┆ ---          ┆ ---          ┆ ---      ┆ ---         │
        │ datetime[μs]        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64      ┆ i64         │
        ╞═════════════════════╪═════════════════════╪══════════════╪══════════════╪══════════╪═════════════╡
        │ 1989-12-08 15:17:00 ┆ 1989-12-10 16:22:00 ┆ 0            ┆ 1            ┆ 1        ┆ 0           │
        └─────────────────────┴─────────────────────┴──────────────┴──────────────┴──────────┴─────────────┘
        >>> print_window("input_start_summary")
        shape: (1, 6)
        ┌─────────────────────┬─────────────────────┬──────────────┬──────────────┬──────────┬─────────────┐
        │ timestamp_at_start  ┆ timestamp_at_end    ┆ is_admission ┆ is_discharge ┆ is_death ┆ is_covid_dx │
        │ ---                 ┆ ---                 ┆ ---          ┆ ---          ┆ ---      ┆ ---         │
        │ datetime[μs]        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64      ┆ i64         │
        ╞═════════════════════╪═════════════════════╪══════════════╪══════════════╪══════════╪═════════════╡
        │ 1983-12-01 22:02:00 ┆ 1989-12-06 15:17:00 ┆ 2            ┆ 1            ┆ 0        ┆ 0           │
        └─────────────────────┴─────────────────────┴──────────────┴──────────────┴──────────┴─────────────┘
        >>> print_window("input_end_summary")
        shape: (1, 6)
        ┌─────────────────────┬─────────────────────┬──────────────┬──────────────┬──────────┬─────────────┐
        │ timestamp_at_start  ┆ timestamp_at_end    ┆ is_admission ┆ is_discharge ┆ is_death ┆ is_covid_dx │
        │ ---                 ┆ ---                 ┆ ---          ┆ ---          ┆ ---      ┆ ---         │
        │ datetime[μs]        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64      ┆ i64         │
        ╞═════════════════════╪═════════════════════╪══════════════╪══════════════╪══════════╪═════════════╡
        │ 1989-12-06 15:17:00 ┆ 1989-12-07 15:17:00 ┆ 1            ┆ 0            ┆ 0        ┆ 0           │
        └─────────────────────┴─────────────────────┴──────────────┴──────────────┴──────────┴─────────────┘
        >>> print_window("pre_node_1yr_summary")
        shape: (1, 6)
        ┌─────────────────────┬─────────────────────┬──────────────┬──────────────┬──────────┬─────────────┐
        │ timestamp_at_start  ┆ timestamp_at_end    ┆ is_admission ┆ is_discharge ┆ is_death ┆ is_covid_dx │
        │ ---                 ┆ ---                 ┆ ---          ┆ ---          ┆ ---      ┆ ---         │
        │ datetime[μs]        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64      ┆ i64         │
        ╞═════════════════════╪═════════════════════╪══════════════╪══════════════╪══════════╪═════════════╡
        │ 1989-12-06 15:17:00 ┆ 1988-12-06 15:17:00 ┆ 0            ┆ 0            ┆ 0        ┆ 0           │
        └─────────────────────┴─────────────────────┴──────────────┴──────────────┴──────────┴─────────────┘
        >>> print_window("pre_node_total_summary")
        shape: (1, 6)
        ┌─────────────────────┬─────────────────────┬──────────────┬──────────────┬──────────┬─────────────┐
        │ timestamp_at_start  ┆ timestamp_at_end    ┆ is_admission ┆ is_discharge ┆ is_death ┆ is_covid_dx │
        │ ---                 ┆ ---                 ┆ ---          ┆ ---          ┆ ---      ┆ ---         │
        │ datetime[μs]        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64      ┆ i64         │
        ╞═════════════════════╪═════════════════════╪══════════════╪══════════════╪══════════╪═════════════╡
        │ 1983-12-01 22:02:00 ┆ 1988-12-06 15:17:00 ┆ 1            ┆ 1            ┆ 0        ┆ 0           │
        └─────────────────────┴─────────────────────┴──────────────┴──────────────┴──────────┴─────────────┘
    """
    recursive_results = []
    predicate_cols = [
        c for c in predicates_df.columns if c not in {"subject_id", "timestamp", EVENT_INDEX_COLUMN}
    ]

    if not subtree.children:
        return subtree_anchor_realizations

    for child in subtree.children:
        logger.info(f"Summarizing subtree rooted at '{child.name}'...")

        # Step 1: Summarize the window from the subtree.root to child
        endpoint_expr = child.endpoint_expr
        if type(endpoint_expr) is tuple:
            endpoint_expr = endpoint_expr + (subtree_root_offset,)
        else:
            endpoint_expr.offset += subtree_root_offset

        match endpoint_expr[1]:
            case timedelta():
                child_root_offset = subtree_root_offset + endpoint_expr[1]
                window_summary_df = (
                    aggregate_temporal_window(predicates_df, endpoint_expr)
                    .with_columns(
                        pl.col("timestamp").alias("subtree_anchor_timestamp"),
                        pl.col("timestamp").alias("child_anchor_timestamp"),
                    )
                    .drop("timestamp")
                )
            case str():
                # In an event bound case, the child root will be a proper extant event, so it will be the
                # anchor as well, and thus the child root offset should be zero.
                child_root_offset = timedelta(days=0)
                window_summary_df = (
                    aggregate_event_bound_window(predicates_df, endpoint_expr)
                    .with_columns(
                        pl.col("timestamp").alias("subtree_anchor_timestamp"),
                        pl.col("timestamp_at_end").alias("child_anchor_timestamp"),
                    )
                    .drop("timestamp")
                )
            case _:
                raise ValueError(f"Invalid endpoint expression: '{endpoint_expr}'")

        # Step 2: Filter to valid subtree anchors
        window_summary_df = window_summary_df.join(
            subtree_anchor_realizations, on=["subject_id", "subtree_anchor_timestamp"], how="inner"
        )

        # Step 3: Filter to where constraints are valid
        window_summary_df = check_constraints(child.constraints, window_summary_df)

        # Step 4: Produce child anchor realizations
        child_anchor_realizations = window_summary_df.select(
            "subject_id",
            pl.col("child_anchor_timestamp").alias("subtree_anchor_timestamp"),
        )

        # Step 5: Recurse
        recursive_result = extract_subtree(
            child,
            child_anchor_realizations,
            predicates_df,
            child_root_offset,
        )

        # Step 6: Join summaries and timestamps
        # Step 6.1: Convert recursive_result up to subtree anchor space.
        recursive_result = (
            recursive_result.rename({"subtree_anchor_timestamp": "child_anchor_timestamp"})
            .join(
                window_summary_df.select("subject_id", "subtree_anchor_timestamp", "child_anchor_timestamp"),
                on=["subject_id", "child_anchor_timestamp"],
                how="left",
            )
            .drop("child_anchor_timestamp")
        )

        # Step 6.2: Summarize the observed window statistics and timestamps for eventual return.
        for_return = window_summary_df.select(
            "subject_id",
            "subtree_anchor_timestamp",
            pl.struct(
                pl.lit(child.name).alias("window_name"),
                "timestamp_at_start",
                "timestamp_at_end",
                pl.col(EVENT_INDEX_COLUMN).alias(LAST_EVENT_INDEX_COLUMN),
                *predicate_cols,
            ).alias(f"{child.name}_summary"),
        )

        recursive_results.append(
            recursive_result.join(for_return, on=["subject_id", "subtree_anchor_timestamp"], how="left")
        )

    # Step 7: Join children recursive results where all children find a valid realization
    all_children = recursive_results[0]
    for df in recursive_results[1:]:
        all_children = all_children.join(df, on=["subject_id", "subtree_anchor_timestamp"], how="inner")

    return all_children
