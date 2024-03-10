# -*- coding: utf-8 -*-
"""## Preparation Functions"""

import re
from typing import Any
import ruamel.yaml
import numpy as np
from bigtree import Node
from bigtree import print_tree
from bigtree import preorder_iter
from datetime import timedelta

import polars as pl

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)


class DotAccessibleDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotAccessibleDict(value)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )


def load_config(config_path: str) -> DotAccessibleDict:
    yaml = ruamel.yaml.YAML()
    with open(config_path, "r") as file:
        config_dict = yaml.load(file)
    return DotAccessibleDict(config_dict)


def parse_timedelta(time_str):
    if not time_str:
        return timedelta(days=0)

    units = {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
    pattern = r"(\d+)\s*(seconds|minutes|hours|days)"
    matches = re.findall(pattern, time_str.lower())
    for value, unit in matches:
        units[unit] = int(value)

    return timedelta(
        days=units["days"],
        hours=units["hours"],
        minutes=units["minutes"],
        seconds=units["seconds"],
    )


def has_event_type(type_str: str) -> pl.Expr:
    has_event_type = pl.col("event_type").cast(pl.Utf8).str.contains(type_str)
    # has_event_type = event_types.str.contains(type_str)
    return has_event_type


def build_tree_from_config(cfg):
    nodes = {}
    windows = [x for x, y in cfg.windows.items()]

    for window_name in windows:
        node = Node(window_name)
        window_info = cfg.windows[window_name]

        if window_info.duration:
            if '-' in window_info.duration:
                end_event = -parse_timedelta(window_info.duration)
            else:
                end_event = parse_timedelta(window_info.duration)
        else:
            end_event = f"is_{window_info.end}"

        st_inclusive = False
        end_inclusive = True
        if "st_inclusive" in window_info:
            st_inclusive = window_info.st_inclusive
        if "end_inclusive" in window_info:
            end_inclusive = window_info.end_inclusive

        offset = parse_timedelta(window_info.offset)
        node.endpoint_expr = (st_inclusive, end_event, end_inclusive, offset)

        constraints = {}
        if window_info.excludes:
            for each_exclusion in window_info.excludes:
                if each_exclusion["predicate"]:
                    constraints[f"is_{each_exclusion['predicate']}"] = (None, 0)

        if window_info.includes:
            for each_inclusion in window_info.includes:
                if each_inclusion["predicate"]:
                    constraints[f"is_{each_inclusion['predicate']}"] = (
                        int(each_inclusion["min"])
                        if "min" in each_inclusion and each_inclusion["min"] is not None
                        else None,
                        int(each_inclusion["max"])
                        if "max" in each_inclusion and each_inclusion["max"] is not None
                        else None,
                    )
        node.constraints = constraints

        if window_info.start:
            node_root = next(
                (substring for substring in windows if substring in window_info.start),
                None,
            )
        elif window_info.end:
            node_root = next(
                (substring for substring in windows if substring in window_info.end),
                None,
            )

        if not node_root:
            nodes[window_name] = node
        elif node_root not in nodes:
            windows.append(window_name)
        else:
            node.parent = nodes[node_root]
            nodes[window_name] = node
    
    root = next(iter(nodes.values())).root

    # if event_bound window follows temporal window, the st_inclusive parameter is set to True
    for each_node in preorder_iter(root):
        if each_node.parent and isinstance(each_node.parent.endpoint_expr[1], timedelta) and isinstance(each_node.endpoint_expr[1], str):
            each_node.endpoint_expr = (True, each_node.endpoint_expr[1], each_node.endpoint_expr[2], each_node.endpoint_expr[3])
    return root


def generate_predicate_columns(cfg, ESD):
    boolean_cols = []
    count_cols = []
    for predicate_name, predicate_info in cfg.predicates.items():
        if predicate_info.system == "boolean":
            boolean_cols.append(f"is_{predicate_name}")
        elif predicate_info.system == "count":
            count_cols.append(f"is_{predicate_name}")
        else:
            raise ValueError(
                f"Invalid predicate system {predicate_info.system} for {predicate_name}."
            )
        if "value" in predicate_info:
            if isinstance(predicate_info["value"], list):
                ESD = ESD.with_columns(
                    pl.when(
                        (
                            pl.col(predicate_info.column)
                            >= (
                                float(predicate_info["value"][0]["min"] or -np.inf)
                                if "min" in predicate_info["value"][0]
                                else float(-np.inf)
                            )
                        )
                        & (
                            pl.col(predicate_info.column)
                            <= (
                                float(predicate_info["value"][0]["max"] or np.inf)
                                if "max" in predicate_info["value"][0]
                                else float(np.inf)
                            )
                        )
                    )
                    .then(1)
                    .otherwise(0)
                    .alias(f"is_{predicate_name}")
                    .cast(pl.Int32)
                )
                print(f"Added predicate column is_{predicate_name}.")
            else:
                if predicate_info.column == "event_type":
                    ESD = ESD.with_columns(
                        has_event_type(predicate_info["value"])
                        .alias(f"is_{predicate_name}")
                        .cast(pl.Int32)
                    )
                else:
                    ESD = ESD.with_columns(
                        pl.when(
                            pl.col(predicate_info.column) == predicate_info["value"]
                        )
                        .then(1)
                        .otherwise(0)
                        .alias(f"is_{predicate_name}")
                        .cast(pl.Int32)
                    )
                print(f"Added predicate column is_{predicate_name}.")
        elif "type" in predicate_info:
            if predicate_info.type == "ANY":
                any_expr = pl.col(f"is_{predicate_info.predicates[0]}")
                for predicate in predicate_info.predicates[1:]:
                    any_expr = any_expr | pl.col(f"is_{predicate}")
                ESD = ESD.with_columns(
                    any_expr.alias(f"is_{predicate_name}")
                )
                print(
                    f"Added predicate column is_{predicate_name}."
                )
            elif predicate_info.type == "ALL":
                all_expr = pl.col(f"is_{predicate_info.predicates[0]}")
                for predicate in predicate_info.predicates[1:]:
                    all_expr = all_expr & pl.col(f"is_{predicate}")
                ESD = ESD.with_columns(
                    all_expr.alias(f"is_{predicate_name}")
                )
                print(
                    f"Added predicate column is_{predicate_name}."
                )
            else:
                raise ValueError(f"Invalid predicate type {predicate_info.type} for {predicate_name}.")
        else:
            raise ValueError(
                f"Invalid predicate specification for {predicate_name}."
            )
            
    ESD = ESD.with_columns(
        pl.when(pl.col("event_type").is_not_null()).then(1).otherwise(0).alias("is_any")
    )

    ESD = ESD.groupby(["subject_id", "timestamp"]).agg(
        *[
            pl.col(c).sum().alias(f"{c}_count").cast(pl.Int32)
            for c in ESD.columns
            if c.startswith("is_")
        ],
        *[
            pl.col(c).max().alias(f"{c}_boolean").cast(pl.Int32)
            for c in ESD.columns
            if c.startswith("is_")
        ],
    )
    
    ESD = ESD.select(
        'subject_id',
        'timestamp',
        *[pl.col(c).alias(c.replace("_boolean", "")) for c in ESD.columns if c.replace("_boolean", "") in boolean_cols],
        *[pl.col(c).alias(c.replace("_count", "")) for c in ESD.columns if c.replace("_count", "") in count_cols],
        pl.col('is_any_boolean').alias('is_any'),
    )

    ESD = ESD.sort(by=['subject_id', 'timestamp'])

    return ESD


"""## Base Functions"""


def summarize_temporal_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    predicate_cols: "list[str]",
    endpoint_expr: Any,
    anchor_to_subtree_root_by_subtree_anchor:  pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    st_inclusive, window_size, end_inclusive, offset = endpoint_expr
    if not offset:
        offset = timedelta(days=0)

    if st_inclusive and end_inclusive:
        closed = "both"
    elif st_inclusive:
        closed = "left"
    elif end_inclusive:
        closed = "right"
    else:
        closed = "none"

    if window_size < timedelta(days=0):
        period = -window_size
        offset = -period + offset
    else:
        period = window_size
        offset = timedelta(days=0) + offset

    result = (
        predicates_df.rolling(
            index_column="timestamp",
            by="subject_id",
            closed=closed,
            period=period,
            offset=offset,
        )
        .agg([pl.col(c).sum().alias(c) for c in predicate_cols])
        .sort(by=["subject_id", "timestamp"])
    )

    filtered_result = result.join(
            anchor_to_subtree_root_by_subtree_anchor,
            on=["subject_id", "timestamp"],
            suffix="_summary",
        ).with_columns(pl.col("timestamp").alias("timestamp_at_anchor"))
        
    filtered_result = filtered_result.select(
            "subject_id",
            "timestamp",
            "timestamp_at_anchor",
            *[
                pl.col(c) for c in filtered_result.columns if c not in ["subject_id", "timestamp", "timestamp_at_anchor"]
            ],
        )

    return filtered_result


def summarize_event_bound_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    predicate_cols: list[str],
    endpoint_expr: tuple[bool, timedelta, bool, timedelta] | tuple[bool, str, bool, timedelta],
    anchor_to_subtree_root_by_subtree_anchor: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Summarizes the event-bound window based on the given predicates and anchor-to-subtree-root mapping.

    Args:
        predicates_df: The dataframe containing the predicates.
        predicate_cols: The list of predicate columns.
        endpoint_expr: The expression defining the event-bound window endpoints.
        anchor_to_subtree_root_by_subtree_anchor: The mapping of anchor to subtree root.

    Returns:
        The summarized dataframe.
    """
    st_inclusive, end_event, end_inclusive, offset = endpoint_expr
    if not offset:
        offset = timedelta(days=0)

    cumsum_predicates_df = predicates_df.with_columns(
        *[pl.col(c).cum_sum().over(pl.col("subject_id")).alias(f"{c}_cumsum") for c in predicate_cols],
    )

    cnts_at_anchor = (
        anchor_to_subtree_root_by_subtree_anchor.select("subject_id", "timestamp")
        .join(cumsum_predicates_df, on=["subject_id", "timestamp"], how="left")
        .select(
            "subject_id",
            "timestamp",
            pl.col("timestamp").alias("timestamp_at_anchor"),
            *[pl.col(c).alias(f"{c}_at_anchor") for c in predicate_cols],
            *[pl.col(f"{c}_cumsum").alias(f"{c}_cumsum_at_anchor") for c in predicate_cols],
        )
    )

    cumsum_predicates_df = cumsum_predicates_df.select(
        "subject_id",
        "timestamp",
        *[pl.col(f"{c}") for c in predicate_cols],
        *[pl.col(f"{c}_cumsum") for c in predicate_cols],
    )   

    cumsum_predicates_df = cumsum_predicates_df.join(
        cnts_at_anchor, on=["subject_id", "timestamp"], how="left"
    ).with_columns(
        pl.col("timestamp_at_anchor").forward_fill().over("subject_id"),
        *[pl.col(f"{c}_at_anchor").forward_fill().over("subject_id") for c in predicate_cols],
        *[pl.col(f"{c}_cumsum_at_anchor").forward_fill().over("subject_id") for c in predicate_cols],
    )
    
    cumsum_anchor_child = cumsum_predicates_df.with_columns(
        "subject_id",
        "timestamp",
        *[
            (pl.col(f"{c}_cumsum") - pl.col(f"{c}_cumsum_at_anchor")).alias(f"{c}_final")
            for c in predicate_cols
        ],
    )

    if st_inclusive:
        cumsum_anchor_child = cumsum_anchor_child.with_columns(
            "subject_id",
            "timestamp",
            *[(pl.col(f"{c}_final") + pl.col(f"{c}_at_anchor")) for c in predicate_cols],
        )
    if not end_inclusive:
        cumsum_anchor_child = cumsum_anchor_child.with_columns(
            "subject_id",
            "timestamp",
            *[(pl.col(f"{c}_final") - pl.col(f"{c}")) for c in predicate_cols],
        )

    at_child_anchor = cumsum_anchor_child.select(
        "subject_id",
        "timestamp",
        "timestamp_at_anchor",
        *[pl.col(f"{c}_final").alias(c) for c in predicate_cols],
    )

    at_child_anchor = at_child_anchor.with_columns(
        *[pl.when(pl.col(c) < 0).then(0).otherwise(pl.col(c)).alias(c) for c in predicate_cols]
    )

    filtered_by_end_event_at_child_anchor = (
        predicates_df.filter(pl.col(end_event) >= 1)
        .join(at_child_anchor, on=["subject_id", "timestamp"], how="inner")
        .select(
            "subject_id",
            "timestamp",
            "timestamp_at_anchor",
            *[pl.col(f"{c}_right").alias(c) for c in predicate_cols],
        )
    )
    
    filtered_result = filtered_by_end_event_at_child_anchor.join(
        anchor_to_subtree_root_by_subtree_anchor,
        left_on=["subject_id", "timestamp_at_anchor"],
        right_on=["subject_id", "timestamp"],
        suffix="_summary",
    )

    return filtered_result


def summarize_window(
    child, anchor_to_subtree_root_by_subtree_anchor, predicates_df, predicate_cols
):
    """
    Args:
        predicates_df:
        anchor_to_subtree_root_by_subtree_anchor: pl.DataFrame,

    Returns:
            # subtree_root_to_child_summary_by_child_anchor has a row for every possible realization
            # of the anchor of the subtree rooted by _child_ (not the input subtree)
            # with the counts occurring between subtree_root and the child

        case event bound:
        1. Do our global cumulative sum
        2. Join anchor_to_subtree_root_by_subtree_anchor into global sum dataframe as "since start" columns
            (will be null wherever the anchor_to_subtree col is not present and otherwise have the anchor_to_subtree values)
        3. Forward fill "since start" columns per subject.
        4. On possible end events, subtract global cumulative sums from "since start" columns.
        5. Filter to rows corresponding to possible end events that were preceeded by a possible start event
            (e.g., didn't have a null in the subtract in step 4)

        At end of this process, we have rows corresponding to possible end events (anchors for child)
        with counts of predicates that have occurred since the subtree root,
        already having handled subtracting the anchor to subtree root component.
    """
    match child.endpoint_expr[1]:
        case timedelta():
            subtree_anchor_to_child_root_by_child_anchor = summarize_temporal_window(
                predicates_df, 
                predicate_cols, 
                child.endpoint_expr,
                anchor_to_subtree_root_by_subtree_anchor,
            )

        case str():
            subtree_anchor_to_child_root_by_child_anchor = summarize_event_bound_window(
                predicates_df,
                predicate_cols,
                child.endpoint_expr,
                anchor_to_subtree_root_by_subtree_anchor,
            )

    subtree_root_to_child_root_by_child_anchor = subtree_anchor_to_child_root_by_child_anchor.select(
        "subject_id",
        "timestamp",
        "timestamp_at_anchor",
        *[pl.col(c) - pl.col(f"{c}_summary") for c in predicate_cols],
    )

    return subtree_root_to_child_root_by_child_anchor


def check_constraints(window_constraints, summary_df):
    """

    Args:
        window_constraints: constraints on counts of predicates that must
            be satsified.
        summary_df: A dataframe containing a row for every possible realization

    Return: A filtered dataframe containing only the rows that satisfy the constraints.

    # Temporal constraint
    # subj_id, ts, pred_A, pred_B
    # 1,       23, 15,     32,    # Means that in the temporal window starting at ts = 23, pred_A occurred 15 times, pred_B 32 times, etc.
    # ...

    # Event bound window:
    # subj_id, ts, pred_A, pred_B
    # 1,       23, 15,     32     # Means that in the event bound window that starts as of node_col_offset after ts and ends at some unspecified next event, A occurs 15 times, etc.
    # ...

    # OR:

    ########## if ts == 24?

    # Event bound window:
    # subj_id, ts, pred_A, pred_B
    # 1,       23, None,   None
    # ...
    # 1,       47, 15,     32     # Means that in the evnet bound window ending at real event that occurs at ts = 47, A occurs 15 times, etc. (same event as in line 31)
    # ...
    """
    valid_exprs = []
    for col, (cnt_ge, cnt_le) in window_constraints.items():
        if cnt_ge is None and cnt_le is None:
            raise ValueError(f"Empty constraint for {col}!")

        if col == "*":
            col = "__ALL_EVENTS"

        if cnt_ge is not None:
            valid_exprs.append(pl.col(col) >= cnt_ge)
        if cnt_le is not None:
            valid_exprs.append(pl.col(col) <= cnt_le)

    if not valid_exprs:
        valid_exprs.append(pl.lit(True))

    summary_df_shape = summary_df.shape[0]
    for condition in valid_exprs:
        dropped = summary_df.filter(~condition)
        summary_df = summary_df.filter(condition)
        if summary_df.shape[0] < summary_df_shape:
            print(f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to constraint: {condition}.")
            summary_df_shape = summary_df.shape[0]

    return summary_df


"""## Recursive Loop"""


def query_subtree(
    subtree,
    anchor_to_subtree_root_by_subtree_anchor: pl.DataFrame | None,
    predicates_df: pl.DataFrame,
    anchor_offset: float,
):
    """
    Args:
        subtree:
          Subtree object.

        ########
        anchor_to_subtree_root_by_subtree_anchor: A dataframe with a row for each possible
          realization of the anchoring node for this subtree containing the
          counts of predicates that have occurred from the anchoring node to
          the subtree root for that realization of `subtree.root`.

          # First iteration:
          subj_id, ts,  is_admission, is_discharge, pred_C
          1,       1,   0,            0,            0
          1,       10,  0,            0,            0
          1,       26,  0,            0,            0
          1,       33,  0,            0,            0
          1,       81,  0,            0,            0
          1,       88,  0,            0,            0
          1,       89,  0,            0,            0
          1,       122, 0,            0,            0

          # Example:
          (admission_event)
          |
          24h
          |
          (node_A)
          |
          to_discharge
          |
          (node_B)
          |
          36h
          |
          (node_C)

          predicates_df: A dataframe containing a row for every event for every
          subject with the following columns:

          - A column ``subject_id`` which contains the subject ID.
          - A column ``timestamp`` which contains the timestamp at which the
            event contained in any given row occurred.
          - A set of "predicate" columns that contain counts of the
            number of times a given predicate is satisfied in the
            event contained in any given row.

          `predicates_df` (can be all bools or all counts ; this is bools but swap T for 1 and F for 0 and it is in counts format)
          subj_id, ts,  is_admission, is_discharge, pred_C
          1,       1,   1,            0,            1
          1,       10,  0,            0,            1
          1,       26,  0,            0,            0
          1,       33,  0,            1,            1
          1,       81,  1,            0,            0
          1,       88,  0,            0,            1
          1,       89,  0,            0,            0
          1,       122, 0,            1,            1

          On the subtree rooted at (node_A), the anchor node is (admission_event), and
          `anchor_to_subtree_root_by_subtree_anchor` would be:

          subj_id, ts, is_admission, is_discharge, pred_C
          1,       1,  1,            0,            2
          1,       81, 1,            0,            1

        anchor_offset: The sum of all timedelta edges between subtree_root and
          the anchor node for this subtree.

        0 for first iteration.

      Returns: A dataframe with a row corresponding to the anchor event for each
        possible valid realization of this subtree (and all its children)
        containing the timestamp values realizing the nodes in this subtree in
        that realization.

      subj_id, ts,  valid_start, valid_end
      1,       1,   ts,            ts,
      1,       10,  ts,            ts,
      1,       26,  ts,            ts,
      1,       33,  ts,            ts,
      1,       81,  ts,            ts,
      1,       88,  ts,            ts,
      1,       89,  ts,            ts,
      1,       122, ts,            ts,
    """
    predicate_cols = [col for col in predicates_df.columns if col.startswith("is_")]

    recursive_results = []

    for child in subtree.children:
        print("\n")
        print(f"Querying subtree rooted at {child.name}...")
        # Added to reset anchor_offset and anchor_to_subtree_root_by_subtree_anchor for diverging subtrees
        # if len(child.parent.children) > 1:
        #     anchor_offset = timedelta(hours=0)
        #     anchor_to_subtree_root_by_subtree_anchor = (
        #         predicates_df.filter(predicates_df[child.parent.endpoint_expr[1]] == 1)
        #             .select('subject_id', 'timestamp', *[pl.col(c) for c in predicate_cols])
        #             .with_columns('subject_id', 'timestamp', *[pl.lit(0).alias(c) for c in predicate_cols])
        #     )

        # Step 1: Summarize the window from the subtree.root to child.

        subtree_root_to_child_root_by_child_anchor = summarize_window(
            child,
            anchor_to_subtree_root_by_subtree_anchor,
            predicates_df,
            predicate_cols,
        )

        # display(subtree_root_to_child_root_by_child_anchor)

        # subtree_root_to_child_root_by_child_anchor... has a row for every possible realization
        # of the anchor of the subtree rooted by _child_ (not the input subtree)
        # with the counts occurring between subtree_root and the child

        # Step 2: Filter to where constraints are valid
        subtree_root_to_child_root_by_child_anchor = check_constraints(
            child.constraints, subtree_root_to_child_root_by_child_anchor
        )

        # Step 3: Update parameters for recursive step:
        match child.endpoint_expr[1]:
            case timedelta():
                anchor_offset_branch = anchor_offset + child.endpoint_expr[1] + child.endpoint_expr[3]
                joined = anchor_to_subtree_root_by_subtree_anchor.join(
                    subtree_root_to_child_root_by_child_anchor,
                    on=["subject_id", "timestamp"],
                    suffix="_summary",
                )
                anchor_to_subtree_root_by_subtree_anchor_branch = joined.select(
                    "subject_id",
                    "timestamp",
                    *[pl.col(c) + pl.col(f"{c}_summary") for c in predicate_cols],
                )

                # anchor_to_subtree_root_by_subtree_anchor_branch_shape = anchor_to_subtree_root_by_subtree_anchor_branch.shape[0]
                # for condition in valid_windows:
                #     dropped = anchor_to_subtree_root_by_subtree_anchor_branch.filter(~condition)
                #     anchor_to_subtree_root_by_subtree_anchor_branch = anchor_to_subtree_root_by_subtree_anchor_branch.filter(condition)
                #     if anchor_to_subtree_root_by_subtree_anchor_branch.shape[0] < anchor_to_subtree_root_by_subtree_anchor_branch_shape:
                #         print(f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to constraint: {condition}.")
                #         anchor_to_subtree_root_by_subtree_anchor_branch_shape = anchor_to_subtree_root_by_subtree_anchor_branch.shape[0]

                # anchor_to_subtree_root_by_subtree_anchor_branch = (
                #     anchor_to_subtree_root_by_subtree_anchor_branch.filter(pl.all_horizontal(valid_windows))
                # )

            case str():
                anchor_offset_branch = timedelta(days=0) + child.endpoint_expr[3]
                # anchor_to_subtree_root_by_subtree_anchor = anchor_to_subtree_root_by_subtree_anchor.with_columns(
                #     "subject_id",
                #     "timestamp",
                #     *[pl.lit(0).alias(c) for c in predicate_cols],
                # )
                joined = anchor_to_subtree_root_by_subtree_anchor.join(
                    subtree_root_to_child_root_by_child_anchor,
                    left_on=["subject_id", "timestamp"],
                    right_on=["subject_id", "timestamp_at_anchor"],
                    suffix="_summary",
                )
                anchor_to_subtree_root_by_subtree_anchor_branch = joined.select(
                    "subject_id",
                    "timestamp_summary",
                    *[pl.col(c) + pl.col(f"{c}_summary") for c in predicate_cols],
                ).rename({"timestamp_summary": "timestamp"})
                        
                # anchor_to_subtree_root_by_subtree_anchor_branch_shape = anchor_to_subtree_root_by_subtree_anchor_branch.shape[0]
                # for condition in valid_windows:
                #     dropped = anchor_to_subtree_root_by_subtree_anchor_branch.filter(~condition)
                #     anchor_to_subtree_root_by_subtree_anchor_branch = anchor_to_subtree_root_by_subtree_anchor_branch.filter(condition)
                #     if anchor_to_subtree_root_by_subtree_anchor_branch.shape[0] < anchor_to_subtree_root_by_subtree_anchor_branch_shape:
                #         print(f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to constraint: {condition}.")
                #         anchor_to_subtree_root_by_subtree_anchor_branch_shape = anchor_to_subtree_root_by_subtree_anchor_branch.shape[0]

                # anchor_to_subtree_root_by_subtree_anchor_branch = (
                #     anchor_to_subtree_root_by_subtree_anchor_branch.filter(pl.all_horizontal(valid_windows))
                # )

                anchor_to_subtree_root_by_subtree_anchor_branch = anchor_to_subtree_root_by_subtree_anchor_branch.with_columns(
                    "subject_id",
                    "timestamp",
                    *[pl.lit(0).alias(c) for c in predicate_cols],
                )
                

        # Step 4: Recurse
        recursive_result = query_subtree(
            child,
            anchor_to_subtree_root_by_subtree_anchor_branch,
            predicates_df,
            anchor_offset_branch,
        )

        match child.endpoint_expr[1]:
            case timedelta():
                recursive_result = recursive_result.with_columns(
                    (pl.col("timestamp") + anchor_offset_branch).alias(
                        f"{child.name}/timestamp"
                    )
                )
            case str():
                recursive_result = recursive_result.with_columns(
                    pl.col("timestamp").alias(f"{child.name}/timestamp")
                )

        # Step 5: Push results back to subtree anchor.
        subtree_root_to_child_root_by_child_anchor = (
            subtree_root_to_child_root_by_child_anchor.with_columns(
                pl.struct([pl.col(c).alias(c) for c in predicate_cols]).alias(
                    f"{child.name}/window_summary"
                )
            )
        )

        match child.endpoint_expr[1]:
            case timedelta():
                final_recursive_result = recursive_result.join(
                    subtree_root_to_child_root_by_child_anchor.select(
                        "subject_id", "timestamp", f"{child.name}/window_summary"
                    ),
                    on=["subject_id", "timestamp"],
                )
            case str():
                # Need a dataframe with one col with a "True" in the possible realizations of
                # subtree anchor and another col with a "True" in the possible valid corresponding realizations
                # of the child node.
                # Make this with anchor_to_subtree_root_by_subtree_anchor
                #   (contains rows corresponding to possible start events).
                # and recursive_result (contains rows corresponding to possible end events).
                final_recursive_result = (
                    recursive_result.join(
                        subtree_root_to_child_root_by_child_anchor.select(
                            "subject_id",
                            "timestamp",
                            "timestamp_at_anchor",
                            f"{child.name}/window_summary",
                        ),
                        on=["subject_id", "timestamp"],
                    )
                    .drop("timestamp")
                    .rename({"timestamp_at_anchor": "timestamp"})
                )

        recursive_results.append(final_recursive_result)

    # Step 6: Join children recursive results where all children find a valid realization
    if not recursive_results:
        all_children = anchor_to_subtree_root_by_subtree_anchor.select(
            "subject_id", "timestamp"
        )
    else:
        all_children = recursive_results[0]
        for df in recursive_results[1:]:
            all_children = all_children.join(
                df, on=["subject_id", "timestamp"], how="inner"
            )

    # Step 7: return
    return all_children


"""# End-to-end Run"""


def query_task(cfg_path, ESD):
    if ESD["timestamp"].dtype != pl.Datetime:
        ESD = ESD.with_columns(pl.col('timestamp').str.strptime(pl.Datetime, format='%m/%d/%Y %H:%M').cast(pl.Datetime))

    if ESD.shape[0] == 0:
        raise ValueError("Empty ESD!")
    if "timestamp" not in ESD.columns:
        raise ValueError("ESD does not have timestamp column!")
    if "subject_id" not in ESD.columns:
        raise ValueError("ESD does not have subject_id column!")

    print("Loading config...\n")
    cfg = load_config(cfg_path)

    print("Generating predicate columns...\n")
    try:
        ESD = generate_predicate_columns(cfg, ESD)
    except Exception as e:
        print(repr(e))
        raise ValueError(
            "Error generating predicate columns from configuration file! Check to make sure the format of the configuration file is valid."
        ) from e

    print("\nBuilding tree...")
    tree = build_tree_from_config(cfg)
    print_tree(tree, style="const_bold")
    print("\n")

    predicate_cols = [col for col in ESD.columns if col.startswith("is_")]

    valid_trigger_exprs = [
        (ESD[f"is_{x['predicate']}"] == 1) for x in cfg.windows.trigger.includes
    ]

    anchor_to_subtree_root_by_subtree_anchor = ESD.clone()
    anchor_to_subtree_root_by_subtree_anchor_shape = anchor_to_subtree_root_by_subtree_anchor.shape[0]
    for i, condition in enumerate(valid_trigger_exprs):
        dropped = anchor_to_subtree_root_by_subtree_anchor.filter(~condition)
        anchor_to_subtree_root_by_subtree_anchor = anchor_to_subtree_root_by_subtree_anchor.filter(condition)
        if anchor_to_subtree_root_by_subtree_anchor.shape[0] < anchor_to_subtree_root_by_subtree_anchor_shape:
            print(f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to trigger condition: {cfg.windows.trigger.includes[i]}.")
            anchor_to_subtree_root_by_subtree_anchor_shape = anchor_to_subtree_root_by_subtree_anchor.shape[0]

    anchor_to_subtree_root_by_subtree_anchor = (anchor_to_subtree_root_by_subtree_anchor
        .select("subject_id", "timestamp", *[pl.col(c) for c in predicate_cols])
        .with_columns(
            "subject_id", "timestamp", *[pl.lit(0).alias(c) for c in predicate_cols]
        )
    )

    print("\n")
    print("Querying...")
    result = query_subtree(
        subtree=tree,
        anchor_to_subtree_root_by_subtree_anchor=anchor_to_subtree_root_by_subtree_anchor,
        predicates_df=ESD,
        anchor_offset=timedelta(hours=0),
    )
    print("\n")
    print("Done.\n")

    output_order = [node for node in preorder_iter(tree)]

    result = result.select(
        "subject_id",
        "timestamp",
        *[f"{c.name}/timestamp" for c in output_order[1:]],
        *[f"{c.name}/window_summary" for c in output_order[1:]],
    ).rename({"timestamp": f"{tree.name}/timestamp"})

    label_window = None
    for window in cfg.windows:
        if "label" in cfg.windows[window]:
            label_window = window
            break

    if label_window:
        label = cfg.windows[label_window].label
        result = result.with_columns(
            pl.col(f"{label_window}/window_summary")
            .struct.field(f"is_{label}")
            .alias("label")
        )
    return result


# config_path = '/content/drive/MyDrive/Colab Notebooks/SickKids/ESGPT/config.yaml'
# data_path = '/content/drive/MyDrive/Colab Notebooks/SickKids/ESGPT/data.csv'

# predicates_df = pl.read_csv(data_path)
# predicates_df = predicates_df.with_columns(pl.col('timestamp').str.strptime(pl.Datetime, format='%m/%d/%Y %H:%M').cast(pl.Datetime))

# result = query_task(cfg_path=config_path, ESD=predicates_df)
# display(result)

# print('\ntrigger window')
# display(result.select(['timestamp']))

# print('\ngap window')
# display(result.select(['gap/timestamp', 'gap/window_summary']).unnest('gap/window_summary'))

# print('\ntarget window')
# display(result.select(['target/timestamp', 'target/window_summary']).unnest('target/window_summary'))

# print('\ninput window')
# display(result.select(['input/timestamp', 'input/window_summary']).unnest('input/window_summary'))