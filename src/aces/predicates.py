"""This module contains functions for generating predicate columns for event sequences."""

from pathlib import Path

import polars as pl
from loguru import logger
from omegaconf import DictConfig

from .config import TaskExtractorConfig
from .types import (
    ANY_EVENT_COLUMN,
    END_OF_RECORD_KEY,
    EVENT_INDEX_COLUMN,
    EVENT_INDEX_TYPE,
    PRED_CNT_TYPE,
    START_OF_RECORD_KEY,
)


def direct_load_plain_predicates(
    data_path: Path, predicates: list[str], ts_format: str | None
) -> pl.DataFrame:
    """Loads a CSV file from disk and verifies that the necessary plain predicate columns are present.

    This CSV file must have the following columns:
        - subject_id: The subject identifier.
        - timestamp: The timestamp of the event, in the format "MM/DD/YYYY HH:MM".
        - Any additional columns specified in the set of desired plain predicates.

    Args:
        data_path: The path to the CSV file.
        predicates: The list of columns to read from the CSV file.

    Returns:
        The Polars DataFrame containing the specified columns.

    Example:
        >>> import tempfile
        >>> CSV_data = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "timestamp": ["01/01/2021 00:00", "01/01/2021 12:00", "01/02/2021 00:00"],
        ...     "is_admission": [1, 0, 1],
        ...     "is_discharge": [0, 1, 0],
        ... })
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_parquet(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], "%m/%d/%Y %H:%M")
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            │
        └────────────┴─────────────────────┴──────────────┴──────────────┘

    If the timestamp column is already a timestamp, then the `ts_format` argument id not needed, but can be
    used without an error.
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     (
        ...         CSV_data
        ...         .with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))
        ...         .write_parquet(data_path)
        ...     )
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], "%m/%d/%Y %H:%M")
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            │
        └────────────┴─────────────────────┴──────────────┴──────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     (
        ...         CSV_data
        ...         .with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))
        ...         .write_parquet(data_path)
        ...     )
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], None)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            │
        └────────────┴─────────────────────┴──────────────┴──────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], "%m/%d/%Y %H:%M")
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            │
        └────────────┴─────────────────────┴──────────────┴──────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_discharge"], "%m/%d/%Y %H:%M")
        shape: (3, 3)
        ┌────────────┬─────────────────────┬──────────────┐
        │ subject_id ┆ timestamp           ┆ is_discharge │
        │ ---        ┆ ---                 ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64          │
        ╞════════════╪═════════════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0            │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 1            │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 0            │
        └────────────┴─────────────────────┴──────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_foobar"], "%m/%d/%Y %H:%M")
        Traceback (most recent call last):
            ...
        polars.exceptions.ColumnNotFoundError: ['is_foobar']
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".foo") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_discharge"], "%m/%d/%Y %H:%M")
        Traceback (most recent call last):
            ...
        ValueError: Unsupported file format: .foo
        >>> with tempfile.TemporaryDirectory() as d:
        ...     data_path = Path(d) / "data.csv"
        ...     assert not data_path.exists()
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], "%m/%d/%Y %H:%M")
        Traceback (most recent call last):
            ...
        FileNotFoundError: Direct predicates file ... does not exist!
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_parquet(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], None)
        Traceback (most recent call last):
            ...
        ValueError: Must provide a timestamp format for direct predicates with str timestamps.
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     (
        ...         CSV_data
        ...         .with_columns(
        ...             pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
        ...             .dt.timestamp()
        ...         )
        ...         .write_parquet(data_path)
        ...     )
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], None)
        Traceback (most recent call last):
            ...
        TypeError: Passed predicates have timestamps of invalid type Int64.
    """

    columns = ["subject_id", "timestamp"] + predicates
    logger.info(f"Attempting to load {columns} from file {str(data_path.resolve())}")

    if not data_path.is_file():
        raise FileNotFoundError(f"Direct predicates file {data_path} does not exist!")

    match data_path.suffix:
        case ".csv":
            data = pl.scan_csv(data_path)
        case ".parquet":
            data = pl.scan_parquet(data_path)
        case _:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        raise pl.ColumnNotFoundError(missing_columns)

    data = data.select(columns).drop_nulls(subset=["subject_id", "timestamp"])

    ts_type = data.schema["timestamp"]
    if ts_type == pl.Utf8:
        if ts_format is None:
            raise ValueError("Must provide a timestamp format for direct predicates with str timestamps.")
        data = data.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format=ts_format))
    elif ts_type.is_temporal():
        if ts_format is not None:
            logger.info(
                f"Ignoring specified timestamp format of {ts_format} as timestamps are already {ts_type}"
            )
    else:
        raise TypeError(f"Passed predicates have timestamps of invalid type {ts_type}.")

    logger.info("Cleaning up predicates dataframe...")
    return (
        data.select("subject_id", "timestamp", *predicates)
        .group_by(["subject_id", "timestamp"], maintain_order=True)
        .agg(pl.all().sum().cast(PRED_CNT_TYPE))
        .collect()
    )


def generate_plain_predicates_from_meds(data_path: Path, predicates: dict) -> pl.DataFrame:
    """Generate plain predicate columns from a MEDS dataset.

    To learn more about the MEDS format, please visit https://github.com/Medical-Event-Data-Standard/meds

    Args:
        data_path: The path to the MEDS dataset file.
        predicates: The dictionary of plain predicate configurations.

    Returns:
        The Polars DataFrame containing the extracted predicates per subject per timestamp across the entire
        MEDS dataset.

    Example:
        >>> import tempfile
        >>> from .config import PlainPredicateConfig
        >>> parquet_data = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 3],
        ...     "timestamp": ["1/1/1989 00:00", "1/1/1989 01:00", "1/1/1989 01:00", "1/1/1989 02:00", None],
        ...     "code": ['admission', 'discharge', 'discharge', 'admission', "gender"],
        ... }).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     parquet_data.write_parquet(data_path)
        ...     generate_plain_predicates_from_meds(
        ...         data_path,
        ...         {"discharge": PlainPredicateConfig("discharge")}
        ...     )
        shape: (3, 3)
        ┌────────────┬─────────────────────┬───────────┐
        │ subject_id ┆ timestamp           ┆ discharge │
        │ ---        ┆ ---                 ┆ ---       │
        │ i64        ┆ datetime[μs]        ┆ i64       │
        ╞════════════╪═════════════════════╪═══════════╡
        │ 1          ┆ 1989-01-01 00:00:00 ┆ 0         │
        │ 1          ┆ 1989-01-01 01:00:00 ┆ 2         │
        │ 2          ┆ 1989-01-01 02:00:00 ┆ 0         │
        └────────────┴─────────────────────┴───────────┘
    """

    logger.info("Loading MEDS data...")
    data = (
        pl.read_parquet(data_path)
        .rename({"patient_id": "subject_id"})
        .drop_nulls(subset=["subject_id", "timestamp"])
    )

    # generate plain predicate columns
    logger.info("Generating plain predicate columns...")
    for name, plain_predicate in predicates.items():
        data = data.with_columns(plain_predicate.MEDS_eval_expr().cast(PRED_CNT_TYPE).alias(name))
        logger.info(f"Added predicate column '{name}'.")

    # clean up predicates_df
    logger.info("Cleaning up predicates dataframe...")
    predicate_cols = list(predicates.keys())
    return (
        data.select(["subject_id", "timestamp"] + predicate_cols)
        .group_by(["subject_id", "timestamp"], maintain_order=True)
        .agg(*(pl.col(c).sum().cast(PRED_CNT_TYPE).alias(c) for c in predicate_cols))
    )


def process_esgpt_data(
    events_df: pl.DataFrame,
    dynamic_measurements_df: pl.DataFrame,
    value_columns: dict[str, str],
    predicates: dict,
) -> pl.DataFrame:
    """Process ESGPT data to generate plain predicate columns.

    Args:
        events_df: The Polars DataFrame containing the events data.
        dynamic_measurements_df: The Polars DataFrame containing the dynamic measurements data.

    Returns:
        The Polars DataFrame containing the extracted predicates per subject per timestamp across the entire
        ESGPT dataset.

    Examples:
        >>> from datetime import datetime
        >>> from .config import PlainPredicateConfig
        >>> events_df = pl.DataFrame({
        ...    "event_id": [1, 2, 3, 4],
        ...    "subject_id": [1, 1, 2, 2],
        ...    "timestamp": [
        ...         datetime(2021, 1, 1, 0, 0),
        ...         datetime(2021, 1, 1, 12, 0),
        ...         datetime(2021, 1, 2, 0, 0),
        ...         datetime(2021, 1, 2, 12, 0),
        ...    ],
        ...    "event_type": ["adm", "dis", "adm", "obs"],
        ...    "age": [30, 30, 40, 40],
        ... })
        >>> dynamic_measurements_df = pl.DataFrame({
        ...    "event_id": [1,     1,    1,    2,    2,    2,    3,     4,    5],
        ...    "adm_loc":  ["foo", None, None, None, None, None, "bar", None, None],
        ...    "dis_loc":  [None,  None, None, None, None, "H",  None,  None, None],
        ...    "HR":       [None,  150,  None, 120,  None, None, None,  177,  89],
        ...    "lab":      [None,  None, "K",  None, "K",  None, None,  None, "SpO2"],
        ...    "lab_val":  [None,  None, 5.1,  None, 3.8,  None, None,  None, 99],
        ... })
        >>> value_columns = {
        ...    "is_admission": None,
        ...    "is_discharge": None,
        ...    "high_HR": "HR",
        ...    "high_Potassium": "lab_val",
        ... }
        >>> predicates = {
        ...    "is_admission": PlainPredicateConfig(code="event_type//adm"),
        ...    "is_discharge": PlainPredicateConfig(code="event_type//dis"),
        ...    "high_HR": PlainPredicateConfig(code="HR", value_min=140),
        ...    "high_Potassium": PlainPredicateConfig(code="lab//K", value_min=5.0),
        ... }
        >>> process_esgpt_data(events_df, dynamic_measurements_df, value_columns, predicates)
        shape: (4, 6)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┬─────────┬────────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge ┆ high_HR ┆ high_Potassium │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          ┆ ---     ┆ ---            │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64     ┆ i64            │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╪═════════╪════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            ┆ 1       ┆ 1              │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            ┆ 0       ┆ 0              │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            ┆ 0       ┆ 0              │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0            ┆ 0            ┆ 1       ┆ 0              │
        └────────────┴─────────────────────┴──────────────┴──────────────┴─────────┴────────────────┘
    """

    logger.info("Generating plain predicate columns...")
    for name, plain_predicate in predicates.items():
        if "event_type" in plain_predicate.code:
            events_df = events_df.with_columns(
                plain_predicate.ESGPT_eval_expr().cast(PRED_CNT_TYPE).alias(name)
            )
        else:
            values_column = value_columns[name]
            dynamic_measurements_df = dynamic_measurements_df.with_columns(
                plain_predicate.ESGPT_eval_expr(values_column).cast(PRED_CNT_TYPE).alias(name)
            )
        logger.info(f"Added predicate column '{name}'.")

    predicate_cols = list(predicates.keys())

    # aggregate dynamic_measurements_df by summing predicates (counts)
    dynamic_measurements_df = (
        dynamic_measurements_df.group_by(["event_id"])
        .agg(
            *[
                pl.col(c).sum().cast(PRED_CNT_TYPE)
                for c in dynamic_measurements_df.columns
                if c in predicate_cols
            ],
        )
        .select(["event_id"] + [c for c in dynamic_measurements_df.columns if c in predicate_cols])
    )

    # join events_df and dynamic_measurements_df for the final predicates_df
    data = events_df.join(dynamic_measurements_df, on="event_id", how="left")

    # clean up predicates_df
    logger.info("Cleaning up predicates dataframe...")
    return data.select(["subject_id", "timestamp"] + predicate_cols)


def generate_plain_predicates_from_esgpt(data_path: Path, predicates: dict) -> pl.DataFrame:
    """Generate plain predicate columns from an ESGPT dataset.

    To learn more about the ESGPT format, please visit https://eventstreamml.readthedocs.io/en/latest/

    Args:
        data_path: The path to the ESGPT dataset directory.
        predicates: The dictionary of plain predicate configurations.

    Returns:
        The Polars DataFrame containing the extracted predicates per subject per timestamp across the entire
        ESGPT dataset.
    """

    try:
        from EventStream.data.dataset_polars import Dataset
    except ImportError as e:
        raise ImportError(
            "The 'EventStream' package is required to load ESGPT datasets. "
            "If you mean to use a MEDS dataset, please specify the 'MEDS' standard. "
            "Otherwise, please install the package from https://github.com/mmcdermott/EventStreamGPT and add "
            "the package to your PYTHONPATH."
        ) from e

    try:
        ESD = Dataset.load(data_path)
    except Exception as e:
        raise ValueError(
            f"Error loading data using ESGPT: {e}. "
            "Please ensure the path provided is a valid ESGPT dataset directory. "
            "If you mean to use a MEDS dataset, please specify the 'MEDS' standard."
        ) from e

    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    config = ESD.config

    value_columns = {}
    for name, plain_predicate in predicates.items():
        if "event_type" in plain_predicate.code:
            value_columns[name] = None
        else:
            measurement_name = plain_predicate.code.split("//")[0]
            value_columns[name] = config.measurement_configs[measurement_name].values_column

    return process_esgpt_data(events_df, dynamic_measurements_df, value_columns, predicates)


def get_predicates_df(cfg: TaskExtractorConfig, data_config: DictConfig) -> pl.DataFrame:
    """Generate predicate columns based on the configuration.

    Args:
        cfg: The TaskExtractorConfig object containing the predicates information.
        data_path: Path to external data (file path to .csv or .parquet, or ESGPT directory) as
            string or Path.
        standard: The data standard, either 'CSV, 'MEDS' or 'ESGPT'.

    Returns:
        pl.DataFrame: The Polars DataFrame with the added predicate columns.

    Raises:
        ValueError: If an invalid predicate type is specified in the configuration.

    Example:
        >>> import tempfile
        >>> from .config import PlainPredicateConfig, DerivedPredicateConfig, EventConfig, WindowConfig
        >>> data = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 2],
        ...     "timestamp": ["01/01/2021 00:00", "01/01/2021 12:00", "01/02/2021 00:00", "01/02/2021 12:00"],
        ...     "adm":       [1, 0, 1, 0],
        ...     "dis":       [0, 1, 0, 0],
        ...     "death":     [0, 0, 0, 1],
        ... })
        >>> predicates = {
        ...     "adm": PlainPredicateConfig("adm"),
        ...     "dis": PlainPredicateConfig("dis"),
        ...     "death": PlainPredicateConfig("death"),
        ...     "death_or_dis": DerivedPredicateConfig("or(death, dis)"),
        ... }
        >>> trigger = EventConfig("adm")
        >>> windows = {
        ...     "input": WindowConfig(
        ...         start=None,
        ...         end="trigger + 24h",
        ...         start_inclusive=True,
        ...         end_inclusive=True,
        ...         has={"_ANY_EVENT": "(32, None)"},
        ...     ),
        ...     "gap": WindowConfig(
        ...         start="input.end",
        ...         end="start + 24h",
        ...         start_inclusive=False,
        ...         end_inclusive=True,
        ...         has={
        ...             "death_or_dis": "(None, 0)",
        ...             "adm": "(None, 0)",
        ...         },
        ...     ),
        ...     "target": WindowConfig(
        ...         start="gap.end",
        ...         end="start -> death_or_dis",
        ...         start_inclusive=False,
        ...         end_inclusive=True,
        ...         has={},
        ...     ),
        ... }
        >>> config = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     data.write_csv(data_path)
        ...     data_config = DictConfig({
        ...         "path": str(data_path), "standard": "direct", "ts_format": "%m/%d/%Y %H:%M"
        ...     })
        ...     get_predicates_df(config, data_config)
        shape: (4, 8)
        ┌────────────┬─────────────────────┬─────┬─────┬───────┬──────────────┬────────────┬──────────────┐
        │ subject_id ┆ timestamp           ┆ adm ┆ dis ┆ death ┆ death_or_dis ┆ _ANY_EVENT ┆ _EVENT_INDEX │
        │ ---        ┆ ---                 ┆ --- ┆ --- ┆ ---   ┆ ---          ┆ ---        ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64 ┆ i64 ┆ i64   ┆ i64          ┆ i64        ┆ i64          │
        ╞════════════╪═════════════════════╪═════╪═════╪═══════╪══════════════╪════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1   ┆ 0   ┆ 0     ┆ 0            ┆ 1          ┆ 0            │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0   ┆ 1   ┆ 0     ┆ 1            ┆ 1          ┆ 1            │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1   ┆ 0   ┆ 0     ┆ 0            ┆ 1          ┆ 0            │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0   ┆ 0   ┆ 1     ┆ 1            ┆ 1          ┆ 1            │
        └────────────┴─────────────────────┴─────┴─────┴───────┴──────────────┴────────────┴──────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     (
        ...         data
        ...         .with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))
        ...         .write_parquet(data_path)
        ...     )
        ...     data_config = DictConfig({"path": str(data_path), "standard": "direct", "ts_format": None})
        ...     get_predicates_df(config, data_config)
        shape: (4, 8)
        ┌────────────┬─────────────────────┬─────┬─────┬───────┬──────────────┬────────────┬──────────────┐
        │ subject_id ┆ timestamp           ┆ adm ┆ dis ┆ death ┆ death_or_dis ┆ _ANY_EVENT ┆ _EVENT_INDEX │
        │ ---        ┆ ---                 ┆ --- ┆ --- ┆ ---   ┆ ---          ┆ ---        ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64 ┆ i64 ┆ i64   ┆ i64          ┆ i64        ┆ i64          │
        ╞════════════╪═════════════════════╪═════╪═════╪═══════╪══════════════╪════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1   ┆ 0   ┆ 0     ┆ 0            ┆ 1          ┆ 0            │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0   ┆ 1   ┆ 0     ┆ 1            ┆ 1          ┆ 1            │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1   ┆ 0   ┆ 0     ┆ 0            ┆ 1          ┆ 0            │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0   ┆ 0   ┆ 1     ┆ 1            ┆ 1          ┆ 1            │
        └────────────┴─────────────────────┴─────┴─────┴───────┴──────────────┴────────────┴──────────────┘
        >>> any_event_trigger = EventConfig("_ANY_EVENT")
        >>> adm_only_predicates = {"adm": PlainPredicateConfig("adm")}
        >>> st_end_windows = {
        ...     "input": WindowConfig(
        ...         start="end - 365d",
        ...         end="trigger + 24h",
        ...         start_inclusive=True,
        ...         end_inclusive=True,
        ...         has={
        ...             "_RECORD_END": "(None, 0)",   # These are added just to show start/end predicates
        ...             "_RECORD_START": "(None, 0)", # These are added just to show start/end predicates
        ...         },
        ...     ),
        ... }
        >>> st_end_config = TaskExtractorConfig(
        ...     predicates=adm_only_predicates, trigger=any_event_trigger, windows=st_end_windows
        ... )
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     data.write_csv(data_path)
        ...     data_config = DictConfig({
        ...         "path": str(data_path), "standard": "direct", "ts_format": "%m/%d/%Y %H:%M"
        ...     })
        ...     get_predicates_df(st_end_config, data_config)
        shape: (4, 6)
        ┌────────────┬─────────────────────┬─────┬────────────┬───────────────┬─────────────┐
        │ subject_id ┆ timestamp           ┆ adm ┆ _ANY_EVENT ┆ _RECORD_START ┆ _RECORD_END │
        │ ---        ┆ ---                 ┆ --- ┆ ---        ┆ ---           ┆ ---         │
        │ i64        ┆ datetime[μs]        ┆ i64 ┆ i64        ┆ i64           ┆ i64         │
        ╞════════════╪═════════════════════╪═════╪════════════╪═══════════════╪═════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1   ┆ 1          ┆ 1             ┆ 0           │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0   ┆ 1          ┆ 0             ┆ 1           │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1   ┆ 1          ┆ 1             ┆ 0           │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0   ┆ 1          ┆ 0             ┆ 1           │
        └────────────┴─────────────────────┴─────┴────────────┴───────────────┴─────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     data.write_csv(data_path)
        ...     data_config = DictConfig({
        ...         "path": str(data_path), "standard": "buzz", "ts_format": "%m/%d/%Y %H:%M"
        ...     })
        ...     get_predicates_df(config, data_config)
        Traceback (most recent call last):
            ...
        ValueError: Invalid data standard: buzz. Options are 'direct', 'MEDS', 'ESGPT'.
    """

    standard = data_config.standard
    data_path = Path(data_config.path)

    # plain predicates
    plain_predicates = cfg.plain_predicates
    match standard.lower():
        case "direct":
            ts_format = data_config.ts_format
            data = direct_load_plain_predicates(data_path, list(plain_predicates.keys()), ts_format)
        case "meds":
            data = generate_plain_predicates_from_meds(data_path, plain_predicates)
        case "esgpt":
            data = generate_plain_predicates_from_esgpt(data_path, plain_predicates)
        case _:
            raise ValueError(f"Invalid data standard: {standard}. Options are 'direct', 'MEDS', 'ESGPT'.")
    predicate_cols = list(plain_predicates.keys())

    # derived predicates
    logger.info("Loaded plain predicates. Generating derived predicate columns...")
    for name, code in cfg.derived_predicates.items():
        data = data.with_columns(code.eval_expr().cast(PRED_CNT_TYPE).alias(name))
        logger.info(f"Added predicate column '{name}'.")
        predicate_cols.append(name)

    data = data.sort(by=["subject_id", "timestamp"])

    # add special predicates:
    # a column of 1s representing any predicate
    # a column of 0s with 1 in the first event of each subject_id representing the start of record
    # a column of 0s with 1 in the last event of each subject_id representing the end of record
    logger.info("Generating special predicate columns...")
    special_predicates = []
    for window in cfg.windows.values():
        if ANY_EVENT_COLUMN in window.referenced_predicates and ANY_EVENT_COLUMN not in special_predicates:
            special_predicates.append(ANY_EVENT_COLUMN)
        if (
            START_OF_RECORD_KEY in window.constraint_predicates
            and START_OF_RECORD_KEY not in special_predicates
        ):
            special_predicates.append(START_OF_RECORD_KEY)
        if END_OF_RECORD_KEY in window.constraint_predicates and END_OF_RECORD_KEY not in special_predicates:
            special_predicates.append(END_OF_RECORD_KEY)

    if (
        cfg.trigger.predicate in [ANY_EVENT_COLUMN, START_OF_RECORD_KEY, END_OF_RECORD_KEY]
        and cfg.trigger.predicate not in special_predicates
    ):
        special_predicates.append(cfg.trigger.predicate)

    if ANY_EVENT_COLUMN in special_predicates:
        data = data.with_columns(pl.lit(1).alias(ANY_EVENT_COLUMN).cast(PRED_CNT_TYPE))
        logger.info(f"Added predicate column '{ANY_EVENT_COLUMN}'.")
        predicate_cols.append(ANY_EVENT_COLUMN)
    if START_OF_RECORD_KEY in special_predicates:
        data = data.with_columns(
            [
                (pl.col("timestamp") == pl.col("timestamp").min().over("subject_id"))
                .cast(PRED_CNT_TYPE)
                .alias(START_OF_RECORD_KEY)
            ]
        )
        logger.info(f"Added predicate column '{START_OF_RECORD_KEY}'.")
    if END_OF_RECORD_KEY in special_predicates:
        data = data.with_columns(
            [
                (pl.col("timestamp") == pl.col("timestamp").max().over("subject_id"))
                .cast(PRED_CNT_TYPE)
                .alias(END_OF_RECORD_KEY)
            ]
        )
        logger.info(f"Added predicate column '{END_OF_RECORD_KEY}'.")
    predicate_cols += special_predicates

    # create a column for event_id
    data = data.with_columns(pl.lit(1).alias(EVENT_INDEX_COLUMN))
    data = data.with_columns(
        (pl.col(EVENT_INDEX_COLUMN).cum_sum().over("subject_id") - 1).cast(EVENT_INDEX_TYPE)
    )

    return data
