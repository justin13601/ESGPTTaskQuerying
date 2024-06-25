"""Tests the full end-to-end extraction process."""

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import subprocess
import tempfile
from pathlib import Path

import polars as pl
from loguru import logger
from polars.testing import assert_frame_equal

pl.enable_string_cache()

TS_FORMAT = "%m/%d/%Y %H:%M"
PRED_CNT_TYPE = pl.Int64
ANY_EVENT_COLUMN = "_ANY_EVENT"

# Data (input)
PREDICATES_CSV = """
subject_id,timestamp,admission,death,discharge,lab,spo2,normal_spo2,abnormally_low_spo2,abnormally_high_spo2,procedure_start,procedure_end,ventilation,diagnosis_ICD9CM/41071,diagnosis_ICD10CM/I214
1,12/1/1989 12:03,1,0,0,0,0,0,0,0,0,0,0,0,0
1,12/1/1989 13:14,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/1/1989 15:17,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/1/1989 16:17,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/1/1989 20:17,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/2/1989 3:00,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/2/1989 9:00,0,0,0,0,0,0,0,0,0,0,0,0,0
1,12/2/1989 10:00,0,0,0,0,0,0,0,0,1,0,1,0,0
1,12/2/1989 14:22,0,0,0,0,0,0,0,0,0,1,1,0,0
1,12/2/1989 15:00,0,0,1,0,0,0,0,0,0,0,0,0,0
1,1/21/1991 11:59,0,0,0,0,0,0,0,0,0,0,0,1,0
1,1/27/1991 23:32,1,0,0,0,0,0,0,0,0,0,0,0,0
1,1/27/1991 23:46,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/28/1991 3:18,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1/28/1991 3:28,0,0,0,0,0,0,0,0,1,0,1,0,0
1,1/28/1991 4:36,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/29/1991 23:32,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/30/1991 5:00,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1/30/1991 8:00,0,0,0,1,1,0,0,1,0,0,0,0,0
1,1/30/1991 11:00,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/30/1991 14:00,0,0,0,0,0,0,0,0,0,0,0,0,0
1,1/30/1991 14:15,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1/31/1991 1:00,0,0,0,0,0,0,0,0,0,1,1,0,0
1,1/31/1991 2:15,0,0,1,0,0,0,0,0,0,0,0,0,0
1,2/8/1991 8:15,0,0,0,1,1,1,0,0,0,0,0,0,0
1,3/3/1991 19:33,1,0,0,0,0,0,0,0,0,0,0,0,0
1,3/3/1991 20:33,0,0,0,1,1,0,1,0,0,0,0,0,0
1,3/3/1991 21:38,0,1,0,0,0,0,0,0,0,0,0,0,0
2,3/8/1996 2:24,1,0,0,0,0,0,0,0,0,0,0,0,0
2,3/8/1996 2:35,0,0,0,1,1,1,0,0,0,0,0,0,0
2,3/8/1996 4:00,0,0,0,1,1,1,0,0,0,0,0,0,0
2,3/8/1996 10:00,0,0,0,0,0,0,0,0,0,0,0,0,0
2,3/8/1996 16:00,0,0,1,0,0,0,0,0,0,0,0,0,0
2,6/5/1996 0:32,1,0,0,0,0,0,0,0,0,0,0,0,0
2,6/5/1996 0:48,0,0,0,0,0,0,0,0,0,0,0,0,1
2,6/5/1996 1:59,0,0,0,0,0,0,0,0,1,0,1,0,0
2,6/7/1996 6:00,0,0,0,1,0,0,0,0,0,0,0,0,0
2,6/7/1996 9:00,0,0,0,1,1,0,1,0,0,0,0,0,0
2,6/7/1996 12:00,0,0,0,1,1,1,0,0,0,0,0,0,0
2,6/7/1996 15:00,0,0,0,0,0,0,0,0,0,1,1,0,0
2,6/7/1996 15:00,0,0,0,0,0,0,0,0,0,0,0,0,0
2,6/8/1996 3:00,0,1,0,0,0,0,0,0,0,0,0,0,0
3,3/8/1996 2:22,0,0,0,0,0,0,0,0,1,0,1,0,0
3,3/8/1996 2:24,1,0,0,0,0,0,0,0,0,0,0,0,0
3,3/8/1996 2:37,0,0,0,1,1,1,0,0,0,0,0,0,0
3,3/9/1996 8:00,0,0,0,1,0,0,0,0,0,0,0,0,0
3,3/9/1996 11:00,0,0,0,1,1,1,0,0,0,0,0,0,0
3,3/9/1996 19:00,0,0,0,1,1,1,0,0,0,0,0,0,0
3,3/9/1996 22:00,0,0,0,0,0,0,0,0,0,0,0,0,0
3,3/11/1996 21:00,0,0,0,0,0,0,0,0,0,1,1,0,0
3,3/12/1996 0:00,0,1,0,0,0,0,0,0,0,0,0,0,0
"""

# Tasks (input)
TASKS_CFGS = {
    "inhospital_mortality": """
      # Task: 24-hour In-hospital Mortality Prediction
      predicates:
        admission:
          code: event_type//ADMISSION
        discharge:
          code: event_type//DISCHARGE
        death:
          code: event_type//DEATH
        discharge_or_death:
          expr: or(discharge, death)

      trigger: admission

      windows:
        input:
          start: NULL
          end: trigger + 24h
          start_inclusive: True
          end_inclusive: True
          has:
            _ANY_EVENT: (5, None)
          index_timestamp: end
        gap:
          start: trigger
          end: start + 48h
          start_inclusive: False
          end_inclusive: True
          has:
            admission: (None, 0)
            discharge: (None, 0)
            death: (None, 0)
        target:
          start: gap.end
          end: start -> discharge_or_death
          start_inclusive: False
          end_inclusive: True
          label: death
      """
}

# Expected output
EXPECTED_OUTPUT = {
    "inhospital_mortality": {
        "subject_id": [1, 2],
        "index_timestamp": ["01/28/1991 23:32", "06/06/1996 00:32"],
        "label": [0, 1],
        "trigger": ["01/27/1991 23:32", "06/05/1996 00:32"],
        "input.end_summary": [
            {
                "window_name": "input.end",
                "timestamp_at_start": "01/27/1991 23:32",
                "timestamp_at_end": "01/28/1991 23:32",
                "_LAST_EVENT_INDEX": 15,
                "admission": 0,
                "discharge": 0,
                "death": 0,
                "discharge_or_death": 0,
                "_ANY_EVENT": 4,
            },
            {
                "window_name": "input.end",
                "timestamp_at_start": "06/05/1996 00:32",
                "timestamp_at_end": "06/06/1996 00:32",
                "_LAST_EVENT_INDEX": 7,
                "admission": 0,
                "discharge": 0,
                "death": 0,
                "discharge_or_death": 0,
                "_ANY_EVENT": 2,
            },
        ],
        "input.start_summary": [
            {
                "window_name": "input.start",
                "timestamp_at_start": "12/01/1989 12:03",
                "timestamp_at_end": "01/28/1991 23:32",
                "_LAST_EVENT_INDEX": 15,
                "admission": 2,
                "discharge": 1,
                "death": 0,
                "discharge_or_death": 1,
                "_ANY_EVENT": 16,
            },
            {
                "window_name": "input.start",
                "timestamp_at_start": "03/08/1996 02:24",
                "timestamp_at_end": "06/06/1996 00:32",
                "_LAST_EVENT_INDEX": 7,
                "admission": 2,
                "discharge": 1,
                "death": 0,
                "discharge_or_death": 1,
                "_ANY_EVENT": 8,
            },
        ],
        "gap.end_summary": [
            {
                "window_name": "gap.end",
                "timestamp_at_start": "01/27/1991 23:32",
                "timestamp_at_end": "01/29/1991 23:32",
                "_LAST_EVENT_INDEX": 16,
                "admission": 0,
                "discharge": 0,
                "death": 0,
                "discharge_or_death": 0,
                "_ANY_EVENT": 5,
            },
            {
                "window_name": "gap.end",
                "timestamp_at_start": "06/05/1996 00:32",
                "timestamp_at_end": "06/07/1996 00:32",
                "_LAST_EVENT_INDEX": 7,
                "admission": 0,
                "discharge": 0,
                "death": 0,
                "discharge_or_death": 0,
                "_ANY_EVENT": 2,
            },
        ],
        "target.end_summary": [
            {
                "window_name": "target.end",
                "timestamp_at_start": "01/29/1991 23:32",
                "timestamp_at_end": "01/31/1991 02:15",
                "_LAST_EVENT_INDEX": 23,
                "admission": 0,
                "discharge": 1,
                "death": 0,
                "discharge_or_death": 1,
                "_ANY_EVENT": 7,
            },
            {
                "window_name": "target.end",
                "timestamp_at_start": "06/07/1996 00:32",
                "timestamp_at_end": "06/08/1996 03:00",
                "_LAST_EVENT_INDEX": 12,
                "admission": 0,
                "discharge": 0,
                "death": 1,
                "discharge_or_death": 1,
                "_ANY_EVENT": 5,
            },
        ],
    }
}


def run_command(script: str, hydra_kwargs: dict[str, str], test_name: str, expected_returncode: int = 0):
    command_parts = [script] + [f"{k}={v}" for k, v in hydra_kwargs.items()]
    command_out = subprocess.run(" ".join(command_parts), shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != expected_returncode:
        raise AssertionError(
            f"{test_name} returned {command_out.returncode} (expected {expected_returncode})!\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stderr, stdout


def assert_df_equal(want: pl.DataFrame, got: pl.DataFrame, msg: str = None, **kwargs):
    try:
        assert_frame_equal(want, got, **kwargs)
    except AssertionError as e:
        pl.Config.set_tbl_rows(-1)
        print(f"DFs are not equal: {msg}\nWant:")
        print(want)
        print("Got:")
        print(got)
        raise AssertionError(f"{msg}\n{e}") from e


def test_e2e():
    # Testing expand_shards
    es_stderr, es_stdout = run_command("expand_shards train/3 tuning/1", {}, "expand_shards")
    assert (
        es_stdout == "train/0,train/1,train/2,tuning/0\n"
    ), f"Expected 'train/0,train/1,train/2,tuning/0' but got '{es_stdout}'"

    # Running with the empty directory
    help_stderr, help_stdout = run_command("aces-cli", {}, "help", expected_returncode=1)
    assert (
        "Usage: aces-cli [OPTIONS]" in help_stdout
    ), f"Expected help message not found in stdout. Got {help_stdout}"

    with tempfile.TemporaryDirectory() as d:
        data_dir = Path(d) / "sample_data"
        configs_dir = Path(d) / "sample_configs"
        output_dir = Path(d) / "sample_output"

        # Create the directories
        data_dir.mkdir()
        configs_dir.mkdir()
        output_dir.mkdir()

        # Write the predicates CSV file
        predicates_csv = data_dir / "sample_data.csv"
        predicates_csv.write_text(PREDICATES_CSV.strip())

        # Run script and check the outputs
        all_stderrs = []
        all_stdouts = []
        full_stderr = ""
        full_stdout = ""
        try:
            for task_name, task_cfg in TASKS_CFGS.items():
                logger.info(f"Running task '{task_name}'...")

                # Write the task config YAMLs
                task_cfg_path = configs_dir / f"{task_name}.yaml"
                task_cfg_path.write_text(task_cfg)

                output_path = output_dir / f"{task_name}.parquet"

                extraction_config_kwargs = {
                    "data.path": str(predicates_csv.resolve()),
                    "data.standard": "direct",
                    "cohort_dir": str(configs_dir.resolve()),
                    "cohort_name": task_name,
                    "output_filepath": str(output_path.resolve()),
                    "hydra.verbose": True,
                }

                stderr, stdout = run_command("aces-cli", extraction_config_kwargs, task_name)
                stderr, stdout = run_command("aces-cli", extraction_config_kwargs, task_name)

                all_stderrs.append(stderr)
                all_stdouts.append(stdout)

                full_stderr = "\n".join(all_stderrs)
                full_stdout = "\n".join(all_stdouts)

                fp = output_dir / f"{task_name}.parquet"
                assert fp.is_file(), f"Expected {fp} to exist."
                got_df = pl.read_parquet(fp)

                # Check the columns
                expected_columns = EXPECTED_OUTPUT[task_name].keys()
                assert got_df.columns == list(expected_columns), f"Columns mismatch for task '{task_name}'"

                # Check the data
                for col_name, expected_data in EXPECTED_OUTPUT[task_name].items():
                    if col_name in ["index_timestamp", "trigger"]:
                        want = pl.DataFrame({col_name: expected_data}).with_columns(
                            pl.col(col_name).str.strptime(pl.Datetime, format=TS_FORMAT)
                        )
                    elif col_name.endswith("_summary"):
                        df_struct = pl.DataFrame(expected_data)
                        df_struct = df_struct.with_columns(
                            pl.col("timestamp_at_start").str.strptime(pl.Datetime, format=TS_FORMAT),
                            pl.col("timestamp_at_end").str.strptime(pl.Datetime, format=TS_FORMAT),
                        )
                        want = df_struct.select(
                            pl.struct(*[col for col in df_struct.columns]).alias(col_name)
                        )
                    else:
                        want = pl.DataFrame({col_name: expected_data}).with_columns(
                            pl.col(col_name).cast(PRED_CNT_TYPE)
                        )
                    got = got_df.select(col_name)
                    assert_df_equal(want, got, f"Data mismatch for task '{task_name}', column '{col_name}'")

        except AssertionError as e:
            print(f"Failed on task '{task_name}'")
            print(f"stderr:\n{full_stderr}")
            print(f"stdout:\n{full_stdout}")
            raise e
