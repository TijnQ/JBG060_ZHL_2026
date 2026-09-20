"""Run with: python -m data_quality.eda_quality_flood.check_flood_eda_data (no raw data required)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_quality.eda_quality_flood.flood_eda_data import (
    _combined_n_unique,
    _fmt_date,
    country_profile,
    expected_file_count,
    flood_files,
    profile_dataframe,
)


def _write_flood_root(temp: Path) -> Path:
    """Write a minimal flood-mask tree: 2 tiles x 1 year x 2 classes."""
    root = temp / "flood_masks"
    (root / "compact_recurring").mkdir(parents=True)
    (root / "compact_unusual").mkdir(parents=True)
    recurrent = pa.table({
        "date": pa.array(["2024-09-02", "2024-09-02", "2024-09-04", None], pa.string()).cast(pa.timestamp("s")),
        "lat": pa.array([9.0, 9.0, 10.0, 11.0], pa.float64()),
        "lon": pa.array([27.0, 28.0, 27.0, 28.0], pa.float64()),
        "tile": pa.array(["h20v08"] * 4),
        "cloud_frac": pa.array([0.0, 0.5, None, 1.0], pa.float64()),
    })
    unusual = pa.table({
        "date": pa.array(["2024-09-10"], pa.string()).cast(pa.timestamp("s")),
        "lat": pa.array([9.0], pa.float64()),
        "lon": pa.array([27.0], pa.float64()),
        "tile": pa.array(["h20v08"]),
        "cloud_frac": pa.array([0.0], pa.float64()),
    })
    # h21v08 files are empty but share the schema, so both tiles are exercised.
    empty = recurrent.slice(0, 0)
    pq.write_table(recurrent, root / "compact_recurring" / "flood_events_h20v08_2024.parquet")
    pq.write_table(unusual, root / "compact_unusual" / "flood_events_h20v08_2024.parquet")
    pq.write_table(empty, root / "compact_recurring" / "flood_events_h21v08_2024.parquet")
    pq.write_table(empty, root / "compact_unusual" / "flood_events_h21v08_2024.parquet")
    return root


def _check_profile_dataframe():
    frame = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": [np.nan, np.nan, np.nan, np.nan],
        "c": ["x", "x", "y", "z"],
        "d": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-03"]),
    })
    result = profile_dataframe(frame, level="sample").set_index("column")
    assert result.loc["a", "nan_count"] == 1 and result.loc["a", "nan_pct"] == 25.0
    assert result.loc["a", "min"] == 1.0 and result.loc["a", "max"] == 4.0
    assert abs(result.loc["a", "mean"] - 7 / 3) < 1e-12
    assert result.loc["b", "min"] is None and result.loc["b", "max"] is None  # all-null column
    assert result.loc["b", "nan_count"] == 4 and result.loc["b", "nan_pct"] == 100.0
    assert result.loc["c", "top_value"] == "x" and result.loc["c", "top_count"] == 2
    assert result.loc["d", "min"] == "2020-01-01" and result.loc["d", "max"] == "2020-01-03"
    assert list(result.index) == ["a", "b", "c", "d"]


def _check_country_profile():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_flood_root(Path(tmp))
        years = [2024]
        profile = country_profile(root, years=years)
        result = profile.set_index("column")
        # Recurring (4 rows) + unusual (1 row) => 5 country rows.
        assert result.loc["lat", "non_null_count"] == 5 and result.loc["lat", "nan_count"] == 0
        # lat range across both files.
        assert result.loc["lat", "min"] == 9.0 and result.loc["lat", "max"] == 11.0
        # cloud_frac has one null in the recurring file.
        assert result.loc["cloud_frac", "nan_count"] == 1
        assert result.loc["cloud_frac", "nan_pct"] == 20.0
        # date range across files and distinct tile / flood_type sets.
        assert result.loc["date", "min"] == "2024-09-02"
        assert result.loc["date", "max"] == "2024-09-10"
        assert result.loc["date", "nan_count"] == 1
        assert result.loc["tile", "top_value"] == "h20v08, h21v08"
        assert result.loc["tile", "n_unique"] == 2
        assert result.loc["flood_type", "top_value"] == "recurring, unusual"
        assert result.loc["flood_type", "n_unique"] == 2


def _check_helpers():
    assert expected_file_count(years=[2024]) == 4  # 2 tiles x 2 classes x 1 year
    assert expected_file_count() == 2 * 2 * 26
    files = flood_files(Path("."), years=[2024])
    assert len(files) == 4
    assert {f[1] for f in files} == {"h20v08", "h21v08"}
    assert {f[2] for f in files} == {"recurring", "unusual"}
    assert _fmt_date(pd.Timestamp("2024-09-02")) == "2024-09-02"
    assert _fmt_date(None) is None
    # _combined_n_unique across coalesced arrays.
    arrays = [pa.array([1.0, 2.0]), pa.array([2.0, 3.0]), pa.array([None])]
    assert _combined_n_unique(arrays) == 3
    assert _combined_n_unique([]) == 0


def run_checks():
    _check_helpers()
    _check_profile_dataframe()
    _check_country_profile()
    print("Passed: profile_table stats, NaN percentages, country aggregation, tile/class sets.")


if __name__ == "__main__":
    run_checks()