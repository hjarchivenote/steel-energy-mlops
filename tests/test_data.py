import csv
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from steel_energy_mlops.data import (
    NUMERIC_COLUMNS,
    SOURCE_COLUMNS,
    DataValidationError,
    EnergyRecord,
    load_dataset,
    split_chronologically,
)
from tests.helpers import make_row, write_csv


def test_loads_bom_csv_with_byte_provenance(dataset_path: Path) -> None:
    result = load_dataset(dataset_path, expected_rows=20)
    assert len(result.records) == 20
    assert result.records[0] == EnergyRecord(datetime(2018, 1, 1), 0.0)
    assert result.source_sha256 == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert load_dataset(dataset_path).records == result.records


def test_weekend_observation_is_valid(tmp_path: Path) -> None:
    row = make_row(datetime(2018, 1, 6, 12, 30))
    result = load_dataset(write_csv(tmp_path / "weekend.csv", [row]))
    assert result.records[0].timestamp == datetime(2018, 1, 6, 12, 30)


@pytest.mark.parametrize("header", [SOURCE_COLUMNS[:-1], SOURCE_COLUMNS[::-1], SOURCE_COLUMNS + ("extra",)])
def test_rejects_changed_schema(tmp_path: Path, header: tuple[str, ...]) -> None:
    with pytest.raises(DataValidationError, match="CSV header"):
        load_dataset(write_csv(tmp_path / "bad.csv", [], header))


def test_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(DataValidationError, match="CSV header"):
        load_dataset(path)


def test_rejects_header_without_observations(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="at least one record"):
        load_dataset(write_csv(tmp_path / "empty.csv", []))


@pytest.mark.parametrize("expected", [0, -1])
def test_rejects_invalid_expected_count(dataset_path: Path, expected: int) -> None:
    with pytest.raises(DataValidationError, match="positive integer"):
        load_dataset(dataset_path, expected)


def test_rejects_wrong_record_count(dataset_path: Path) -> None:
    with pytest.raises(DataValidationError, match="Expected 21 records, found 20"):
        load_dataset(dataset_path, expected_rows=21)


@pytest.mark.parametrize("cells", [10, 12])
def test_rejects_ragged_rows(tmp_path: Path, cells: int) -> None:
    with pytest.raises(DataValidationError, match="exactly 11 cells"):
        load_dataset(write_csv(tmp_path / "bad.csv", [["value"] * cells]))


@pytest.mark.parametrize("empty", ["", "   "])
def test_rejects_missing_cell_without_dropping_row(tmp_path: Path, empty: str) -> None:
    row = make_row(datetime(2018, 1, 1))
    row[1] = empty
    with pytest.raises(DataValidationError, match="Row 2: empty cells"):
        load_dataset(write_csv(tmp_path / "bad.csv", [row]))


@pytest.mark.parametrize("date", ["not-a-date", "31/02/2018 00:00", "2018-01-01 00:00"])
def test_rejects_invalid_or_ambiguous_date_format(tmp_path: Path, date: str) -> None:
    row = make_row(datetime(2018, 1, 1))
    row[0] = date
    with pytest.raises(DataValidationError, match="DD/MM/YYYY"):
        load_dataset(write_csv(tmp_path / "bad.csv", [row]))


def test_rejects_off_grid_timestamp(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="15-minute boundary"):
        load_dataset(write_csv(tmp_path / "bad.csv", [make_row(datetime(2018, 1, 1, 0, 1))]))


@pytest.mark.parametrize("column", NUMERIC_COLUMNS)
@pytest.mark.parametrize("value", ["not-numeric", "NaN", "inf", "-inf", "-0.01"])
def test_rejects_invalid_numeric_measurements(tmp_path: Path, column: str, value: str) -> None:
    row = make_row(datetime(2018, 1, 1))
    row[SOURCE_COLUMNS.index(column)] = value
    with pytest.raises(DataValidationError, match=column.replace("(", r"\(").replace(")", r"\)")):
        load_dataset(write_csv(tmp_path / "bad.csv", [row]))


@pytest.mark.parametrize("column", ["Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"])
def test_rejects_impossible_power_factor(tmp_path: Path, column: str) -> None:
    row = make_row(datetime(2018, 1, 1))
    row[SOURCE_COLUMNS.index(column)] = "100.1"
    with pytest.raises(DataValidationError, match="at most 100"):
        load_dataset(write_csv(tmp_path / "bad.csv", [row]))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("NSM", "900", "NSM does not match"),
        ("Day_of_week", "Tuesday", "Day_of_week does not match"),
        ("WeekStatus", "Weekend", "WeekStatus does not match"),
        ("Load_Type", "Unknown", "unknown Load_Type"),
    ],
)
def test_rejects_inconsistent_metadata(tmp_path: Path, column: str, value: str, message: str) -> None:
    row = make_row(datetime(2018, 1, 1))
    row[SOURCE_COLUMNS.index(column)] = value
    with pytest.raises(DataValidationError, match=message):
        load_dataset(write_csv(tmp_path / "bad.csv", [row]))


def test_rejects_duplicate_timestamps_even_with_different_targets(tmp_path: Path) -> None:
    rows = [make_row(datetime(2018, 1, 1), usage) for usage in (10.0, 20.0)]
    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        load_dataset(write_csv(tmp_path / "bad.csv", rows))


def test_rejects_malformed_quoting(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(",".join(SOURCE_COLUMNS) + '\n"unclosed quote', encoding="utf-8")
    with pytest.raises(csv.Error):
        load_dataset(path)


def test_split_sorts_irregular_timestamps_without_changing_observations() -> None:
    start = datetime(2018, 1, 1)
    records = tuple(EnergyRecord(start + timedelta(minutes=15 * i * i), float(i)) for i in reversed(range(20)))
    split = split_chronologically(records)
    assert [len(part) for _, part in split.partitions()] == [14, 3, 3]
    combined = split.train + split.validation + split.test
    assert combined == tuple(sorted(records, key=lambda row: row.timestamp))
    assert records[0].timestamp > records[-1].timestamp
    assert split.train[-1].timestamp < split.validation[0].timestamp
    assert split.validation[-1].timestamp < split.test[0].timestamp
    assert len({record.timestamp for record in combined}) == 20
    assert combined[2].timestamp - combined[1].timestamp == timedelta(minutes=45)


def test_explicit_split_fractions(dataset_path: Path) -> None:
    split = split_chronologically(load_dataset(dataset_path).records, 0.5, 0.25)
    assert [len(part) for _, part in split.partitions()] == [10, 5, 5]


@pytest.mark.parametrize(
    ("train", "validation"),
    [(0.0, 0.15), (-0.1, 0.15), (0.7, 0.0), (0.7, 0.3), (1.0, 0.1), (float("nan"), 0.1), (0.7, float("inf"))],
)
def test_invalid_split_fractions(dataset_path: Path, train: float, validation: float) -> None:
    with pytest.raises(DataValidationError, match="fractions"):
        split_chronologically(load_dataset(dataset_path).records, train, validation)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_split_requires_nonempty_partitions(count: int) -> None:
    records = tuple(EnergyRecord(datetime(2018, 1, 1) + timedelta(minutes=15 * i), 1.0) for i in range(count))
    with pytest.raises(DataValidationError, match="Each chronological partition"):
        split_chronologically(records)


def test_split_rejects_duplicate_timestamps() -> None:
    record = EnergyRecord(datetime(2018, 1, 1), 1.0)
    with pytest.raises(DataValidationError, match="Duplicate timestamps"):
        split_chronologically([record] * 20)


def test_split_rejects_timezone_conversion() -> None:
    record = EnergyRecord(datetime(2018, 1, 1, tzinfo=timezone.utc), 1.0)
    with pytest.raises(DataValidationError, match="timezone-naive"):
        split_chronologically([record])


def test_target_values_do_not_determine_partition_membership(dataset_path: Path) -> None:
    records = load_dataset(dataset_path).records
    changed_targets = tuple(EnergyRecord(row.timestamp, 1_000_000 - row.usage_kwh) for row in records)
    original = split_chronologically(records)
    changed = split_chronologically(changed_targets)
    for (_, left), (_, right) in zip(original.partitions(), changed.partitions(), strict=True):
        assert [record.timestamp for record in left] == [record.timestamp for record in right]
