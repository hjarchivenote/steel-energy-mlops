"""Validate the original CSV schema and isolate chronological evaluation data."""

import csv
import hashlib
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TARGET = "Usage_kWh"
SOURCE_COLUMNS = (
    "date",
    TARGET,
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
)
NUMERIC_COLUMNS = SOURCE_COLUMNS[1:8]
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
LOAD_TYPES = frozenset({"Light_Load", "Medium_Load", "Maximum_Load"})


class DataValidationError(ValueError):
    """The dataset or preparation options violate the input contract."""


@dataclass(frozen=True)
class EnergyRecord:
    """Only timestamp and target survive validation; other measurements are excluded."""

    timestamp: datetime
    usage_kwh: float


@dataclass(frozen=True)
class ValidatedDataset:
    records: tuple[EnergyRecord, ...]
    source_sha256: str


@dataclass(frozen=True)
class DataSplit:
    train: tuple[EnergyRecord, ...]
    validation: tuple[EnergyRecord, ...]
    test: tuple[EnergyRecord, ...]

    def partitions(self) -> tuple[tuple[str, tuple[EnergyRecord, ...]], ...]:
        return (("train", self.train), ("validation", self.validation), ("test", self.test))


def _parse_record(row: dict[str, str], line: int) -> EnergyRecord:
    if any(not value.strip() for value in row.values()):
        raise DataValidationError(f"Row {line}: empty cells are not allowed.")
    try:
        timestamp = datetime.strptime(row["date"], "%d/%m/%Y %H:%M")
    except ValueError as exc:
        raise DataValidationError(f"Row {line}: date must use DD/MM/YYYY HH:MM.") from exc
    if timestamp.minute % 15:
        raise DataValidationError(f"Row {line}: timestamp must be on a 15-minute boundary.")

    numeric: dict[str, float] = {}
    for column in NUMERIC_COLUMNS:
        try:
            value = float(row[column])
        except ValueError as exc:
            raise DataValidationError(f"Row {line}: {column} must be numeric.") from exc
        if not math.isfinite(value) or value < 0:
            raise DataValidationError(f"Row {line}: {column} must be finite and non-negative.")
        numeric[column] = value
    for column in ("Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"):
        if numeric[column] > 100:
            raise DataValidationError(f"Row {line}: {column} must be at most 100.")

    if numeric["NSM"] != timestamp.hour * 3600 + timestamp.minute * 60:
        raise DataValidationError(f"Row {line}: NSM does not match the timestamp.")
    if row["Day_of_week"] != WEEKDAYS[timestamp.weekday()]:
        raise DataValidationError(f"Row {line}: Day_of_week does not match the timestamp.")
    expected_week_status = "Weekend" if timestamp.weekday() >= 5 else "Weekday"
    if row["WeekStatus"] != expected_week_status:
        raise DataValidationError(f"Row {line}: WeekStatus does not match the timestamp.")
    if row["Load_Type"] not in LOAD_TYPES:
        raise DataValidationError(f"Row {line}: unknown Load_Type.")
    return EnergyRecord(timestamp, numeric[TARGET])


def load_dataset(path: Path, expected_rows: int | None = None) -> ValidatedDataset:
    """Read a single byte snapshot, validate every row, and retain its provenance hash."""
    if expected_rows is not None and expected_rows < 1:
        raise DataValidationError("expected_rows must be a positive integer.")
    raw = path.read_bytes()
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    if next(reader, None) != list(SOURCE_COLUMNS):
        raise DataValidationError("CSV header must match the 11 original columns in their original order.")
    records: list[EnergyRecord] = []
    seen: set[datetime] = set()
    for line, values in enumerate(reader, start=2):
        if len(values) != len(SOURCE_COLUMNS):
            raise DataValidationError(f"Row {line}: expected exactly 11 cells.")
        record = _parse_record(dict(zip(SOURCE_COLUMNS, values, strict=True)), line)
        if record.timestamp in seen:
            raise DataValidationError(f"Row {line}: duplicate timestamp.")
        seen.add(record.timestamp)
        records.append(record)
    if not records:
        raise DataValidationError("Dataset must contain at least one record.")
    if expected_rows is not None and len(records) != expected_rows:
        raise DataValidationError(f"Expected {expected_rows} records, found {len(records)}.")
    return ValidatedDataset(tuple(records), hashlib.sha256(raw).hexdigest())


def split_chronologically(
    records: Sequence[EnergyRecord], train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> DataSplit:
    """Sort unique timestamps and reserve the latest records for untouched final evaluation."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise DataValidationError("Train and validation fractions must be positive and sum to less than one.")
    if any(record.timestamp.tzinfo is not None for record in records):
        raise DataValidationError("Timestamps must be timezone-naive, consistent with the source CSV.")
    if len({record.timestamp for record in records}) != len(records):
        raise DataValidationError("Duplicate timestamps cannot cross evaluation boundaries.")
    ordered = tuple(sorted(records, key=lambda record: record.timestamp))
    train_end = int(len(ordered) * train_fraction)
    validation_end = train_end + int(len(ordered) * validation_fraction)
    if min(train_end, validation_end - train_end, len(ordered) - validation_end) < 1:
        raise DataValidationError("Each chronological partition must contain at least one record.")
    return DataSplit(ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:])
