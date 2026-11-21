"""Verified historical context shared by replay and input-only compatibility checks."""

import csv
import hashlib
from datetime import datetime, timedelta

import numpy as np

from steel_energy_mlops.data import SOURCE_COLUMNS, DataValidationError, _parse_record
from steel_energy_mlops.forecast_inputs import AvailableObservation, build_forecast_inputs

DELAY = timedelta(minutes=15)


def is_midnight(stamp):
    return stamp.hour == 0 and stamp.minute == 0


def context_observations(source, partitions, expected_sha256):
    """Read numerical context only through the last requested target, preserving original dates."""
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise DataValidationError("Original context source checksum differs from the frozen plan.")
    targets = tuple(stamp for partition in partitions for stamp in partition.timestamps)
    records = {}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != list(SOURCE_COLUMNS):
            raise DataValidationError("Original context must retain the 11-column source schema.")
        for line, values in enumerate(reader, 2):
            if len(values) != len(SOURCE_COLUMNS):
                raise DataValidationError("Original context must have exactly 11 cells per row.")
            stamp = datetime.strptime(values[0], "%d/%m/%Y %H:%M")
            if stamp > targets[-1]:
                continue
            if stamp in records:
                raise DataValidationError("Original context contains a duplicate timestamp.")
            records[stamp] = _parse_record(dict(zip(SOURCE_COLUMNS, values, strict=True)), line)
    if any(stamp not in records for stamp in targets):
        raise DataValidationError("Original context is missing a fixed target observation.")
    expected = np.concatenate([partition.target for partition in partitions])
    if not np.allclose([records[stamp].usage_kwh for stamp in targets], expected, rtol=0, atol=1e-9):
        raise DataValidationError("Original context target values differ from the fixed sample.")
    return tuple(AvailableObservation(row, row.timestamp + DELAY) for row in records.values()
                 if not is_midnight(row.timestamp))


def context_inputs(source, partitions, expected_sha256):
    targets = tuple(stamp for partition in partitions for stamp in partition.timestamps)
    return build_forecast_inputs(targets, context_observations(source, partitions, expected_sha256))
