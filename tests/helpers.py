"""Small synthetic observations keep tests independent of the private local DVC cache."""

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

from steel_energy_mlops.data import SOURCE_COLUMNS, WEEKDAYS
from steel_energy_mlops.preparation import prepare_dataset


def make_row(timestamp: datetime, usage: float = 10.0) -> list[str]:
    return [
        timestamp.strftime("%d/%m/%Y %H:%M"),
        str(usage),
        "2.0",
        "1.0",
        "0.01",
        "80.0",
        "90.0",
        str(timestamp.hour * 3600 + timestamp.minute * 60),
        "Weekend" if timestamp.weekday() >= 5 else "Weekday",
        WEEKDAYS[timestamp.weekday()],
        "Light_Load",
    ]


def write_csv(path: Path, rows: list[list[str]], header: tuple[str, ...] = SOURCE_COLUMNS) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def make_training_data(directory: Path) -> Path:
    rows = []
    for index in range(160):
        timestamp = datetime(2018, 1, 1) + timedelta(minutes=15 * index)
        phase = math.tau * (index % 96) / 96
        usage = (8 + 2 * math.sin(phase) + math.cos(phase)) ** 2
        rows.append(make_row(timestamp, usage))
    source = write_csv(directory / "training_source.csv", rows)
    prepared = directory / "prepared"
    prepare_dataset(source, prepared, expected_rows=160)
    return prepared
