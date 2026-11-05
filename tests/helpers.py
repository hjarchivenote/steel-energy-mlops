"""Small synthetic observations keep tests independent of the private local DVC cache."""

import csv
from datetime import datetime
from pathlib import Path

from steel_energy_mlops.data import SOURCE_COLUMNS, WEEKDAYS


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
