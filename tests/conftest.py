from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.helpers import make_row, make_training_data, write_csv


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    start = datetime(2018, 1, 1)
    return write_csv(tmp_path / "source.csv", [make_row(start + timedelta(minutes=15 * i), i) for i in range(20)])


@pytest.fixture
def prepared_training(tmp_path: Path) -> Path:
    return make_training_data(tmp_path)
