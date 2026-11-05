import json
import runpy
import sys
from pathlib import Path

import pytest

from steel_energy_mlops.cli import main


def test_cli_prepares_data(dataset_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "prepared"
    result = main(["prepare-data", "--input", str(dataset_path), "--output", str(output), "--expected-rows", "20"])
    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert json.loads(captured.out)["splits"]["test"]["row_count"] == 3


@pytest.mark.parametrize("failure", ["missing", "wrong-count", "encoding", "quoting"])
def test_cli_reports_actionable_input_errors(
    dataset_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    path = dataset_path
    if failure == "missing":
        path = tmp_path / "missing.csv"
    elif failure == "encoding":
        path.write_bytes(b"\xff\xff")
    elif failure == "quoting":
        path.write_bytes(path.read_bytes().splitlines()[0] + b'\n"unclosed')
    result = main(["prepare-data", "--input", str(path), "--output", str(tmp_path / "output")])
    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "Data preparation failed:" in captured.err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("arguments", [[], ["unknown"], ["prepare-data", "--expected-rows", "text"]])
def test_cli_rejects_invalid_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_module_entry_point(
    dataset_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["steel-energy", "prepare-data", "--input", str(dataset_path), "--output", str(tmp_path / "out"),
         "--expected-rows", "20"],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_module("steel_energy_mlops", run_name="__main__")
    assert error.value.code == 0
    assert json.loads(capsys.readouterr().out)["source"]["row_count"] == 20
