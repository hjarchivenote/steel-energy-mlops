"""Damaged or incompatible artifacts must reach the application's model error boundary."""

from zipfile import BadZipFile

import pytest

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.model_io import load_checked_model


@pytest.mark.parametrize("failure", [BadZipFile, EOFError, ImportError, TypeError, OSError, ValueError])
def test_deserializer_failures_keep_the_cause_without_exposing_artifact_details(failure):
    def loader(uri):
        assert uri == "models:/pinned/1"
        raise failure("private path")
    with pytest.raises(DataValidationError, match="Stored model") as caught:
        load_checked_model(loader, "models:/pinned/1")
    assert isinstance(caught.value.__cause__, failure)
    assert "private path" not in str(caught.value)


def test_successful_deserialization_preserves_the_loaded_instance():
    sentinel = object()
    assert load_checked_model(lambda uri: sentinel, "models:/pinned/1") is sentinel
