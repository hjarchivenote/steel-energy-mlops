"""Translate artifact deserialization failures into an unavailable-model error."""

from collections.abc import Callable
from typing import Any

from steel_energy_mlops.data import DataValidationError


def load_checked_model(loader: Callable[[str], Any], uri: str) -> Any:
    try:
        return loader(uri)
    except Exception as exc:
        # Archive, serializer and estimator-version failures do not share a library exception type.
        raise DataValidationError("Stored model could not be loaded; verify its artifacts and dependencies.") from exc
