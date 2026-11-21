"""Thin Streamlit entry point; loading and presentation are tested in src/."""

import os
from pathlib import Path

from steel_energy_mlops.dashboard import render_dashboard
from steel_energy_mlops.tracking import DEFAULT_MODEL, DEFAULT_WORKSPACE

render_dashboard(Path(os.environ.get("STEEL_ENERGY_WORKSPACE", str(DEFAULT_WORKSPACE))),
                 os.environ.get("STEEL_ENERGY_MODEL_NAME", DEFAULT_MODEL))
