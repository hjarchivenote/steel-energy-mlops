"""Read saved monitoring results for the deployed model and render a Streamlit view."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st

from steel_energy_mlops.data import DataValidationError
from steel_energy_mlops.monitoring import DRIFT_THRESHOLD, MINIMUM_ROWS, MONITORED_COLUMNS
from steel_energy_mlops.tracking import champion_version, require_compatible_version, validate_model_name

LOGGER = logging.getLogger(__name__)
LABELS = dict(zip(MONITORED_COLUMNS, (
    "Time of day · sine", "Time of day · cosine", "Weekday · sine", "Weekday · cosine",
    "Month · sine", "Month · cosine", "Weekend", "Predicted consumption",
), strict=True))


@dataclass(frozen=True)
class DashboardView:
    model_name: str
    model_version: str | None
    state: str
    summary: dict[str, Any] | None = None
    report_html: str | None = None
    stale: bool = False


def _validate_summary(summary: Any, run_id: str, model_name: str, version: str, checksum: str) -> datetime:
    try:
        if (not isinstance(summary, dict) or summary["run_id"] != run_id or summary["model_name"] != model_name
                or summary["model_version"] != version or summary["train_sha256"] != checksum
                or summary["minimum_rows_per_dataset"] != MINIMUM_ROWS or summary["drift_threshold"] != DRIFT_THRESHOLD
                or type(summary["run_limit_reached"]) is not bool or type(summary["max_runs"]) is not int
                or not 1 <= summary["max_runs"] <= 1000
                or any(summary[key] is not False for key in
                       ("performance_evaluated", "test_evaluated", "registry_mutated"))):
            raise ValueError("Incompatible monitoring provenance.")
        counts = [summary[key] for key in ("raw_rows", "unique_rows", "duplicate_rows_removed", "reference_rows")]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Invalid monitoring counts.")
        raw, unique, duplicate, reference = counts
        if unique + duplicate != raw:
            raise ValueError("Inconsistent monitoring counts.")
        sufficient = min(unique, reference) >= MINIMUM_ROWS
        scores, drifted, status = summary["column_scores"], summary["drifted_columns"], summary["status"]
        if status == "insufficient_data":
            if sufficient or scores != {} or drifted != [] or summary["html_artifact"] is not None:
                raise ValueError("Insufficient data cannot have a drift verdict.")
        else:
            if (not sufficient or status not in ("distribution_changed", "no_drift_detected")
                    or not isinstance(scores, dict) or set(scores) != set(MONITORED_COLUMNS)
                    or any(type(score) not in (int, float) or not math.isfinite(score) or score < 0
                           for score in scores.values())
                    or not isinstance(drifted, list) or not all(isinstance(name, str) for name in drifted)
                    or len(drifted) != len(set(drifted))
                    or set(drifted) != {name for name, score in scores.items() if score >= DRIFT_THRESHOLD}
                    or bool(drifted) != (status == "distribution_changed")
                    or summary["html_artifact"] != f"runs:/{run_id}/drift_report.html"):
                raise ValueError("Inconsistent monitoring verdict.")
        since = datetime.fromisoformat(summary["since_request_time"])
        as_of = datetime.fromisoformat(summary["as_of_request_time"])
        if (since.tzinfo is None or as_of.tzinfo is None or since >= as_of
                or as_of > datetime.now(timezone.utc) + timedelta(minutes=5)):
            raise ValueError("Invalid monitoring snapshot times.")
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError("Saved monitoring summary is invalid or incompatible.") from exc
    return as_of


def load_dashboard(workspace: Path, model_name: str) -> DashboardView:
    """Use an explicit client, without creating runs, changing global tracking state, or loading a model."""
    validate_model_name(model_name)
    database = workspace.resolve() / "mlflow.db"
    if not database.is_file():
        return DashboardView(model_name, None, "no_store")
    uri = "sqlite:///" + str(database)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    version = champion_version(client, model_name)
    if version is None:
        return DashboardView(model_name, None, "no_model")
    require_compatible_version(version)
    number = str(version.version)
    checksum = version.tags.get("train_sha256", "")
    experiment = client.get_experiment_by_name("steel-energy-monitoring")
    if experiment is None:
        return DashboardView(model_name, number, "no_snapshot")
    runs = client.search_runs([experiment.experiment_id], max_results=1,
                              filter_string=f"params.model_name = '{model_name}' AND params.model_version = '{number}' "
                                            f"AND params.train_sha256 = '{checksum}' "
                                            "AND tags.operation = 'request_distribution_monitoring'",
                              order_by=["attributes.start_time DESC", "attributes.run_id ASC"])
    if not runs:
        return DashboardView(model_name, number, "no_snapshot")
    run = runs[0]
    if run.info.status != "FINISHED":
        state = "monitoring_running" if run.info.status == "RUNNING" else "monitoring_failed"
        return DashboardView(model_name, number, state)
    summary = json.loads(Path(client.download_artifacts(run.info.run_id, "monitoring_summary.json")).read_text())
    as_of = _validate_summary(summary, run.info.run_id, model_name, number, checksum)
    report = None
    if summary["html_artifact"] is not None:
        report = Path(client.download_artifacts(run.info.run_id, "drift_report.html")).read_text(encoding="utf-8")
        if "<html" not in report.lower():
            raise DataValidationError("Saved Evidently report is not a complete HTML document.")
    return DashboardView(model_name, number, summary["status"], summary, report,
                         datetime.now(timezone.utc) - as_of > timedelta(hours=24))


def render_dashboard(workspace: Path, model_name: str) -> None:
    st.set_page_config(page_title="Steel Energy · Monitoring", page_icon="⚡", layout="wide")
    st.title("Steel Energy · Monitoring")
    st.caption("Request schedules and prediction distributions · Saved monitoring snapshots")
    st.button("Refresh saved results", key="refresh")
    try:
        view = load_dashboard(workspace, model_name)
    except (OSError, ValueError, MlflowException, SQLAlchemyError):
        LOGGER.exception("Could not load saved monitoring results")
        st.error("Monitoring results are unavailable. Check the monitoring service before relying on a drift verdict.")
        return
    st.caption(f"Model: {view.model_name} · Version: {view.model_version or 'unavailable'}")
    if view.summary is None:
        messages = {
            "no_store": "No saved project results are available yet.",
            "no_model": "No approved prediction model is available yet.",
            "no_snapshot": "No monitoring snapshot is available for the deployed model yet.",
            "monitoring_running": "A monitoring report is being generated. Refresh after it finishes.",
            "monitoring_failed": "The latest monitoring attempt failed. Earlier results are not shown as current.",
        }
        show = st.error if view.state == "monitoring_failed" else st.info
        show(messages[view.state])
        return
    summary = view.summary
    as_of = datetime.fromisoformat(summary["as_of_request_time"]).astimezone(timezone.utc)
    st.subheader("Saved snapshot")
    st.caption(f"As of {as_of:%Y-%m-%d %H:%M UTC} · Refresh reloads saved results; it does not generate a new report.")
    if view.stale:
        st.warning("This snapshot is more than 24 hours old. Generate a fresh monitoring report before taking action.")
    if summary["run_limit_reached"]:
        st.warning(f"This snapshot covers only the newest {summary['max_runs']} matching prediction requests.")
    columns = st.columns(4)
    labels = ("Unique prediction times", "Minimum required", "Reference rows", "Repeated records removed")
    values = (summary["unique_rows"], summary["minimum_rows_per_dataset"],
              summary["reference_rows"], summary["duplicate_rows_removed"])
    for column, label, value in zip(columns, labels, values, strict=True):
        column.metric(label, value, border=True)
    if view.state == "insufficient_data":
        st.warning("Not enough data to assess drift. Both datasets need at least 100 rows after deduplicating "
                   "prediction timestamps. No drift verdict or Evidently report is available for this snapshot.")
    elif view.state == "distribution_changed":
        st.warning("Request or prediction distributions changed. Review the schedule and flagged columns.")
    else:
        st.info("No distribution change was detected in this snapshot. This does not assess prediction accuracy.")
    st.info("Calendar-only model: these results describe requested times and predicted consumption. "
            "Observed factory consumption is not available here, so prediction accuracy and factory-load changes "
            "are not assessed. A change signal calls for investigation.")
    if view.report_html is not None:
        st.subheader("Distribution checks")
        st.dataframe({"Column": [LABELS[name] for name in MONITORED_COLUMNS],
                      "Distance": [summary["column_scores"][name] for name in MONITORED_COLUMNS],
                      "Review": ["Yes" if name in summary["drifted_columns"] else "No" for name in MONITORED_COLUMNS]},
                     hide_index=True, width="stretch")
        st.subheader("Evidently report")
        st.iframe(view.report_html, height=900, alt="Evidently distribution drift report")
    with st.expander("Snapshot details"):
        st.write({"Request window start (UTC)": summary["since_request_time"],
                  "Request window end (UTC)": summary["as_of_request_time"], "Monitoring run": summary["run_id"],
                  "Distance threshold": summary["drift_threshold"]})
