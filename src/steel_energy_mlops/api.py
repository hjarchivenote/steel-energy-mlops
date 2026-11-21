"""Local prediction service with strict input contracts and correlated HTTP/MLflow records."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from filelock import FileLock, Timeout
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException

from steel_energy_mlops.prediction import model_status, predict_consumption
from steel_energy_mlops.modeling import LEARNED_FAMILIES
from steel_energy_mlops.promotion import MINIMUM_IMPROVEMENT
from steel_energy_mlops.tracking import DEFAULT_MODEL, DEFAULT_WORKSPACE, validate_model_name
from steel_energy_mlops.training_jobs import TrainingBusyError, TrainingJobs, UnknownJobError

LOGGER = logging.getLogger(__name__)
# MLflow fluent tracking configuration and native thread limits have process-wide state.
# Serialize API model operations across app instances; HTTP validation and liveness remain independent.
MODEL_OPERATION_LOCK = Lock()


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise ValueError("Use an ISO datetime string such as 2019-01-07T10:00:00.")
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Use a valid ISO datetime string.") from exc
    if stamp.tzinfo is not None:
        raise ValueError("Use timezone-naive source time; timezone conversion is not supported.")
    if stamp.minute % 15 or stamp.second or stamp.microsecond:
        raise ValueError("Use a timestamp on the 15-minute grid with zero seconds and microseconds.")
    return stamp


Timestamp = Annotated[datetime, BeforeValidator(_timestamp, json_schema_input_type=str)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamps: list[Timestamp] = Field(
        min_length=1, max_length=1000,
        description="1-1000 ISO datetime strings in source local time, without timezone, on the 15-minute grid.",
        examples=[["2019-01-07T10:00:00", "2019-01-07T10:15:00"]],
    )


class PredictionPoint(BaseModel):
    timestamp: datetime
    usage_kwh: float = Field(ge=0, allow_inf_nan=False)


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    promotion_policy: str = Field(description="Policy at original promotion; legacy-v1 is not v2 certification.")
    feature_contract: str
    unit: Literal["kWh"]


class PredictionResponse(ModelInfo):
    request_id: str
    run_id: str
    predictions: list[PredictionPoint] = Field(min_length=1, max_length=1000)


class ReadyResponse(ModelInfo):
    status: Literal["ready"] = "ready"
    request_id: str


class ValidationIssue(BaseModel):
    location: list[str | int]
    message: str
    kind: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    issues: list[ValidationIssue] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail
    job_id: str | None = None
    status_url: str | None = None


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    families: list[Literal["ridge", "random_forest", "hist_gradient_boosting"]] = Field(
        default_factory=lambda: list(LEARNED_FAMILIES), min_length=1, max_length=3,
    )
    forest_trees: int = Field(default=100, ge=1, le=500)
    minimum_improvement: float = Field(default=MINIMUM_IMPROVEMENT, ge=MINIMUM_IMPROVEMENT, lt=1, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2 ** 32 - 1)

    @field_validator("families")
    @classmethod
    def unique_families(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("Training families must be unique.")
        return values


class TrainingJobResponse(BaseModel):
    request_id: str
    job_id: str
    training_request_id: str
    model_name: str
    status: Literal["running", "succeeded", "failed", "interrupted"]
    status_url: str
    created_at: datetime
    updated_at: datetime
    parameters: TrainingRequest
    train_sha256: str
    validation_sha256: str
    result: dict[str, Any] | None
    error: dict[str, Any] | None


def _error(
    request: Request, status: int, code: str, message: str, issues: list[ValidationIssue] | None = None
) -> JSONResponse:
    request.state.error_code = code
    job_id = getattr(request.state, "training_job_id", None)
    body = ErrorResponse(request_id=request.state.request_id, job_id=job_id,
                         status_url=None if job_id is None else f"/train/{job_id}",
                         error=ErrorDetail(code=code, message=message, issues=issues or []))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json", exclude_none=True))


def append_audit(path: Path, record: dict[str, Any]) -> None:
    with FileLock(path.with_suffix(".lock"), timeout=5):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def create_app(
    workspace: Path = DEFAULT_WORKSPACE, model_name: str = DEFAULT_MODEL,
    training_data: Path = Path("data/processed"),
) -> FastAPI:
    validate_model_name(model_name)
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    audit_path = workspace / "http_requests.jsonl"
    jobs = TrainingJobs(workspace, training_data, model_name)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            await asyncio.to_thread(jobs.close)

    app = FastAPI(
        title="Steel Energy MLOps API", version="0.1.0", lifespan=lifespan,
        responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        description=(
            "Calendar-only electricity consumption estimates. Source local time is timezone-naive. "
            "The model promotion policy is disclosed; a legacy model is not certified under the current policy."
        ),
    )

    @app.middleware("http")
    async def audit_request(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        started = perf_counter()
        request.state.request_id = uuid4().hex
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception("Unhandled API failure for request %s", request.state.request_id)
            response = _error(request, 500, "internal_error", "An unexpected server error occurred.")
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "request_id": request.state.request_id,
            "method": request.method, "path": request.url.path, "status_code": response.status_code,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "run_id": getattr(request.state, "run_id", None),
            "model_version": getattr(request.state, "model_version", None),
            "error_code": getattr(request.state, "error_code", None),
            "training_job_id": getattr(request.state, "training_job_id", None),
        }
        try:
            await asyncio.to_thread(append_audit, audit_path, record)
        except (OSError, Timeout):
            LOGGER.exception("HTTP audit storage failed for request %s", request.state.request_id)
            response = _error(request, 503, "audit_unavailable", "Request recording is unavailable; retry later.")
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        issues = [ValidationIssue(location=list(error["loc"]), message=error["msg"], kind=error["type"])
                  for error in exc.errors()[:20]]
        return _error(request, 422, "invalid_request", "Check the request body and timestamp format.", issues)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        response = _error(request, exc.status_code, "http_error", str(exc.detail))
        response.headers.update(exc.headers or {})
        return response

    @app.get("/health", summary="Check that the HTTP service is running")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyResponse, responses={503: {"model": ErrorResponse}},
             summary="Check champion compatibility and that its saved model can be loaded")
    def ready(request: Request) -> ReadyResponse | JSONResponse:
        try:
            with MODEL_OPERATION_LOCK:
                metadata = model_status(workspace, model_name)
            request.state.model_version = metadata["model_version"]
            return ReadyResponse.model_validate({"request_id": request.state.request_id, **metadata})
        except (ValueError, MlflowException, OSError):
            LOGGER.exception("Model readiness failed for request %s", request.state.request_id)
            return _error(request, 503, "model_unavailable", "No compatible, loadable champion is available.")

    @app.post("/predict", response_model=PredictionResponse,
              responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
              summary="Estimate electricity consumption using one pinned champion version")
    def predict(body: PredictionRequest, request: Request) -> PredictionResponse | JSONResponse:
        try:
            with MODEL_OPERATION_LOCK:
                output = predict_consumption(tuple(body.timestamps), workspace, model_name,
                                             request_id=request.state.request_id)
        except (ValueError, MlflowException, OSError):
            LOGGER.exception("Prediction failed for request %s", request.state.request_id)
            return _error(request, 503, "prediction_unavailable",
                          "A compatible model and working prediction storage are required; check /ready.")
        request.state.run_id = output["run_id"]
        request.state.model_version = output["model_version"]
        return PredictionResponse.model_validate(output)

    def job_response(request: Request, record: dict[str, Any]) -> TrainingJobResponse:
        request.state.training_job_id = record["job_id"]
        if record["result"] is not None:
            request.state.run_id = record["result"]["run_id"]
        return TrainingJobResponse.model_validate({**record, "request_id": request.state.request_id,
                                                   "status_url": f"/train/{record['job_id']}"})

    @app.post("/train", status_code=202, response_model=TrainingJobResponse,
              responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
              summary="Start one training job on the server-configured data snapshot")
    def train(body: TrainingRequest, request: Request, response: Response) -> TrainingJobResponse | JSONResponse:
        try:
            record = jobs.start(body.model_dump(), request.state.request_id)
        except (TrainingBusyError, Timeout):
            return _error(request, 409, "training_busy", "Another training job is active; wait for it to finish.")
        except (ValueError, OSError, RuntimeError):
            LOGGER.exception("Training admission failed for request %s", request.state.request_id)
            return _error(request, 503, "training_unavailable", "Training inputs or job storage are unavailable.")
        response.headers["Location"] = f"/train/{record['job_id']}"
        return job_response(request, record)

    @app.get("/train/{job_id}", response_model=TrainingJobResponse, responses={404: {"model": ErrorResponse}},
             summary="Read training progress, final metrics, and the promotion decision")
    def training_status(job_id: str, request: Request) -> TrainingJobResponse | JSONResponse:
        try:
            return job_response(request, jobs.get(job_id))
        except UnknownJobError:
            return _error(request, 404, "job_not_found", "Unknown training job.")
        except (ValueError, OSError):
            LOGGER.exception("Training status failed for request %s", request.state.request_id)
            return _error(request, 503, "job_storage_unavailable", "Training job status is temporarily unavailable.")

    return app
