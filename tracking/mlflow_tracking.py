"""Configure MLflow tracking for either a local run or remote Databricks run."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import urlparse

import mlflow
from mlflow import MlflowClient


DATABRICKS_REQUIRED_SETTINGS = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "MLFLOW_EXPERIMENT_ID",
)
DEFAULT_DATABRICKS_TRACKING_URI = "databricks"
DEFAULT_LOCAL_EXPERIMENT_NAME = "sentiment140-benchmark-v1-local"
VALID_TRACKING_MODES = {"local", "databricks"}


class TrackingConfigurationError(RuntimeError):
    """Raised when MLflow tracking cannot be configured safely."""


@dataclass(frozen=True)
class TrackingConfig:
    """Resolved MLflow backend and experiment, without exposing credentials."""

    backend: str
    experiment_id: str
    experiment_name: str
    tracking_uri: str
    host: str | None = None
    token: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class SmokeTestResult:
    """Identifiers returned after a successful end-to-end MLflow smoke test."""

    backend: str
    experiment_id: str
    run_id: str
    artifact_path: str


def _is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_kaggle_secret(name: str) -> str | None:
    """Read one Kaggle secret, returning None when it is unavailable."""

    if not _is_kaggle():
        return None

    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
    except Exception:
        # A concise validation error is raised later. Secret manager exceptions
        # are deliberately not surfaced because they can disclose secret details.
        return None
    return value.strip() if value else None


def _read_setting(
    name: str,
    environment: Mapping[str, str] | None,
    secret_reader: Callable[[str], str | None],
) -> str | None:
    source = os.environ if environment is None else environment
    value = source.get(name)
    if value and value.strip():
        return value.strip()
    return secret_reader(name)


def _load_dotenv_when_available(environment: Mapping[str, str] | None) -> None:
    if environment is not None:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


def _resolve_tracking_mode(
    environment: Mapping[str, str] | None,
    secret_reader: Callable[[str], str | None],
) -> str:
    requested_mode = _read_setting("MLFLOW_TRACKING_MODE", environment, secret_reader)
    if requested_mode:
        mode = requested_mode.lower()
        if mode not in VALID_TRACKING_MODES:
            valid_modes = ", ".join(sorted(VALID_TRACKING_MODES))
            raise TrackingConfigurationError(
                f"MLFLOW_TRACKING_MODE must be one of: {valid_modes}."
            )
    elif _is_kaggle():
        mode = "databricks"
    else:
        # Existing local configurations that already provide the three
        # Databricks values keep their previous remote behavior.
        configured_values = [
            _read_setting(name, environment, secret_reader)
            for name in DATABRICKS_REQUIRED_SETTINGS
        ]
        mode = "databricks" if all(configured_values) else "local"

    if _is_kaggle() and mode != "databricks":
        raise TrackingConfigurationError(
            "Kaggle runs must use MLFLOW_TRACKING_MODE=databricks so benchmark "
            "experiments are persisted remotely."
        )
    return mode


def _configure_local_tracking(
    environment: Mapping[str, str] | None,
    secret_reader: Callable[[str], str | None],
) -> TrackingConfig:
    local_directory = _read_setting("MLFLOW_LOCAL_DIR", environment, secret_reader)
    artifact_root = Path(local_directory).expanduser() if local_directory else _project_root() / "mlruns"
    if not artifact_root.is_absolute():
        artifact_root = _project_root() / artifact_root
    artifact_root = artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    experiment_name = (
        _read_setting("MLFLOW_EXPERIMENT_NAME", environment, secret_reader)
        or DEFAULT_LOCAL_EXPERIMENT_NAME
    )
    # MLflow 3 no longer enables its filesystem metadata store by default.
    # SQLite remains fully local while preserving a stable database backend;
    # artifacts stay in a human-accessible directory next to that database.
    database_path = artifact_root / "mlflow.db"
    tracking_uri = f"sqlite:///{database_path.as_posix()}"
    artifact_location = (artifact_root / "artifacts").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
        experiment = client.get_experiment(experiment_id)
    mlflow.set_experiment(experiment.name)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_ID"] = experiment.experiment_id
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment.name

    return TrackingConfig(
        backend="local",
        experiment_id=experiment.experiment_id,
        experiment_name=experiment.name,
        tracking_uri=tracking_uri,
    )


def _load_databricks_values(
    environment: Mapping[str, str] | None,
    secret_reader: Callable[[str], str | None],
) -> tuple[str, str, str, str]:
    values = {
        name: _read_setting(name, environment, secret_reader)
        for name in DATABRICKS_REQUIRED_SETTINGS
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        missing_values = ", ".join(missing)
        raise TrackingConfigurationError(
            "Databricks MLflow is not configured. Add these values as Kaggle Secrets "
            f"or environment variables: {missing_values}."
        )

    host = values["DATABRICKS_HOST"].rstrip("/")
    parsed_host = urlparse(host)
    if parsed_host.scheme != "https" or not parsed_host.netloc:
        raise TrackingConfigurationError(
            "DATABRICKS_HOST must be a complete HTTPS workspace URL, for example "
            "https://<workspace>.cloud.databricks.com."
        )
    experiment_id = values["MLFLOW_EXPERIMENT_ID"]
    if not experiment_id.isdigit():
        raise TrackingConfigurationError("MLFLOW_EXPERIMENT_ID must be the numeric MLflow experiment ID.")
    tracking_uri = (
        _read_setting("MLFLOW_TRACKING_URI", environment, secret_reader)
        or DEFAULT_DATABRICKS_TRACKING_URI
    )
    return host, values["DATABRICKS_TOKEN"], experiment_id, tracking_uri


def configure_tracking(
    environment: Mapping[str, str] | None = None,
    secret_reader: Callable[[str], str | None] = _read_kaggle_secret,
) -> TrackingConfig:
    """Configure and select the correct MLflow backend for the current environment.

    - Local runs use `./mlruns` by default and require no account or secret.
    - Kaggle runs always require Databricks credentials and a remote experiment.
    """

    _load_dotenv_when_available(environment)
    mode = _resolve_tracking_mode(environment, secret_reader)
    if mode == "local":
        return _configure_local_tracking(environment, secret_reader)

    host, token, experiment_id, tracking_uri = _load_databricks_values(
        environment, secret_reader
    )
    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_ID"] = experiment_id
    mlflow.set_tracking_uri(tracking_uri)

    experiment = MlflowClient().get_experiment(experiment_id)
    if experiment is None:
        raise TrackingConfigurationError(
            f"No MLflow experiment was found for MLFLOW_EXPERIMENT_ID={experiment_id}."
        )
    if experiment.lifecycle_stage != "active":
        raise TrackingConfigurationError(
            f"MLflow experiment {experiment_id} is not active ({experiment.lifecycle_stage})."
        )
    return TrackingConfig(
        backend="databricks",
        experiment_id=experiment_id,
        experiment_name=experiment.name,
        tracking_uri=tracking_uri,
        host=host,
        token=token,
    )


def configure_databricks_tracking() -> TrackingConfig:
    """Backward-compatible Databricks-only configuration entry point."""

    config = configure_tracking()
    if config.backend != "databricks":
        raise TrackingConfigurationError(
            "Databricks configuration was requested, but the active backend is local. "
            "Set MLFLOW_TRACKING_MODE=databricks."
        )
    return config


def run_smoke_test(config: TrackingConfig | None = None) -> SmokeTestResult:
    """Log and read back one parameter, metric, and text artifact in the active backend."""

    config = config or configure_tracking()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_name = "tracking_smoke_test.txt"

    with mlflow.start_run(
        experiment_id=config.experiment_id,
        run_name=f"tracking-smoke-test-{timestamp}",
        tags={
            "run_type": "tracking_smoke_test",
            "tracking_backend": config.backend,
            "execution_environment": "kaggle" if _is_kaggle() else "local",
        },
    ) as active_run:
        mlflow.log_param("smoke_test", "true")
        mlflow.log_metric("tracking_connection", 1.0)
        with tempfile.TemporaryDirectory(prefix="mlflow-smoke-") as temporary_directory:
            artifact_path = Path(temporary_directory) / artifact_name
            artifact_path.write_text(
                "MLflow tracking smoke test completed successfully.\n",
                encoding="utf-8",
            )
            mlflow.log_artifact(str(artifact_path))
        run_id = active_run.info.run_id

    artifact_names = {artifact.path for artifact in MlflowClient().list_artifacts(run_id)}
    if artifact_name not in artifact_names:
        raise TrackingConfigurationError(
            "The smoke-test run finished but its artifact could not be read back."
        )
    return SmokeTestResult(
        backend=config.backend,
        experiment_id=config.experiment_id,
        run_id=run_id,
        artifact_path=artifact_name,
    )
