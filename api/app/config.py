"""Environment-driven configuration for the prediction API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - deployment always installs python-dotenv.
    load_dotenv = None


ModelMode = Literal['light', 'advanced']


@dataclass(frozen=True)
class ApiSettings:
    """Capture the selected model release and non-secret API settings."""

    model_mode: ModelMode
    database_path: Path
    mlflow_tracking_uri: str | None
    mlflow_experiment_id: str | None
    light_run_name: str
    advanced_run_name: str
    light_run_id: str | None
    advanced_run_id: str | None
    model_artifact_path: str

    @property
    def selected_run_name(self) -> str:
        """Return the configured parent-run name for the active API mode."""
        return self.light_run_name if self.model_mode == 'light' else self.advanced_run_name

    @property
    def selected_run_id(self) -> str | None:
        """Return an optional exact run pin for the active API mode."""
        return self.light_run_id if self.model_mode == 'light' else self.advanced_run_id

    @classmethod
    def from_environment(cls) -> 'ApiSettings':
        """Read settings without loading a model or exposing credentials."""
        if load_dotenv is not None:
            load_dotenv(override=False)
        mode = os.getenv('API_MODEL_MODE', 'light').strip().lower()
        if mode not in {'light', 'advanced'}:
            raise ValueError('API_MODEL_MODE must be light or advanced.')
        root = Path(__file__).resolve().parents[2]
        raw_database_path = os.getenv('API_SQLITE_PATH', 'api/data/sentiment_api.sqlite3')
        database_path = Path(raw_database_path)
        if not database_path.is_absolute():
            database_path = root / database_path
        return cls(
            model_mode=mode,  # type: ignore[arg-type]
            database_path=database_path,
            mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
            mlflow_experiment_id=os.getenv('MLFLOW_EXPERIMENT_ID'),
            light_run_name=os.getenv('LIGHT_MODEL_RUN_NAME', 'tfidf-logreg-benchmark'),
            advanced_run_name=os.getenv(
                'ADVANCED_MODEL_RUN_NAME', 'distilbert-frozen-classifier-benchmark'
            ),
            light_run_id=os.getenv('LIGHT_MODEL_RUN_ID') or None,
            advanced_run_id=os.getenv('ADVANCED_MODEL_RUN_ID') or None,
            model_artifact_path=os.getenv('MODEL_ARTIFACT_PATH', 'selected_model'),
        )
