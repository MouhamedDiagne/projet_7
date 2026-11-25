"""Local API tests that use a deterministic fake release instead of Databricks."""

from pathlib import Path

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import ApiSettings
from app.database import PredictionDatabase
from app.main import create_app
from app.model_store import LoadedRelease
from app.schemas import FeedbackRequest, PredictionRequest


class FakePredictor:
    """Return a fixed positive probability for isolated HTTP tests."""

    def predict_positive_probability(self, tweet: str) -> float:
        """Return a deterministic value while asserting the API supplies text."""
        assert tweet
        return 0.82


def ready_app(tmp_path: Path, mode: str = 'light'):
    """Build an initialized API state without opening network sockets in tests."""
    settings = ApiSettings(
        model_mode=mode,
        database_path=tmp_path / 'predictions.sqlite3',
        mlflow_tracking_uri=None,
        mlflow_experiment_id=None,
        light_run_name='tfidf-logreg-benchmark',
        advanced_run_name='distilbert-frozen-classifier-benchmark',
        light_run_id=None,
        advanced_run_id=None,
        model_artifact_path='selected_model',
    )
    release = LoadedRelease(FakePredictor(), mode, 'run-123', settings.selected_run_name, 0.7)
    app = create_app(settings, release_loader=lambda _: release)
    database = PredictionDatabase(settings.database_path)
    database.initialize()
    app.state.database = database
    app.state.release = release
    return app, settings


def endpoint(app, path: str):
    """Retrieve a synchronous FastAPI endpoint for a focused unit test."""
    return next(route.endpoint for route in app.routes if getattr(route, 'path', None) == path)


def test_prediction_feedback_and_sqlite_creation(tmp_path: Path):
    """Create the database on startup and link feedback to one prediction."""
    app, settings = ready_app(tmp_path)
    request = SimpleNamespace(app=app)
    assert endpoint(app, '/health')(request)['model_run_id'] == 'run-123'

    prediction = endpoint(app, '/predict')(PredictionRequest(tweet='I love this flight'), request)
    assert prediction.sentiment == 'positive'
    assert prediction.decision_threshold == 0.7

    feedback = endpoint(app, '/feedback')(
        FeedbackRequest(prediction_id=prediction.prediction_id, user_expected_sentiment='negative'), request,
    )
    assert feedback.prediction_id == prediction.prediction_id

    assert settings.database_path.is_file()


def test_rejects_invalid_tweet_and_unknown_feedback(tmp_path: Path):
    """Reject empty requests and feedback that does not match a stored prediction."""
    app, _ = ready_app(tmp_path, mode='advanced')
    request = SimpleNamespace(app=app)
    with pytest.raises(ValidationError):
        PredictionRequest(tweet='')
    with pytest.raises(HTTPException) as error:
        endpoint(app, '/feedback')(FeedbackRequest(prediction_id='unknown'), request)
    assert error.value.status_code == 404
