"""FastAPI application exposing one MLflow-selected sentiment model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, status

from .config import ApiSettings
from .database import FeedbackNotFoundError, PredictionDatabase
from .model_store import LoadedRelease, load_release_from_mlflow
from .schemas import FeedbackRequest, FeedbackResponse, PredictionRequest, PredictionResponse


def create_app(
    settings: ApiSettings | None = None,
    release_loader: Callable[[ApiSettings], LoadedRelease] = load_release_from_mlflow,
) -> FastAPI:
    """Create the API, loading one configured model release during startup."""
    configured_settings = settings or ApiSettings.from_environment()
    database = PredictionDatabase(configured_settings.database_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize SQLite before loading the model exactly once per process."""
        database.initialize()
        app.state.database = database
        app.state.release = release_loader(configured_settings)
        yield

    app = FastAPI(
        title='Sentiment API',
        version='1.0.0',
        description='Predict tweet sentiment from a benchmark model stored in MLflow.',
        lifespan=lifespan,
    )

    @app.get('/health')
    def health(request: Request) -> dict:
        """Expose the selected immutable model release without serving a prediction."""
        release = request.app.state.release
        return {
            'status': 'ok',
            'model_mode': release.model_mode,
            'model_run_id': release.run_id,
            'model_run_name': release.run_name,
            'decision_threshold': release.threshold,
        }

    @app.post('/predict', response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        """Predict one tweet and store an auditable SQLite prediction record."""
        release = request.app.state.release
        start = perf_counter()
        probability = release.predictor.predict_positive_probability(payload.tweet)
        elapsed_ms = (perf_counter() - start) * 1_000
        sentiment = 'positive' if probability >= release.threshold else 'negative'
        stored = request.app.state.database.record_prediction(
            tweet=payload.tweet,
            sentiment=sentiment,
            probability=probability,
            threshold=release.threshold,
            model_mode=release.model_mode,
            model_run_id=release.run_id,
            model_run_name=release.run_name,
            inference_time_ms=elapsed_ms,
        )
        return PredictionResponse(
            prediction_id=stored.prediction_id,
            sentiment=sentiment,
            positive_probability=probability,
            decision_threshold=release.threshold,
            model_mode=release.model_mode,
            model_run_id=release.run_id,
            model_run_name=release.run_name,
            inference_time_ms=elapsed_ms,
        )

    @app.post('/feedback', response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
    def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
        """Store explicit user feedback linked to an existing prediction."""
        try:
            feedback_id, created_at = request.app.state.database.record_feedback(
                prediction_id=payload.prediction_id,
                is_incorrect=payload.is_incorrect,
                user_expected_sentiment=payload.user_expected_sentiment,
            )
        except FeedbackNotFoundError as error:
            raise HTTPException(status_code=404, detail='Prediction identifier was not found.') from error
        return FeedbackResponse(
            feedback_id=feedback_id,
            prediction_id=payload.prediction_id,
            created_at=created_at,
        )

    return app


app = create_app()
