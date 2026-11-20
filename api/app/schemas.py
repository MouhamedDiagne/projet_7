"""HTTP request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """One tweet submitted for sentiment prediction."""

    tweet: str = Field(min_length=1, max_length=2_000, examples=['This flight was delayed again.'])


class PredictionResponse(BaseModel):
    """A prediction and the immutable model release that produced it."""

    prediction_id: str
    sentiment: Literal['negative', 'positive']
    positive_probability: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(gt=0.0, lt=1.0)
    model_mode: Literal['light', 'advanced']
    model_run_id: str
    model_run_name: str
    inference_time_ms: float = Field(ge=0.0)


class FeedbackRequest(BaseModel):
    """A user report that an earlier prediction should be reviewed."""

    prediction_id: str = Field(min_length=1)
    user_expected_sentiment: Literal['negative', 'positive'] | None = None
    is_incorrect: bool = True


class FeedbackResponse(BaseModel):
    """Confirmation that feedback was stored locally."""

    feedback_id: int
    prediction_id: str
    created_at: datetime
