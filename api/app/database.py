"""Small SQLite store for local prediction and feedback audit records."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class StoredPrediction:
    """Values persisted after a successful API prediction."""

    prediction_id: str
    created_at: datetime


class FeedbackNotFoundError(LookupError):
    """Raised when feedback references an unknown prediction identifier."""


class PredictionDatabase:
    """Create and query the lightweight SQLite audit store."""

    def __init__(self, path: Path):
        """Keep the database location without opening a long-lived connection."""
        self.path = path

    def initialize(self) -> None:
        """Create the SQLite file and schema when they do not already exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                '''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    tweet TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    positive_probability REAL NOT NULL,
                    decision_threshold REAL NOT NULL,
                    model_mode TEXT NOT NULL,
                    model_run_id TEXT NOT NULL,
                    model_run_name TEXT NOT NULL,
                    inference_time_ms REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_incorrect INTEGER NOT NULL,
                    user_expected_sentiment TEXT,
                    FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
                );
                '''
            )

    def record_prediction(
        self, tweet: str, sentiment: str, probability: float, threshold: float,
        model_mode: str, model_run_id: str, model_run_name: str, inference_time_ms: float,
    ) -> StoredPrediction:
        """Persist one prediction so subsequent feedback refers to a known event."""
        record = StoredPrediction(str(uuid4()), datetime.now(timezone.utc))
        with self._connect() as connection:
            connection.execute(
                '''INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    record.prediction_id, record.created_at.isoformat(), tweet, sentiment,
                    probability, threshold, model_mode, model_run_id, model_run_name,
                    inference_time_ms,
                ),
            )
        return record

    def record_feedback(
        self, prediction_id: str, is_incorrect: bool, user_expected_sentiment: str | None,
    ) -> tuple[int, datetime]:
        """Attach explicit user feedback to an existing prediction record."""
        created_at = datetime.now(timezone.utc)
        with self._connect() as connection:
            found = connection.execute(
                'SELECT 1 FROM predictions WHERE prediction_id = ?', (prediction_id,)
            ).fetchone()
            if found is None:
                raise FeedbackNotFoundError(prediction_id)
            cursor = connection.execute(
                '''INSERT INTO feedback(prediction_id, created_at, is_incorrect, user_expected_sentiment)
                   VALUES (?, ?, ?, ?)''',
                (prediction_id, created_at.isoformat(), int(is_incorrect), user_expected_sentiment),
            )
        return int(cursor.lastrowid), created_at

    def _connect(self) -> sqlite3.Connection:
        """Open one transaction-safe connection with foreign-key checks enabled."""
        connection = sqlite3.connect(self.path)
        connection.execute('PRAGMA foreign_keys = ON')
        return connection
