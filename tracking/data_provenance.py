"""Sentiment140 data identity checks used by the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SENTIMENT140_TRAINING_FILENAME = "training.1600000.processed.noemoticon.csv"
SENTIMENT140_ENCODING = "cp1252"


class DatasetProvenanceError(ValueError):
    """Raised when the local Sentiment140 file is not the expected corpus."""


@dataclass(frozen=True)
class DatasetManifest:
    """Minimal, non-sensitive dataset identity suitable for MLflow logging."""

    filename: str
    byte_size: int
    encoding: str = SENTIMENT140_ENCODING

    def as_mlflow_params(self) -> dict[str, str | int]:
        return {
            "dataset_filename": self.filename,
            "dataset_byte_size": self.byte_size,
            "dataset_encoding": self.encoding,
        }


def verify_sentiment140_training_file(path: str | Path) -> DatasetManifest:
    """Verify the expected filename and return its lightweight manifest."""

    dataset_path = Path(path)
    if dataset_path.name != SENTIMENT140_TRAINING_FILENAME:
        raise DatasetProvenanceError(
            f"Expected `{SENTIMENT140_TRAINING_FILENAME}`, received `{dataset_path.name}`."
        )
    if not dataset_path.is_file():
        raise DatasetProvenanceError(f"Dataset file does not exist: {dataset_path}")

    return DatasetManifest(
        filename=dataset_path.name,
        byte_size=dataset_path.stat().st_size,
    )
