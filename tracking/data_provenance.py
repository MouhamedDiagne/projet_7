"""Sentiment140 data identity checks used by the benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


SENTIMENT140_TRAINING_FILENAME = "training.1600000.processed.noemoticon.csv"
SENTIMENT140_TRAINING_SHA256 = (
    "08cba6317a49528fcd074f9043aafcd5ad6c6be45ede159c4e36cec33af24afe"
)
SENTIMENT140_ENCODING = "cp1252"


class DatasetProvenanceError(ValueError):
    """Raised when the local Sentiment140 file is not the expected corpus."""


@dataclass(frozen=True)
class DatasetManifest:
    """Minimal, non-sensitive dataset identity suitable for MLflow logging."""

    filename: str
    byte_size: int
    sha256: str
    encoding: str = SENTIMENT140_ENCODING

    def as_mlflow_params(self) -> dict[str, str | int]:
        return {
            "dataset_filename": self.filename,
            "dataset_byte_size": self.byte_size,
            "dataset_sha256": self.sha256,
            "dataset_encoding": self.encoding,
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest without loading the complete CSV into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sentiment140_training_file(path: str | Path) -> DatasetManifest:
    """Verify the expected filename and checksum, then return its manifest."""

    dataset_path = Path(path)
    if dataset_path.name != SENTIMENT140_TRAINING_FILENAME:
        raise DatasetProvenanceError(
            f"Expected `{SENTIMENT140_TRAINING_FILENAME}`, received `{dataset_path.name}`."
        )
    if not dataset_path.is_file():
        raise DatasetProvenanceError(f"Dataset file does not exist: {dataset_path}")

    sha256 = sha256_file(dataset_path)
    if sha256 != SENTIMENT140_TRAINING_SHA256:
        raise DatasetProvenanceError(
            "The Sentiment140 SHA-256 checksum does not match the benchmark reference. "
            f"Expected {SENTIMENT140_TRAINING_SHA256}, received {sha256}."
        )
    return DatasetManifest(
        filename=dataset_path.name,
        byte_size=dataset_path.stat().st_size,
        sha256=sha256,
    )
