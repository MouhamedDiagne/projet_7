"""Deterministic-seed helpers shared by local and Kaggle benchmark runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


PROJECT_SEED_ENV_VAR = "PROJECT_SEED"
DEFAULT_PROJECT_SEED = 42
MAX_NUMPY_SEED = 2**32 - 1


class ReproducibilityConfigurationError(ValueError):
    """Raised when the configured project seed is invalid."""


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Applied random-state configuration and framework-specific outcomes."""

    seed: int
    tensorflow_status: str
    pytorch_status: str
    deterministic_operations_requested: bool

    def as_mlflow_params(self) -> dict[str, str | int]:
        """Return scalar values safe to log from within an active MLflow run."""

        return {
            "project_seed": self.seed,
            "deterministic_operations_requested": str(
                self.deterministic_operations_requested
            ).lower(),
            "tensorflow_seed_status": self.tensorflow_status,
            "pytorch_seed_status": self.pytorch_status,
        }


def _read_kaggle_seed() -> str | None:
    """Return the optional Kaggle `PROJECT_SEED` secret when available."""

    if not Path("/kaggle/working").exists():
        return None
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(PROJECT_SEED_ENV_VAR)
    except Exception:
        return None
    return value.strip() if value else None


def get_project_seed(environment: Mapping[str, str] | None = None) -> int:
    """Read `PROJECT_SEED`, defaulting to 42 when it is not configured."""

    source = os.environ if environment is None else environment
    raw_seed = source.get(PROJECT_SEED_ENV_VAR)
    if raw_seed is None and environment is None:
        raw_seed = _read_kaggle_seed()
    raw_seed = (raw_seed or str(DEFAULT_PROJECT_SEED)).strip()
    try:
        seed = int(raw_seed)
    except ValueError as error:
        raise ReproducibilityConfigurationError(
            f"{PROJECT_SEED_ENV_VAR} must be an integer between 0 and {MAX_NUMPY_SEED}."
        ) from error
    if not 0 <= seed <= MAX_NUMPY_SEED:
        raise ReproducibilityConfigurationError(
            f"{PROJECT_SEED_ENV_VAR} must be between 0 and {MAX_NUMPY_SEED}."
        )
    return seed


def configure_reproducibility(
    seed: int | None = None,
    deterministic_operations: bool = True,
) -> ReproducibilityConfig:
    """Seed Python, NumPy, TensorFlow and PyTorch when installed.

    GPU libraries and kernels can still introduce minor floating-point variation
    across operating systems and hardware. Framework flags below request the
    strongest deterministic behavior available without aborting the benchmark.
    """

    seed = get_project_seed() if seed is None else seed
    if not 0 <= seed <= MAX_NUMPY_SEED:
        raise ReproducibilityConfigurationError(
            f"seed must be between 0 and {MAX_NUMPY_SEED}."
        )

    # Setting PYTHONHASHSEED at runtime affects child processes; Python's hash
    # seed for the current interpreter is chosen at process startup.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    tensorflow_status = "not_installed"
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        if deterministic_operations:
            try:
                tf.config.experimental.enable_op_determinism()
                tensorflow_status = "seeded_and_deterministic_ops_enabled"
            except (AttributeError, RuntimeError):
                tensorflow_status = "seeded_deterministic_ops_unavailable"
        else:
            tensorflow_status = "seeded"
    except ImportError:
        pass

    pytorch_status = "not_installed"
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_operations:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            pytorch_status = "seeded_and_deterministic_ops_requested"
        else:
            pytorch_status = "seeded"
    except ImportError:
        pass

    return ReproducibilityConfig(
        seed=seed,
        tensorflow_status=tensorflow_status,
        pytorch_status=pytorch_status,
        deterministic_operations_requested=deterministic_operations,
    )


def log_reproducibility_metadata(config: ReproducibilityConfig) -> None:
    """Log seed metadata to the currently active MLflow run only."""

    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError(
            "An active MLflow run is required before logging reproducibility metadata."
        )
    mlflow.log_params(config.as_mlflow_params())
