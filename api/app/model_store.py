"""Resolve a benchmark parent run and load its logged model artifacts from MLflow."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import ApiSettings
from .preprocessing import PreprocessingConfig, preprocess_tweet


class ModelReleaseError(RuntimeError):
    """Raised when a configured MLflow run cannot become an API model release."""


class SentimentPredictor(Protocol):
    """Minimal inference contract shared by the two API model modes."""

    def predict_positive_probability(self, tweet: str) -> float:
        """Return the probability that one tweet belongs to the positive class."""


@dataclass(frozen=True)
class LoadedRelease:
    """One immutable, resolved model release served by the API process."""

    predictor: SentimentPredictor
    model_mode: str
    run_id: str
    run_name: str
    threshold: float


class SklearnPredictor:
    """Adapt a fitted TF-IDF plus Logistic Regression pipeline to API inference."""

    def __init__(self, model):
        """Store the fitted pipeline once it has been downloaded from MLflow."""
        self.model = model

    def predict_positive_probability(self, tweet: str) -> float:
        """Read P(y=1) without assuming a fixed fitted-class ordering."""
        probabilities = self.model.predict_proba([tweet])
        classes = np.asarray(self.model.classes_)
        positive_index = int(np.where(classes == 1)[0][0])
        return float(probabilities[0, positive_index])


class TorchTransformerPredictor:
    """Adapt the logged frozen-DistilBERT wrapper and tokenizer to API inference."""

    def __init__(self, model, tokenizer, preprocessing, max_sequence_length: int):
        """Freeze runtime behavior for a downloaded advanced-model release."""
        self.model = model.cpu().eval()
        self.tokenizer = tokenizer
        self.preprocessing = preprocessing
        self.max_sequence_length = max_sequence_length

    def predict_positive_probability(self, tweet: str) -> float:
        """Preprocess, tokenize and score one tweet on CPU."""
        import torch

        cleaned_tweet = preprocess_tweet(tweet, self.preprocessing)
        inputs = self.tokenizer(
            cleaned_tweet,
            padding='max_length',
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='pt',
        )
        with torch.inference_mode():
            logits = self.model(**dict(inputs))
            return float(torch.softmax(logits, dim=1)[0, 1].item())


class FrozenTransformerClassifier:
    """Serving-time counterpart of the frozen encoder and dense head benchmark model."""

    def __init__(self, encoder, hidden_size: int, classifier_hidden_units: int, dropout: float):
        """Attach the checkpoint-compatible dense classifier to its Transformer encoder."""
        import torch

        self.module = torch.nn.Module()
        self.module.encoder = encoder
        self.module.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, classifier_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(classifier_hidden_units, 2),
        )

        def forward(**encoder_inputs):
            outputs = self.module.encoder(**encoder_inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoder_inputs['attention_mask'].unsqueeze(-1).type_as(token_embeddings)
            sentence_embedding = (token_embeddings * attention_mask).sum(dim=1)
            sentence_embedding = sentence_embedding / attention_mask.sum(dim=1).clamp_min(1e-9)
            return self.module.classifier(sentence_embedding)

        self.module.forward = forward


def load_release_from_mlflow(settings: ApiSettings) -> LoadedRelease:
    """Resolve the configured benchmark run and load its selected model artifact."""
    mlflow, client, run = _resolve_mlflow_run(settings)
    threshold = _read_json_artifact(client, run.info.run_id, 'decision_threshold.json')['threshold']
    if settings.model_mode == 'light':
        import mlflow.sklearn

        model = mlflow.sklearn.load_model(f'runs:/{run.info.run_id}/{settings.model_artifact_path}')
        predictor: SentimentPredictor = SklearnPredictor(model)
    else:
        predictor = _load_advanced_predictor(mlflow, client, run.info.run_id, settings)
    return LoadedRelease(
        predictor=predictor,
        model_mode=settings.model_mode,
        run_id=run.info.run_id,
        run_name=run.info.run_name,
        threshold=float(threshold),
    )


def _resolve_mlflow_run(settings: ApiSettings):
    """Find a finished parent run by its exact name, preferring an explicit run ID."""
    if not settings.mlflow_experiment_id:
        raise ModelReleaseError('MLFLOW_EXPERIMENT_ID is required to resolve a model release.')
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as error:
        raise ModelReleaseError('Install requirements-api.txt before loading an MLflow release.') from error
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    if settings.selected_run_id:
        run = client.get_run(settings.selected_run_id)
        if run.info.status != 'FINISHED':
            raise ModelReleaseError(f'Configured run {run.info.run_id} is not finished.')
        return mlflow, client, run

    safe_name = settings.selected_run_name.replace("'", "\\'")
    runs = client.search_runs(
        [settings.mlflow_experiment_id],
        filter_string=f"attributes.run_name = '{safe_name}'",
        order_by=['attributes.end_time DESC'],
        max_results=1_000,
    )
    candidates = [
        run for run in runs
        if run.info.status == 'FINISHED'
        and run.data.tags.get('run_role', 'model_family_parent') == 'model_family_parent'
    ]
    if not candidates:
        raise ModelReleaseError(
            f'No finished parent run named {settings.selected_run_name!r} was found in '
            f'experiment {settings.mlflow_experiment_id}. Set LIGHT_MODEL_RUN_ID or '
            'ADVANCED_MODEL_RUN_ID to pin an exact release.'
        )
    return mlflow, client, candidates[0]


def _read_json_artifact(client, run_id: str, artifact_path: str) -> dict:
    """Download and parse one small JSON artifact from an MLflow run."""
    with tempfile.TemporaryDirectory(prefix='sentiment-api-') as directory:
        try:
            downloaded_path = client.download_artifacts(run_id, artifact_path, directory)
            return json.loads(Path(downloaded_path).read_text(encoding='utf-8'))
        except Exception as error:
            raise ModelReleaseError(
                f'Run {run_id} is missing or cannot read required artifact {artifact_path!r}.'
            ) from error


def _load_advanced_predictor(mlflow, client, run_id: str, settings: ApiSettings) -> TorchTransformerPredictor:
    """Load the logged BERT wrapper with its tokenizer and preprocessing contract."""
    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as error:
        raise ModelReleaseError('Advanced mode requires torch and transformers from requirements-api.txt.') from error
    try:
        model_config_payload = _read_json_artifact(client, run_id, 'transformer_model_config.json')
        contract = _read_json_artifact(client, run_id, 'inference_contract.json')
        with tempfile.TemporaryDirectory(prefix='sentiment-api-') as directory:
            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps(model_config_payload), encoding='utf-8')
            model_config = AutoConfig.from_pretrained(directory, local_files_only=True)
            model = FrozenTransformerClassifier(
                encoder=AutoModel.from_config(model_config),
                hidden_size=int(model_config.hidden_size),
                classifier_hidden_units=int(contract['classifier_hidden_units']),
                dropout=float(contract['classifier_dropout']),
            ).module
            checkpoint_path = client.download_artifacts(run_id, 'transformer_checkpoint.pt', directory)
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except TypeError:  # Compatibility with older PyTorch installations.
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
    except Exception as error:
        raise ModelReleaseError(
            f'Could not reconstruct the advanced model from run {run_id}. '
            'Run the current DistilBERT benchmark so it logs transformer_checkpoint.pt.'
        ) from error
    preprocessing = PreprocessingConfig.from_mlflow_payload(
        _read_json_artifact(client, run_id, 'preprocessing_config.json')
    )
    sequence = _read_json_artifact(client, run_id, 'sequence_config.json')
    with tempfile.TemporaryDirectory(prefix='sentiment-api-') as directory:
        try:
            tokenizer_path = client.download_artifacts(run_id, 'transformer_tokenizer', directory)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception as error:
            raise ModelReleaseError(
                f'Run {run_id} is missing the transformer_tokenizer artifact required by advanced mode.'
            ) from error
    return TorchTransformerPredictor(
        model=model,
        tokenizer=tokenizer,
        preprocessing=preprocessing,
        max_sequence_length=int(sequence['max_sequence_length']),
    )
