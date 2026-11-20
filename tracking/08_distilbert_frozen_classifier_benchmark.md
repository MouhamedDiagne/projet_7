# Benchmark DistilBERT gelé + tête dense

## Objectif

Cette dernière famille reprend l'approche historique du projet : un encodeur DistilBERT pré-entraîné, gelé, suivi d'une tête dense binaire. Elle est intégrée au protocole commun : baseline, recherche Optuna sur validation, sélection métier, réglage du seuil sur validation, puis une seule évaluation sur test.

## Chargement direct depuis Internet

Le notebook charge `distilbert-base-uncased` avec `AutoTokenizer.from_pretrained(...)` et `AutoModel.from_pretrained(...)`. Il ne cherche aucun poids dans un Dataset Kaggle.

Avant une première exécution, activer **Internet** dans les réglages du notebook Kaggle. Hugging Face télécharge alors le tokenizer et les poids ; les candidates suivantes les relisent depuis le cache de la session. Si le téléchargement échoue, le message indique explicitement ce réglage Kaggle.

Le manifest MLflow conserve l'identifiant demandé, la révision demandée et la révision effectivement résolue quand elle est fournie par Hugging Face.

## Prétraitement et absence de fuite

Le profil `social_text_v1` est configuré pour adapter URLs, mentions, hashtags, emojis, émoticônes et répétitions tout en conservant chiffres et ponctuation pour le tokenizer Transformer. Il est appliqué à train et validation pendant la préparation des ressources. Le test n'est prétraité et tokenisé qu'après la sélection de la configuration et le gel du seuil.

Le tokenizer est pré-entraîné et ne reçoit aucun ajustement sur les textes du projet. Aucun vocabulaire n'est donc appris sur validation ou test.

## Architecture de baseline

La baseline `baseline_distilbert_frozen_head256` est :

`tweet tokenisé -> DistilBERT gelé -> masked mean pooling -> Dense(256) + ReLU + Dropout(0.3) -> 2 logits`

Les poids du backbone ne reçoivent pas de gradients et ses couches de dropout restent en mode évaluation. Seule la tête dense est entraînée, avec `AdamW`, `lr=1e-3`, des lots de 128, un maximum de 8 époques et un early stopping sur `val_loss` (patience 2). La longueur maximale est 64 tokens.

## Recherche Optuna

La recherche est fixe, séquentielle et limitée à 8 essais ou deux heures : `TPESampler(seed=PROJECT_SEED)`, `NopPruner`, `n_jobs=1`.

| Paramètre de tête | Valeurs |
| --- | --- |
| Longueur de séquence | 64, 96 |
| Unités de la couche dense | 128, 256, 384 |
| Dropout | 0.1, 0.3, 0.5 |
| Learning rate AdamW | 1e-4, 3e-4, 1e-3 |
| Batch size | 64, 128 |

Le backbone reste DistilBERT gelé pour toutes les candidates : les résultats isolent donc l'effet de la tête de classification.

## Exécution

Dans `notebooks/modelisation.ipynb`, après les sections de préparation, activer Internet dans Kaggle puis régler :

```python
RUN_DISTILBERT_FROZEN_CLASSIFIER_BENCHMARK = True
```

Les sorties publiques sont `DISTILBERT_VALIDATION_RESULTS`, `DISTILBERT_OPTUNA_RESULTS`, `DISTILBERT_THRESHOLD_RESULTS`, `DISTILBERT_TEST_METRICS` et `DISTILBERT_BENCHMARK_SUMMARY`.

## Tracking MLflow

Le parent run journalise le manifeste du backbone et sa configuration Transformers. Chaque candidate a un run enfant contenant ses paramètres, ses métriques validation, le nombre de paramètres entraînables et totaux, les durées et l'historique par époque.

Le run final contient notamment :

- `transformer_manifest.json` et `transformer_model_config.json` ;
- `transformer_checkpoint.pt`, utilisable pour reconstruire le modèle hors du notebook ;
- les fichiers du tokenizer sous `transformer_tokenizer/` ;
- `preprocessing_config.json`, `sequence_config.json` et `inference_contract.json` ;
- seuil, rapports et métriques de test ;
- le wrapper PyTorch sélectionné sous `selected_model`.
