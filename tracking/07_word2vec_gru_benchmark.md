# Benchmark Word2Vec + GRU

## Objectif

Cette troisième famille de benchmark associe des représentations Word2Vec entraînées localement sur les tweets de train à un classifieur GRU. Elle utilise le même protocole partagé que TF-IDF + Logistic Regression et GloVe + LSTM : baseline, recherche Optuna sur validation, sélection métier, réglage du seuil sur validation, puis une unique évaluation sur test.

## Données et absence de fuite

`prepare_word2vec_gru_resources(...)` applique le prétraitement configuré aux textes train et validation. Le modèle Word2Vec et le tokenizer Keras sont ajustés uniquement sur train. Validation est uniquement transformé pour évaluer les candidates ; test est prétraité et prédit seulement après le choix de la configuration et le gel du seuil.

Les paramètres Word2Vec sont fixes pendant toute l'exécution d'une famille. Ils ne font pas partie de la recherche Optuna, ce qui permet de comparer les GRU avec le même espace vectoriel. `workers=1` et `PROJECT_SEED` privilégient la reproductibilité.

## Baseline

La baseline `baseline_word2vec100_gru196` est la suivante :

`tweets prétraités -> Word2Vec entraîné sur train (100 dimensions, gelé) -> GRU(196, dropout=0.2) -> sigmoïde`

Word2Vec utilise Skip-gram (`sg=1`), une fenêtre de 5 tokens, `min_count=2`, 10 échantillons négatifs et 10 époques. Les séquences ont au plus 64 tokens, le vocabulaire du tokenizer est plafonné à 75 000, et l'entraînement de la GRU suit Adam avec un taux de `1e-3`, des lots de 64 et un early stopping sur `val_loss` (patience 2).

## Recherche Optuna

La recherche est volontairement compacte : 8 essais au maximum, un timeout de deux heures, `TPESampler(seed=PROJECT_SEED)`, `NopPruner` et `n_jobs=1`.

| Paramètre GRU | Valeurs |
| --- | --- |
| Vocabulaire tokenizer | 50 000, 75 000 |
| Longueur de séquence | 48, 64 |
| Unités GRU | 128, 196 |
| Dropout | 0.1, 0.2, 0.3 |
| Learning rate Adam | 3e-4, 1e-3 |
| Batch size | 64, 128 |

Les vecteurs Word2Vec restent gelés. À chaque candidate, la matrice d'embeddings est limitée au vocabulaire demandé et sa couverture est mesurée.

## Activation dans le notebook

Dans `notebooks/modelisation.ipynb`, exécuter les sections précédentes puis régler :

```python
RUN_WORD2VEC_GRU_BENCHMARK = True
```

La cellule prépare Word2Vec une fois, exécute la baseline et les essais Optuna, puis publie les DataFrames `WORD2VEC_GRU_VALIDATION_RESULTS`, `WORD2VEC_GRU_OPTUNA_RESULTS`, `WORD2VEC_GRU_THRESHOLD_RESULTS`, `WORD2VEC_GRU_TEST_METRICS` et `WORD2VEC_GRU_BENCHMARK_SUMMARY`.

## Tracking MLflow

Le run parent journalise la configuration Word2Vec, son manifeste et le fichier `word2vec.model`. Chaque run enfant journalise les paramètres GRU, les métriques validation, la couverture du vocabulaire, les durées de préparation et d'entraînement, ainsi que l'historique Keras.

Après sélection, les artefacts d'inférence comprennent notamment :

- `word2vec_training_config.json` et `word2vec_manifest.json` ;
- `word2vec.model` ;
- `tokenizer.json`, `preprocessing_config.json` et `sequence_config.json` ;
- `word2vec_coverage.json` ;
- `decision_threshold.json`, `inference_contract.json`, les résultats de validation et les rapports/métriques de test.

Le modèle TensorFlow sélectionné est journalisé sous `selected_model`.
