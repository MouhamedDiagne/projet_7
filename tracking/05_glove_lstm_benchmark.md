# Benchmark : GloVe + LSTM

## Périmètre

Cette famille transpose la baseline historique du projet dans le protocole de
benchmark commun : embeddings GloVe Twitter 27B de 50 dimensions gelés,
suivis d'un LSTM et d'une sortie sigmoïde. Son implémentation reste entièrement
dans `notebooks/modelisation.ipynb`, dans les utilitaires de benchmark puis la
section `GloVe + LSTM`.

Elle ne s'exécute que lorsque `RUN_GLOVE_LSTM_BENCHMARK = True`. Cette sécurité
évite qu'un notebook sans asset GloVe attaché échoue après les autres familles.

## Asset GloVe requis

Le fichier exact attendu est `glove.twitter.27B.50d.txt`. Il ne doit pas être
téléchargé pendant le benchmark : l'utilisateur l'ajoute comme dataset Kaggle
au notebook, avec son slug réel dans `kernel-metadata.json`. Le notebook le
détecte alors sous `/kaggle/input`; `GLOVE_EMBEDDING_PATH` permet de choisir un
chemin explicite si plusieurs copies sont attachées.

Avant chaque run, le notebook vérifie la dimension 50, puis journalise le nom,
le chemin et la taille de l'asset. Le fichier de vecteurs et la
matrice complète ne sont jamais téléversés dans MLflow.

## Prétraitement et tokenisation

La famille utilise `social_text_v1` avec tous ses interrupteurs explicites.
Par rapport au réglage par défaut TF-IDF, elle active deux comportements issus
du notebook initial :

- `replace_ascii_emoticons=True` ;
- `reduce_elongated_tokens=True`.

Les URL, mentions, nombres, hashtags et emojis restent représentés par des
tokens stables. Le tokenizer TensorFlow préserve donc les underscores de
`url_token`, `user_token` et des autres tokens de signal.

Il est ajusté exclusivement sur les tweets prétraités de `train`, avec un token
OOV. `validation` est seulement transformé pendant la sélection ; `test` n'est
prétraité et encodé qu'après le gel de la configuration et du seuil. Le taux de
couverture GloVe du vocabulaire de chaque candidate est journalisé.

## Baseline et recherche

La baseline reprend l'architecture historique :

```text
Input(64) -> Embedding GloVe 50d gelé -> LSTM(196, dropout=0.2) -> Dense(1, sigmoid)
```

Adam, binary cross-entropy, batch size 64 et dix époques au plus sont utilisés.
`EarlyStopping` suit `val_loss`, avec restauration des meilleurs poids.

La recherche Optuna TPE est volontairement bornée à huit essais séquentiels,
avec une limite de deux heures. Elle ne fait varier que :

| Élément | Valeurs |
|---|---|
| Taille du vocabulaire | 50 000, 75 000 |
| Longueur de séquence | 48, 64 |
| Unités LSTM | 128, 196 |
| Dropout | 0.1, 0.2, 0.3 |
| Learning rate Adam | 3e-4, 1e-3 |
| Batch size | 64, 128 |

La dimension GloVe, la source des vecteurs, l'embedding gelé, le
`recurrent_dropout=0`, le prétraitement et le nombre maximal d'époques restent
fixes. Chaque candidate repart d'une session TensorFlow nettoyée et de
`PROJECT_SEED`.

## Protocole de sélection

1. Baseline et essais Optuna s'entraînent sur `train` et sont mesurés sur
   `validation` au seuil 0.5.
2. L'objectif Optuna est le F-beta négatif lorsque l'accuracy validation est au
   moins 60 %, sinon `-1.0`.
3. La sélection finale applique la règle partagée : accuracy ≥ 60 %, F-beta
   négatif décroissant, bad buzz manqués croissant, puis accuracy décroissante.
4. La configuration retenue est réentraînée une seule fois, puis son seuil est
   réglé exclusivement sur `validation`.
5. Le jeu `test` est évalué une seule fois avec ce seuil gelé.

## Tracking MLflow

Le run parent est `glove-lstm-benchmark`; il possède des runs enfants pour la
baseline, chaque essai Optuna et le réglage de seuil. Chaque réseau neuronal
journalise ses hyperparamètres, sa couverture GloVe, ses temps, ses métriques
validation, `training_history.csv` et `learning_curves.png`.

Le parent conserve en plus :

- `glove_asset_manifest.json` et `glove_coverage.json` ;
- `validation_configuration_results.csv`, `optuna_trial_results.csv` et
  `optuna_search_config.json` ;
- `threshold_search_results.csv`, `test_metrics.csv`, la matrice de confusion
  et le rapport de classification ;
- `tokenizer.json`, `preprocessing_config.json`, `sequence_config.json` et
  `inference_contract.json` ;
- le modèle Keras sélectionné sous `selected_model`.

Le modèle reste marqué `selected_for_test=true`; il n'est pas encore le
champion du benchmark, qui sera choisi lors de la comparaison finale.
