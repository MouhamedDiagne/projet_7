# Données et reproductibilité

## Contrat de données

Le benchmark utilise exclusivement
`training.1600000.processed.noemoticon.csv` du corpus Sentiment140. Les
instructions de téléchargement, la source, la citation et l'encodage sont dans
[`../data/README.md`](../data/README.md).

Le notebook vérifie que le fichier attendu existe et porte le bon nom avant de
charger les tweets.

## Asset externe GloVe

La famille GloVe + LSTM dépend aussi du fichier
`glove.twitter.27B.50d.txt`. Il est attaché au notebook comme dataset Kaggle,
jamais téléchargé implicitement. Son chemin, son nom, sa taille et sa dimension
sont journalisés dans le run MLflow parent. Les détails de configuration sont dans
[`05_glove_lstm_benchmark.md`](05_glove_lstm_benchmark.md).

## Seed projet

La variable non sensible `PROJECT_SEED` centralise la seed du benchmark. Sa
valeur par défaut est `42`; elle peut être changée dans `.env` local, dans un
secret Kaggle portant ce nom, ou dans la configuration du notebook avant son
exécution.

Au démarrage, `configure_reproducibility()` applique cette valeur à :

- Python (`random`) ;
- NumPy ;
- TensorFlow et Keras ;
- PyTorch et CUDA lorsqu'ils sont disponibles.

Les options déterministes de TensorFlow et PyTorch sont demandées lorsque les
frameworks les prennent en charge. Les résultats doivent néanmoins être
comparés avec une tolérance : certaines opérations GPU restent susceptibles de
présenter de légères différences numériques selon le matériel, les pilotes et
les versions de bibliothèques.

Cette fonction, ainsi que le manifeste léger du dataset, est intégrée dans la
section `Utils` de `notebooks/modelisation.ipynb` pour l'exécution Kaggle. Les
modules `tracking/` restent la référence réutilisable des contrôles locaux.

## Métadonnées à journaliser dans chaque run

Après l'instrumentation des modèles, chaque `mlflow.start_run()` doit recevoir
les paramètres de provenance déjà préparés par le notebook :

```python
with mlflow.start_run(experiment_id=tracking_config.experiment_id):
    mlflow.log_params(RUN_PROVENANCE_PARAMS)
    # entraînement et métriques du modèle
```

`RUN_PROVENANCE_PARAMS` contient la seed, l'état des options déterministes, le
nom et la taille du fichier, ainsi que son encodage. Ces valeurs ne contiennent
ni tweet, ni token, ni autre donnée sensible.

## Vérification rapide locale

```powershell
.\env\Scripts\python.exe -c "from tracking.data_provenance import verify_sentiment140_training_file; print(verify_sentiment140_training_file('data/sentiment140/training.1600000.processed.noemoticon.csv'))"
```

La commande doit afficher le nom, la taille et l'encodage du fichier.
