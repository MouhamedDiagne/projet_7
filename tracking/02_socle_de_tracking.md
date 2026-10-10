# Socle de tracking MLflow

## Objectif

Le projet offre deux backends MLflow derrière la même fonction
`configure_tracking()` :

| Mode | Cas d'usage | Stockage des runs |
|---|---|---|
| `local` | Travail local, sans compte externe | `<racine-du-projet>/mlruns/` (SQLite + artifacts) |
| `databricks` | Benchmark exécuté sur Kaggle | Experiment MLflow Databricks distant |

Kaggle refuse volontairement le mode `local` : un benchmark GPU doit rester
traçable dans Databricks. Hors Kaggle, l'absence de configuration Databricks
active automatiquement le mode local.

## Fichiers fournis

- `mlflow_tracking.py` sélectionne le backend, le configure et fournit le
  smoke test.
- `smoke_test.py` est le point d'entrée de smoke test.
- `../requirements-tracking.txt` contient MLflow et l'intégration Databricks.
- `../requirements-kaggle.txt` contient les dépendances complémentaires du
  notebook sans réinstaller TensorFlow ni PyTorch, déjà fournis avec le support GPU de Kaggle.
- `../.env.example` documente les variables locales et distantes sans secret.

## Smoke test local — sans Kaggle ni Databricks

Ce test vérifie qu'une personne possédant seulement une copie du projet peut
créer, relire et consulter des runs MLflow locaux.

1. Depuis la racine du projet, créer puis activer un environnement Python.

   ```powershell
   py -m venv env
   .\env\Scripts\Activate.ps1
   ```

2. Installer les dépendances du notebook.

   ```powershell
   pip install --upgrade-strategy only-if-needed -r requirements-kaggle.txt
   ```

3. Créer un fichier `.env` local sans secret.

   ```powershell
   Copy-Item .env.example .env
   ```

   Vérifier que `.env` contient `MLFLOW_TRACKING_MODE=local`.

4. Lancer le smoke test.

   ```powershell
   python tracking\smoke_test.py
   ```

   Une sortie réussie indique le backend `local`, un experiment ID, un run ID
   et l'artifact `tracking_smoke_test.txt`.

5. Consulter l'interface MLflow locale, puis ouvrir l'URL affichée (en général `http://127.0.0.1:5000`) dans un navigateur.

   ```powershell
   mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
   ```

   Le mode local utilise SQLite pour les métadonnées et `mlruns/artifacts/` pour les artifacts. Arrêter le serveur avec `Ctrl+C`. Le dossier `mlruns` est ignoré par Git.

## Smoke test Kaggle / Databricks

1. Activer Internet dans le notebook Kaggle.
2. Dans **Add-ons > Secrets**, créer et autoriser ces secrets pour le notebook :

   | Secret Kaggle | Valeur |
   |---|---|
   | `MLFLOW_TRACKING_MODE` | `databricks` |
   | `DATABRICKS_HOST` | URL HTTPS du workspace, sans slash final |
   | `DATABRICKS_TOKEN` | Jeton d'accès Databricks |
   | `MLFLOW_EXPERIMENT_ID` | ID numérique de `sentiment140-benchmark-v1` |

3. Rendre le dossier `tracking` et
   `requirements-kaggle.txt` accessibles dans `/kaggle/working` (via le bundle du notebook à l'étape 7), puis installer les dépendances :

   ```python
   !pip install --quiet --upgrade-strategy only-if-needed -r requirements-kaggle.txt
   ```

4. Dans une cellule, lancer le smoke test :

   ```python
   import sys
   from pathlib import Path

   sys.path.insert(0, str(Path('/kaggle/working')))
   from tracking.mlflow_tracking import run_smoke_test

   result = run_smoke_test()
   print(result)
   ```

5. Dans Databricks, ouvrir `sentiment140-benchmark-v1`. Le run
   `tracking-smoke-test-...` doit afficher le paramètre `smoke_test`, la métrique
   `tracking_connection` et l'artifact `tracking_smoke_test.txt`.

Le test n'envoie ni donnée du corpus, ni poids de modèle, ni secret. Il échoue
explicitement si les paramètres sont absents, si l'experiment est inaccessible
ou si l'artifact ne peut pas être relu.

## Utilisation future dans le benchmark

Le notebook exécutera une fois :

```python
from tracking.mlflow_tracking import configure_tracking

tracking_config = configure_tracking()
```

Chaque run utilisera ensuite explicitement
`experiment_id=tracking_config.experiment_id`. L'instrumentation complète du
notebook est reportée à l'étape 7.
