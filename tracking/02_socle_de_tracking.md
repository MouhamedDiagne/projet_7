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
  smoke test pour les usages Python locaux.
- `smoke_test.py` est le point d'entrée de smoke test local.
- `../requirements-tracking.txt` contient MLflow et l'intégration Databricks.
- `../requirements-kaggle.txt` contient les dépendances que le notebook peut
  installer sur Kaggle, sans réinstaller TensorFlow ni PyTorch déjà fournis
  avec le support GPU.
- `../requirements-local.txt` compose un environnement local complet : il
  inclut `requirements-kaggle.txt`, TensorFlow, PyTorch et le client Kaggle
  utilisé par les scripts de synchronisation.
- `../.env.example` documente les variables locales et distantes sans secret.

Sur Kaggle, `notebooks/modelisation.ipynb` est volontairement autonome : sa
section `Utils` contient les fonctions de tracking, de provenance et de seed,
et sa première cellule de dépendances installe les packages requis. Cette
duplication ciblée est nécessaire car `kaggle kernels push` publie le notebook
et ses métadonnées, pas les fichiers Python ou requirements voisins.

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
   pip install --upgrade-strategy only-if-needed -r requirements-local.txt
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
   | `DATABRICKS_HOST` | URL HTTPS du workspace, sans slash final |
   | `DATABRICKS_TOKEN` | Jeton d'accès Databricks |
   | `MLFLOW_EXPERIMENT_ID` | ID numérique de `sentiment140-benchmark-v1` |

   `MLFLOW_TRACKING_MODE=databricks` est facultatif sur Kaggle : le notebook
   sélectionne automatiquement ce mode dans cet environnement. Les trois
   labels ci-dessus sont sensibles à la casse et doivent être autorisés pour
   ce notebook.

3. Exécuter les cellules **Dependencies**, **Utils**, **Imports** et
   **Tracking configuration** du notebook. La cellule Dependencies installe
   les packages nécessaires directement depuis sa liste intégrée ; aucun fichier
   requirements n'est attendu dans `/kaggle/working`.

4. Dans une cellule placée après la configuration, lancer le smoke test intégré :

   ```python
   result = run_tracking_smoke_test(tracking_config)
   print(result)
   ```

5. Dans Databricks, ouvrir `sentiment140-benchmark-v1`. Le run
   `tracking-smoke-test-...` doit afficher le paramètre `smoke_test`, la métrique
   `tracking_connection` et l'artifact `tracking_smoke_test.txt`.

Le test n'envoie ni donnée du corpus, ni poids de modèle, ni secret. Il échoue
explicitement si les paramètres sont absents, si l'experiment est inaccessible
ou si l'artifact ne peut pas être relu.

## Utilisation future dans le benchmark

Le notebook exécute une fois, depuis ses utilitaires intégrés :

```python
tracking_config = configure_tracking()
```

Chaque run utilisera ensuite explicitement
`experiment_id=tracking_config.experiment_id`. L'instrumentation complète du
notebook est reportée à l'étape 7.
