# API de prédiction locale

L'API charge un artifact `selected_model` depuis le run parent MLflow du benchmark. Le mode est choisi au démarrage, jamais à la requête : `light` charge TF-IDF + Logistic Regression et `advanced` charge DistilBERT.

## Configuration

Copier les paramètres API de `.env.example` dans `.env`, puis renseigner les secrets Databricks déjà utilisés pour le tracking. Le nom du run parent est utilisé par défaut ; l'API choisit le dernier run parent terminé portant ce nom. Pour figer précisément une release, définir `LIGHT_MODEL_RUN_ID` ou `ADVANCED_MODEL_RUN_ID`.

## Lancement local

```powershell
py -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements-api.txt
$env:API_MODEL_MODE = 'light'
uvicorn app.main:app --app-dir api --reload
```

L'API initialise automatiquement `api/data/sentiment_api.sqlite3`. Consulter `http://127.0.0.1:8000/docs`, puis appeler `POST /predict`. La réponse contient un `prediction_id` à fournir à `POST /feedback` si un utilisateur estime la prédiction incorrecte.

## Tests sans modèle distant

```powershell
pytest api/tests
```

Les tests injectent un modèle factice : ils vérifient les handlers FastAPI et SQLite sans identifiants Databricks, ni téléchargement de poids. Le smoke test HTTP réel sera exécuté après le premier lancement Uvicorn hors environnement de test isolé.

## Déploiement Azure du mode light

Le `requirements.txt` à la racine référence `requirements-api-light.txt`. Il contient uniquement les dépendances nécessaires au mode `API_MODEL_MODE=light` et évite donc d'installer Torch et Transformers sur l'App Service F1. Les dépendances complètes, dont le mode DistilBERT, restent dans `requirements-api.txt` pour le développement local et un futur déploiement advanced sur une ressource plus dimensionnée.

## Artifacts attendus

Tous les modes requièrent `selected_model` et `decision_threshold.json` dans le run parent. Le mode `advanced` reconstruit le modèle à partir de `transformer_checkpoint.pt`, `transformer_model_config.json`, `transformer_tokenizer/`, `preprocessing_config.json` et `sequence_config.json`. Ces artifacts sont journalisés par la version actuelle du benchmark DistilBERT ; exécuter cette version avant de sélectionner le run avancé de l'API.
