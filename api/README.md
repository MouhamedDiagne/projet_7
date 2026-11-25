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

## CI/CD GitHub Actions vers Azure

Le workflow `.github/workflows/api-ci-cd.yml` teste l'API sur les pull requests vers `main`, les push sur `main` et les lancements manuels. Après réussite des tests, seuls les push sur `main` et les lancements manuels depuis `main` déploient le mode light sur `oc-project-7`. Les déploiements sont sérialisés et vérifient `/health` pendant 10 minutes au maximum. Si la variable GitHub Actions `LIGHT_MODEL_RUN_ID` est définie, elle doit correspondre au run chargé ; sinon le contrôle exige seulement un run identifié en mode light. Ce contrôle échoue sans retour arrière automatique.

Le workflow empaquette uniquement `api/app`, `requirements.txt` et `requirements-api-light.txt` ; il ne déploie ni `.env` ni la base SQLite locale. L'App Service doit conserver `SCM_DO_BUILD_DURING_DEPLOYMENT=true`, sa commande de démarrage et ses variables Databricks. Le `DATABRICKS_TOKEN` reste dans Azure et n'est pas nécessaire à GitHub Actions.

Pour autoriser GitHub à déployer, configurer une identité Azure fédérée OIDC avec :

- issuer : `https://token.actions.githubusercontent.com` ;
- subject : `repo:MouhamedDiagne/projet_7:ref:refs/heads/main` ;
- audience : `api://AzureADTokenExchange` ;
- rôle `Website Contributor` limité à la Web App `oc-project-7`.

Dans GitHub, sous **Settings → Secrets and variables → Actions**, créer les secrets `AZURE_CLIENT_ID`, `AZURE_TENANT_ID` et `AZURE_SUBSCRIPTION_ID` avec les identifiants de cette identité et de la souscription. On peut aussi définir la variable facultative `LIGHT_MODEL_RUN_ID` avec le même run que dans l'App Service pour vérifier la release après déploiement. Ces identifiants OIDC sont distincts du token Databricks.

Le workflow doit être commité et poussé pour s'exécuter. Après le premier déploiement, vérifier les logs GitHub Actions, puis appeler `/predict` et `/feedback` manuellement : les tests isolés et `/health` ne valident pas à eux seuls une prédiction réelle.
