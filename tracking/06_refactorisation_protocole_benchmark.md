# Protocole de benchmark partagé

## Objectif

TF-IDF + Logistic Regression et GloVe + LSTM utilisent maintenant le même protocole dans `notebooks/modelisation.ipynb`. La refactorisation réduit les cellules d'orchestration sans modifier les données, les baselines, les recherches Optuna, les métriques ou les règles métier.

## Étape 1 — Contrats communs

`BenchmarkDefinition` contient les métadonnées d'une famille et son contrat de recherche. `BenchmarkExecutionState` porte les résultats intermédiaires. `CandidateTrackingExtras`, `TestPrediction`, `TrackedCandidate` et `BenchmarkRunResult` structurent les informations échangées. `BenchmarkAdapter` rassemble les callbacks spécifiques à une famille.

## Étape 2 — Résumés des candidats

`CandidateResult.summary_record()` et `GloveLstmCandidateResult.summary_record()` restent dans leurs dataclasses respectives. Ils produisent les lignes de résultats validation, y compris les données GloVe propres au modèle neural.

## Étape 3 — Tracking d'un candidat

`run_and_track_candidate(...)` ouvre un run enfant MLflow, entraîne une baseline ou un essai, applique l'objectif Optuna existant, journalise les données communes et délègue les compléments au callback de famille.

## Étape 4 — Baseline et Optuna

`run_baseline_phase(...)` exécute la baseline fixe. `run_optuna_phase(...)` conserve le `TPESampler` seedé, `NopPruner`, `n_jobs=1`, les nombres d'essais et les timeouts existants.

## Étape 5 — Validation, sélection et refit

`finalize_validation_search(...)` écrit les résultats validation et les artefacts Optuna. `select_and_refit_candidate(...)` applique la règle métier existante et réentraîne le candidat retenu sur train seul.

## Étape 6 — Seuil

`tune_selected_threshold(...)` ne reçoit que validation. Il applique la grille existante, journalise le run enfant de réglage et fige le seuil gagnant.

## Étape 7 — Test et reporting

`evaluate_test_and_finalize(...)` est le seul point commun qui appelle la prédiction test. Il produit les métriques, rapport, matrice de confusion et artefacts communs. `run_benchmark_protocol(...)` enchaîne toutes les étapes dans l'ordre imposé.

## Étape 8 — Adaptateurs

`build_tfidf_adapter()` encapsule le pipeline scikit-learn et son modèle MLflow. `build_glove_lstm_adapter(...)` encapsule les ressources GloVe, l'historique neural, les séquences test, les artefacts d'inférence et le nettoyage TensorFlow. `build_word2vec_gru_adapter(...)` fournit le même contrat pour Word2Vec + GRU, tout en préparant le modèle Word2Vec une seule fois à partir du train. `build_distilbert_adapter(...)` adapte le protocole à un backbone téléchargé depuis Hugging Face et gelé pendant l'entraînement de la tête dense.

## Étape 9 — Cellules de familles

Les deux grandes cellules sont désormais limitées à la construction de leur définition, de leur adaptateur et à l'appel de `run_benchmark_protocol(...)`. Les DataFrames publics conservent leurs noms actuels. Le garde-fou `RUN_GLOVE_LSTM_BENCHMARK` est inchangé.

## Étape 10 — Invariants

La refactorisation préserve les splits 60/20/20, `PROJECT_SEED`, le premier accès au test après le seuil figé, les espaces de recherche, les noms de runs/artefacts et la règle orientée détection de bad buzz. Word2Vec + GRU et DistilBERT + tête dense appliquent désormais ce contrat : ce sont les familles prévues pour ce benchmark.
