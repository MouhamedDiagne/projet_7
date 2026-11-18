# Benchmark : TF-IDF + Logistic Regression

## Périmètre

Cette famille constitue la première référence classique du benchmark. Son code
et son orchestration MLflow se trouvent dans la section `Benchmark Utils` puis
les cellules TF-IDF de `notebooks/modelisation.ipynb`, afin que le notebook
exécuté sur Kaggle soit autonome.

Le protocole est fixe : entraînement sur `train`, sélection et réglage du seuil
sur `validation`, puis une unique évaluation sur `test`.

## Prétraitement configurable

Deux profils de prétraitement sont comparés sans utiliser le jeu de test :

- `whitespace_v1` : référence minimale historique ; valeurs manquantes
  remplacées par une chaîne vide et espaces normalisés.
- `social_text_v1` : normalisation pour des tweets, intégrée comme première
  étape du pipeline scikit-learn afin d'être sérialisée avec le modèle retenu.

Les paramètres de `social_text_v1` sont exposés dans la cellule de
configuration du notebook. Chaque comportement s'active ou se désactive en
changeant sa valeur booléenne :

```python
SOCIAL_TEXT_V1_PARAMETERS = {
    'normalize_whitespace': True,
    'decode_html_entities': True,
    'expand_contractions': True,
    'replace_urls': True,
    'replace_emails': True,
    'replace_mentions': True,
    'expand_hashtags': True,
    'demojize_emojis': True,
    'replace_ascii_emoticons': False,
    'reduce_elongated_tokens': False,
    'replace_numbers': True,
    'encode_sentiment_punctuation': True,
    'remove_digits': False,
    'remove_punctuation': False,
}
```

Les remplacements par des tokens stables (`url_token`, `user_token`,
`number_token`, etc.) sont activés par défaut afin de conserver le signal sans
garder des valeurs trop spécifiques. `replace_numbers` et `remove_digits` sont
des options alternatives ; `encode_sentiment_punctuation` et
`remove_punctuation` le sont aussi. Les émoticônes ASCII et la réduction des
mots allongés sont disponibles pour les familles séquentielles mais restent
désactivées pour cette référence TF-IDF. Le notebook refuse une configuration
qui active simultanément deux options incompatibles.

## Stratégie de recherche

Une baseline fixe, `baseline_unigram_c1`, est d'abord évaluée avec
`whitespace_v1`. Optuna exécute ensuite au plus 20 essais TPE séquentiels,
avec la seed projet et un délai maximal d'une heure.

L'espace de recherche reste volontairement borné :

| Élément | Valeurs recherchées |
|---|---|
| Profil de prétraitement | `whitespace_v1`, `social_text_v1` |
| N-grammes mots | 1–1, 1–2 |
| `min_df` | 2, 5, 10 |
| `max_df` | 0.95, 1.0 |
| `max_features` | 75 000, 120 000 |
| Pénalité LogReg | `l1`, `l2` |
| `C` | échelle logarithmique de 0.1 à 5.0 |

Le solveur reste `liblinear`, `max_iter=1000`, `sublinear_tf=True` et
`random_state=PROJECT_SEED`. Aucun simulateur de temps ni pruner Optuna n'est
utilisé : une régression logistique `liblinear` ne fournit pas de métrique
intermédiaire utile pendant son entraînement.

## Processus appliqué

1. La baseline et chaque essai Optuna sont entraînés exclusivement sur le split
   `train`, puis évalués sur `validation` au seuil par défaut `0.5`.
2. L'objectif interne Optuna est le F-beta négatif lorsque l'accuracy atteint
   60 %, sinon `-1.0`. Il guide les propositions, mais ne décide pas du modèle
   final.
3. La baseline et tous les essais terminés sont soumis à la règle métier :
   accuracy validation ≥ 60 %, puis F-beta négatif (`beta=3`) décroissant,
   bad buzz manqués pour 1 000 tweets négatifs croissant, puis accuracy
   décroissante.
4. La configuration retenue est réentraînée une fois sur `train`, sans
   conserver en mémoire les pipelines de tous les essais. Son seuil est ensuite
   recherché sur `validation` uniquement.
5. Le seuil 0.5 est inclus dans une grille de 181 seuils distincts ; le meilleur
   seuil est gelé.
6. Le jeu `test` est utilisé une seule fois, après ces sélections.

## Données journalisées

Le run parent est `tfidf-logreg-benchmark`. Il contient un run enfant direct
pour la baseline, un pour chaque essai Optuna, puis un pour le réglage du seuil.
Chaque essai journalise la provenance, la seed, le profil de prétraitement et
tous ses interrupteurs, les paramètres TF-IDF/LogReg, les métriques validation,
les temps et la taille du vocabulaire.

Le run parent conserve en plus :

- `validation_configuration_results.csv` ;
- `optuna_trial_results.csv` ;
- `optuna_search_config.json` ;
- `threshold_search_results.csv` ;
- `test_metrics.csv` ;
- `test_confusion_matrix.png` ;
- `test_classification_report.json` ;
- `decision_threshold.json` ;
- le pipeline scikit-learn retenu sous `selected_model`.

Le modèle n'est pas déclaré champion à ce stade : il est seulement marqué
`selected_for_test=true`. Le champion sera choisi lors de la comparaison finale
entre toutes les familles de modèles.
