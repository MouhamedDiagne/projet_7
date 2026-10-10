# Architecture et règles de conservation

## Objet

Ce document fixe les décisions d'architecture pour l'intégration de Databricks
Free Edition au benchmark Sentiment140. Elles doivent être respectées par les
notebooks et scripts exécutés sur Kaggle.

## Architecture retenue

| Composant | Responsabilité | Persistance |
|---|---|---|
| Dépôt local | Code versionné, documentation et configuration non sensible | Git |
| Kaggle | Données attachées, entraînement GPU et fichiers temporaires | Éphémère |
| Databricks Free Edition | Serveur MLflow : expériences, runs, métriques et artifacts | Persistante |

Databricks est utilisé comme serveur de suivi MLflow distant. Il ne sert pas à
l'entraînement : les calculs restent exécutés sur les GPU Kaggle.

## Décisions

1. **Jeu de données** — Le fichier CSV brut Sentiment140 ne sera pas envoyé à
   Databricks. Chaque run journalise à la place un manifeste de provenance :
   source et nom du jeu de données, empreinte SHA-256, encodage `cp1252`,
   stratégie d'échantillonnage, seed, définition du split, tailles des jeux et
   distribution des classes.

2. **Artifacts légers** — Toutes les configurations journalisent leurs
   paramètres, métriques, configuration complète, environnement d'exécution,
   courbes et rapports produits par le benchmark. La matrice de confusion est
   journalisée lorsqu'elle est pertinente à l'analyse.

3. **Artifacts lourds** — Les poids complets, checkpoints et modèles
   sérialisés ne sont conservés dans Databricks que pour les configurations
   retenues comme candidates sérieuses, puis pour les modèles sélectionnés.
   Les essais écartés ne doivent pas encombrer le stockage avec des poids ou
   checkpoints intermédiaires.

4. **Connectivité** — L'accès Internet doit être activé dans le notebook
   Kaggle. La communication Kaggle vers Databricks s'effectue en HTTPS. Un
   micro-run MLflow sera exécuté avant tout entraînement GPU afin de valider
   authentification, écriture des métriques et dépôt d'un artifact.

5. **Secrets** — Le jeton Databricks est stocké uniquement dans Kaggle Secrets.
   Pour les vérifications locales, il est lu depuis un fichier `.env` ignoré
   par Git. Aucun secret ne doit être ajouté au code, aux sorties de notebook,
   aux artifacts ou au dépôt.

6. **Résilience** — La connexion initiale au serveur de suivi est bloquante :
   aucun benchmark ne commence sans tracking fonctionnel. Après cette
   validation, les envois d'artifacts bénéficient de tentatives de reprise
   limitées pour absorber les incidents réseau transitoires.

7. **Organisation MLflow** — Une campagne de benchmark utilise un experiment
   unique, versionné dans son nom, par exemple
   `sentiment140-benchmark-v1`. Les runs sont organisés par famille de modèles
   puis par configuration. Les tags distinguent notamment les essais
   `candidate`, `selected_for_test` et `champion`.

## Conséquence opérationnelle

Le suivi doit être exhaustif pour les informations de comparaison, tout en
restant sélectif sur la conservation des fichiers volumineux. Cette politique
préserve la reproductibilité du benchmark et reste adaptée aux quotas de
Databricks Free Edition.
