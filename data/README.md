# Données : Sentiment140

Le benchmark utilise le corpus d'entraînement **Sentiment140** contenant
1 600 000 tweets annotés automatiquement par supervision distante. Le fichier
brut n'est pas versionné dans ce dépôt, car il est volumineux et soumis aux
conditions d'utilisation de sa source.

## Source et citation

- Archive source : <https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip>
- Page du jeu de données : <http://help.sentiment140.com/home>
- Référence : Go, A., Bhayani, R. & Huang, L. (2009), *Twitter Sentiment
  Classification using Distant Supervision*.

Le fichier attendu est `training.1600000.processed.noemoticon.csv`, fourni dans
l'archive sous `training.1600000.processed.noemoticon.csv`.

Avant de le télécharger ou de le redistribuer, l'utilisateur doit vérifier les
conditions applicables à Sentiment140 et aux contenus Twitter/X. Aucune licence
machine-readable n'est déclarée par la source de référence ; le fichier ne doit
donc pas être ajouté à Git.

## Installation locale

Depuis la racine du dépôt, créer le dossier attendu puis télécharger et extraire
l'archive :

```powershell
New-Item -ItemType Directory -Force data\sentiment140 | Out-Null
Invoke-WebRequest `
  -Uri https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip `
  -OutFile data\sentiment140\trainingandtestdata.zip
Expand-Archive `
  -LiteralPath data\sentiment140\trainingandtestdata.zip `
  -DestinationPath data\sentiment140 -Force
Remove-Item -LiteralPath data\sentiment140\trainingandtestdata.zip
```

Le notebook attend ensuite le chemin suivant :

```text
data/sentiment140/training.1600000.processed.noemoticon.csv
```

## Provenance enregistrée

Le notebook vérifie que le fichier attendu existe et porte le nom prévu avant
tout échantillonnage. Il journalise dans MLflow son nom, sa taille et son
encodage `cp1252`, jamais le corpus brut.
