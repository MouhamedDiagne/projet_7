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

## Contrôle d'intégrité

Le CSV de référence attendu par ce projet est encodé en `cp1252`, mesure
238 803 811 octets et a l'empreinte SHA-256 suivante :

```text
08cba6317a49528fcd074f9043aafcd5ad6c6be45ede159c4e36cec33af24afe
```

Vérifier localement après extraction :

```powershell
(Get-FileHash `
  data\sentiment140\training.1600000.processed.noemoticon.csv `
  -Algorithm SHA256).Hash.ToLower()
```

Le notebook applique ce contrôle avant tout échantillonnage. Il ne journalise
dans MLflow que le nom, la taille, l'encodage et l'empreinte du fichier — jamais
le corpus brut.
