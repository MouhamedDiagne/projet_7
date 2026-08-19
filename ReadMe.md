Ce document est une feuille de routes ainsi qu'un journal de bord dans la réalisation du projet 7 sur l'analyse de sentiment du parcours AI Engineer d'OpenClassrooms.





##### **Description du Projet :**



Vous êtes ingénieur IA chez Marketing Intelligence Center. Votre boîte est missionnée pour créer un produit permettant d'anticiper les bad buzz sur Internet.



##### **Objectif :**



Concevoir un prototype d'application employant l'IA pour prédire le sentiment associé à un tweet.



##### **Données :**

&#x20;

Jeu de données Open Source : tweets, informations sur les tweets, label binaire (sentiment exprimé: positif vs négatif).



##### **Plan d'Action :**



* Exploration et Analyse des données (notebook).
* Modélisation et Benchmarking (notebook).
* Exposer une API de prédiction du sentiment à partir d'un tweet.
* Deployer le cet API sur le Cloud (GCP).
* Mettre en place des alertes lorsqu'un seuil critique de mauvaises prédictions est atteint.
* Mettre en place une interface locale (streamlit) pour faire appel à l'api
* Préparer une présentation du prototype ainsi que du processus de conception et de déploiement pour un public non-technique.



##### **Exploration et Analyse de Données** :



* Données : initialement contenues dans fichier csv.
* Import : on utilise Pandas pour importer le dataset. On prend en compte le choix de l'algorithme d'encodage des données.

  * "UTF-8" is not valid for reading the file unless we specify "ignore" or "replace" for the "errors" parameter.
  * "latin-1" est supporté mais cependant il transforme certains caractères en caractères de contrôle C1.
  * une autre méthode d'encodage utilisable est "cp1252" qui ne génère pas de caractères de contrôle.
  * on note cependant que le fichier csv semble ne pas avoir un encodage unique. On note la présence de caractères de remplacement "" en lisant le fichier. Ces caractères semblent eux encodés en "utf-8", alors que la lecture du fichier ne peut se faire en "utf-8". En utilisant "latin-1" ou "cp1252", ce caractère devient "ï¿½". Cette information mérite d'être prise en compte lors de l'analyse de données.
  * On décide d'utiliser l'encodage "cp1252" pour construire le Dataset que nous allons utiliser dans le reste de notre projet car il offre plus de stabilité et moins cas spéciaux à prendre en compte par rapport aux autres méthodes.
* Stratégie de Sampling \& Signification des Labels

  * Les données brutes contiennent 1.6 millions de lignes labelisées par la colonne target qui prend les valeurs 0 et 4.
  * Pour s'accommoder des moyens à disposition et anticiper sur les solutions de Hardware (local CPU ou cloud GPU) pour l'entrainement des modèles, on se limite à un échantillon limité mais significatif de 100.000 lignes avec des classes de labels équilibrées issues du dataset original.
  * On effectue également un remapping des labels (4 --> 1) pour mieux s'adapter à la modélisation binaire qui viendra dans la suite.
* Analyse Textuelle des Tweets : Longueur des tweets

  * une analyse de la longueur des tweets est pertinente car elle nous permet de mieux comprendre notre dataset.
  * on mesure : le nombre de caractères total des tweets, le nombre de mots (groupes de caractères séparés par un vide) par tweet, la taille moyenne des mots utilisés pour chaque tweet.
  * l'étude du nombre de mots par tweet peut nous permettre d'estimer la valeur optimale à prendre en compte pour le choix du paramètre "maxlen" (taille maximale) souvent nécessaire pour beaucoup d'algorithmes de NLP.
  * la taille moyenne des mots utilisés observée nous permet de poser la question des elongations, des liens et artifacts souvent utilisés dans les réseaux sociaux.
* Analyse des Artifacts Twitter (liens, mentions, elongations de mots, ponctuation, etc)

  * cette étape nous permet de voir quels types d'artifacts apparaissent fréquemment dans notre corpus : mentions, elongations, ponctuations spéciales ("?", "!").
  * cette analyse nous permet aussi de voir si la présence de ces artifacts peut avoir une influence sur le sentiment associé à un tweet.
  * grâce à cela, on peut décider si un type d'artifact peut être supprimé ou si on doit essayer de normaliser ou remplacer par un placeholder.
  * on constate que certains artifacts (urls, mentions, elongations, ponctuations-"!") apparaissent plus ou moins fréquemment selon le groupe de tweets (positif/négatif) que l'on observe. Une information que l'on prendra définitivement en compte lors du pre processing en amont des étapes de mise en embedding et de classification.
* Analyse Lexicale

  * on étudie maintenant le contenu lexical de notre corpus. On essaie de répondre à certaines questions telles que : quels sont les mots les plus fréquents dans notre corpus ? Existe-t-il des mots qui deviennent plus fréquent selon le sentiment associé aux tweets ?
  * en amont, on effectue une normalisation (principalement en remplaçant par des Placeholders) sur certains artifacts parmi ceux que nous avons répertorié précédemment(mentions, hashtags, lien) afin de relever uniquement les mots et expressions décrivant le contenu d'un tweet.
  * le premier résultat sur les mots les plus fréquents est prévisible : un classement dominé par les placeholders issus de notre traitement et surtout par des mots génériques de la langue généralement appelés "stopwords" (articles, pronoms, conjonctions, etc).
  * dans un second temps, on filtre ces stopwords. On calcule ensuite une différence de fréquence relative pour chaque mot et par rapport à chaque sentiment. Autrement dit, pour chaque sentiment, on calcule la fréquence d'apparition du mot dans l'ensemble des tweets associé à ce sentiment et on fait ensuite la différence entre les deux fréquences. On obtient alors une valeur qui nous permet de classer l'ensemble des mots de notre corpus pour trouver ceux qui apparaissent plus ou moins souvent d'un sentiment à l'autre.
  * En allant plus loin, on peut penser que les différences de prévalence que l'on vient de calculer ne permettent pas de trouver les termes dont la fréquence d'apparition varie de manière significative entre les 2 classes de sentiment. On répond à cela en divisant la différence précédente par la valeur absolue maximale entre les deux valeurs de prévalence du terme par rapport à chaque classe de sentiment. En observant les valeurs extrêmes de cette nouvelle mesure, on trouve les mots qui apparaissent en grande majorité, et parfois exclusivement, dans les tweets associés à un unique sentiment.
  * on réalise une étude similaire sur les expressions (bigrammes et trigrammes) afin de relever les groupes de mots les plus fréquents et leurs comportements selon la classe de sentiment.
  * Résultats :

    * grâce au calcul de la *"différence de prévalence relative"* pour tous les termes(ou expressions), nous avons pu relever les mots(ou groupe de mots) dont les fréquences d'apparition par rapport aux 2 classes de tweet sont les plus désequilibrées.
    * en observant ces termes, on observe qu'ils sont souvent des mots portant une intention ou un sentiment très clairement orientées vers une classe(positive ou négative). On peut se demander alors si, en reconnaissant ces termes ou expressions, certains modèles pourront effectuer la prédiction de la classe d'un tweet. Est-ce que cela mènera à des performances améliorées ou est-ce possible d'en arriver à des situations d'overfitting à cause de cela ?
    * On observe aussi que les valeurs de *différence de prévalence relative* sont souvent plus grandes en valeur absolue. Donc, on pourrait poser l'hypothèse que la prévalence des termes ou expressions négatives dans les tweets négatifs est plus forte que celle des termes et expressions positive dans les tweets positifs. Cela pourrait-il induire de meilleures performance dans la classification pour les individus de la classe négative ?
    * on confirme l'influence des artifacts, notamment les mentions, dans la classification en sentiment des tweets. Beaucoup de bigrammes ou trigrammes incluant le placeholder *user* se retrouvent dans les expressions ayant les valeurs de *différence de prévalence relative* les plus extrêmes et donc apparaissent très majoritairement dans une classe de sentiment.





##### **Modélisation \& Benchmarking** :



* Dependencies / Setup
* Data Import

  * with cp1252 encoding as done in the EDA
  * fixed random seed.
  * label remapping
  * balanced sampling of 125\_000 lines
* Train / Validation / Test Split

  * stratified split
  * rations 0.6 / 0.2 / 0.2
* Performance Metrics

  * between different epochs of training, for the model to optimize its parameters ---> training loss
  * during validation, and for comparing different models or different hyperparameter configurations of a same model ---> accuracy, recall\_negative, fbeta\_negative,
  * for business purposes, to see the impact of each model on the problem we want to solve (badbuzz) --> customized metric : for 1\_000 negative tweets, how much are missed by the model ?
* Shared Benchmarking Protocol

  * Comparison Scope

    * one same dataset of 125\_000 for all models, with the same fixed train / validation / test split
    * for each model type, experiments are made to optimize the hyperparameters, the best model configuration is selected using the validation set.
    * the test set is used at the end to evaluate the selected models
  * performance measurement / model comparison metrics

    * Primary Metrics : fbeta\_negative (beta=3), badbuzz\_miss\_per\_1000\_negative\_tweets
    * Secondary Metrics : Accuracy, Precision\_negative, recall\_negative
    * Additionnal Metrics : training time, inference time
  * tracking method and protocol

    * framework : MLFlow
    * method :

      * one experiment for the whole benchmark
      * one run for each model type and one sub run for each model configuration
      * for each model type, one run is selected using the specified model selection rule and threshold policies.
      * each run/sub run must log : model family, model name, preprocessing type and settings, embedding model type and parameters, classifier model type and parameters, Validation Metrics, Test Metrics for the selected runs only, training time, inference time, confusion matrix artifact for selected runs, learning curves for neural networks.
      * 
  * Model Selection Rules

    * during each model hyperparameter optimization, the best model must be chosen.
    * first we do not consider models that have : less than 60% validation accuracy,
    * then we rank the models by fbeta\_score
    * then the badbuzz\_miss\_per\_1000 metric serves as first tie-breaker and accuracy as the second tie breaker
  * Threshold Policy

    * we chose to tune model thresholds to optimize the selected model's performances
    * use the default model threshold first
    * threshold tuning on the validation set
    * freezing the threshold after the choice is made.
  * generic experiment process for each model

    * model-specific preprocessing (cleaning, etc)
    * embedding
    * training / validation on one baseline model setup
    * hyperparameter optimization (training + validation on multiple configurations of the model) - with experiment tracking
    * tracking all metrics, hyperparameters and metadata during the optimization
    * best model selection by validation metrics using "Model Selection Rules" 
    * threshold policy implementation.
    * final reporting includes the chosen model’s validation metrics and one final evaluation on the untouched test set.
  * reporting format

    * per model section : preprocessing choices, (key) hyperparameters, validation metrics, test metrics and confusion matrix of the selected model, interpretation of the model's performances
    * final benchmark synthesis : summary dataframe with one row by selected model, columns : model family, metrics.
* TF-idf + Logistic Regression
* GloVe + LSTM
* Word2Vec + GRU
* BERT + DNN
* Final Benchmark Comparison
* Selected Model \& Next Steps

