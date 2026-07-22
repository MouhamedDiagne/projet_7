Ce document est une présentation globale du projet 7 sur l'analyse de sentiment du parcours AI Engineer d'OpenClassrooms.





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
  * dans un second temps, on filtre ces stopwords. On calcule ensuite une différence de fréquence relative pour chaque mot et par rapport à chaque sentiment. Autrement dit, pour chaque sentiment, on calcule la fréquence d'apparition du mot dans l'ensemble des tweets associé à ce sentiment et on fait ensuite la différence entre les deux fréquences. On obtient alors une valeur qui nous permet de classer l'ensemble des mots de notre corpus pour trouver ceux qui apparaissent plus ou moins souvent en passant d'un sentiment à l'autre.
  * &#x20;





##### **Modélisation \& Benchmarking** :





