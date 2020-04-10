# PREDICTION.ML.TITANIC
L’objectif de ce brief projet est de retracer scientifiquement l’histoire du naufrage du Titanic en
utilisant les données disponibles sur le Kaggle.
Il s’agit d’un jeu de données public très facile d’accès et qui possède plusieurs vertus pédagogiques.
Bien que l’analyse des données liées au naufrage du Titanic n’a aucun intérêt métier, les données sont
riches pour pouvoir mettre en pratique les techniques et les modèles que nous avons abordés les
dernières semaines.
Pour cela, il vous est demandé de mettre en œuvre une démarche complète d’exploitation de données
allant de la compréhension du besoin jusqu’à l’évaluation des modèles élaborés en passant par une
phase de préparation et d’analyse de données.
I. MISE EN ŒUVRE
Dans ce qui suit, vous trouverez une trame à suivre pour mener à bien ce projet.
Cette trame a valeur d’exemple, il n’est pas obligatoire de la suivre scrupuleusement.
1. PREPARATION DES DONNEES
Les données sont à télécharger sur Kaggle Titanic. Vous disposez de deux jeux de données :
- Train : jeu d’apprentissage
- Test : jeu de test

TRAVAIL DEMANDE
1. Importer les deux jeux de données.
2. Transformer les tables en data frame.
3. Analyser la signification de chaque variable.
4. Afficher le type de chacune des variables ainsi que le nombre de valeurs nulles par variables.
5. Supprimer les observations incomplètes.
2. EXPLORATION DES DONNEES
Avant d’aller plus loin, il est essentiel d’explorer les données pour les comprendre et les utiliser de
manière efficiente.
DATA VISUALISATION
1. Pour chacune des variables suivantes, créer un ou plusieurs diagrammes qui la résume au
mieux :
- Alive
- Age
- Sex
- Class
- Fare
2. Analyser les diagrammes et en tirer des conclusions et/ou des hypothèses
TESTS D’HYPOTHESES
LES FEMMES ET LES ENFANTS D’ABORD
1. Faire un test d’hypothèse pour savoir si oui ou non, les enfants ont été privilégiés lors du
naufrage.
2. Faire un test d’hypothèse pour vérifier si oui ou non, les femmes ont été privilégiées lors du
naufrage.
3. Conclure.
L’INFLUENCE DU PRIX DU BILLET SUR LA SURVIE DES PASSAGERS
1. Faire un test d’hypothèse pour savoir si oui ou non, le prix du billet a une influence sur la survie
d’un passager.
2. Conclure.
3. MACHINE LEARNING
L’objectif est de trouver un modèle qui nous permettra de prédire si un passager est mort ou vivant
en se basant sur les données du dataset.
REGRESSION LINEAIRE
1. Elaborer des modèles de régression linaire pour prédire la variable (target) alive. En
sélectionnant comme variables explicatives (features) :
a. age, sex, class, fare prises individuellement
b. age, sex, class, fare
c. différentes combinaisons de age, sex, class, fare
d. ayant une forte corrélation avec la variable alive
2. Evaluer chaque modèle de régression.
3. Conclure.
REGRESSION LOGISTIQUE


#SOMMAIRE
#Nous avons commencé par l'exploration des données où nous avons eu une idée de l'ensemble de données,
#vérifié les données manquantes et appris quelles fonctionnalités sont importantes. 
#Au cours de ce processus, nous avons utilisé seaborn et matplotlib pour effectuer les visualisations.
#Au cours de la partie de prétraitement des données, nous avons calculé les valeurs manquantes, 
#converti les entités en valeurs numériques, regroupé les valeurs en catégories et créé quelques nouvelles entités.
#Ensuite, nous avons commencé à former 4 modèles d'apprentissage automatique différents, 
#en avons choisi un (forêt aléatoire) et y avons appliqué une validation croisée. 
#Ensuite, nous avons discuté du fonctionnement de la forêt aléatoire, 
#examiné l'importance qu'elle attribue aux différentes fonctionnalités et optimisé ses performances en optimisant ses valeurs hyperparamétriques. 
#Enfin, nous avons examiné sa matrice de confusion et calculé la précision, le rappel et le f-score du modèle.


####TESTE A 4 VARIABLE PREDICTION D'UN PASSAGER ####
#derniere faire recréé un test de model regression logistiqque qui pour moi me semblais beaucoup plus fiable pour ce cas que la regression linear .
#cette fois ci on a utiliser que 4 variable qui sont lié entre elle avec le model deja entrainé on a pue effectuer la recherche d'un des passagers 
#exemple: clement, on a recupré sa classe, le pris de billet ,le tranche age, sont sex. 
#le model nous prédit 0 . 
#donc qu'il ne survivait pas a l'accident.
