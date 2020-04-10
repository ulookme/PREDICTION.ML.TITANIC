

import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
full = pd.concat([train, test]) # Assemble les deux jeux de données
    


# Ticket & Prix  

Tout d’abord regardons la caractéristique Ticket. Vérifions que tous les passagers ont bien un ticket :
noticket = []
full['Ticket'].fillna('X')
for ticketnn in full['Ticket']:
    if (ticketnn == 'X'):
        noticket.append(1)
    else:
        noticket.append(0)
pd.DataFrame(noticket)[0].value_counts()

Bonne nouvelle, tous les passagers possèdent cette information !
test['Ticket'].value_counts().head()


# Calcul du prix unitaire du billet

Pour ce faire nous allons utiliser les capacités de Pandas a effectuer des jointures (gauche) entres DataFrame. Au préalable nous allons constituer un DataFrame qui regroupe les Tickets avec leur nombre d’occurences : TicketCounts. Ensuite nous ferons une jointure gauche entre le jeu de données et ce nouveau DataFrame. Nous n’aurons ensuite plus qu’à ajouter une colonne PrixUnitaire qui divise le prix total par le nombre de personne sur le Ticket. Attention ici de bien utiliser la fonction fillna() sur le nombre de ticket.

# Prépartion d'un DF (TicketCounts) contenant les ticket avec leur nb d'occurence
TicketCounts = pd.DataFrame(test['Ticket'].value_counts().head())
TicketCounts['TicketCount'] = TicketCounts['Ticket'] # renomme la colonne Ticket
TicketCounts['Ticket'] = TicketCounts.index # rajoute une colonne Ticket pour le merge (jointure)
 
# Reporte le résultat dans le dataframe test (jointure des datasets)
fin = pd.merge(test, TicketCounts, how='left', on='Ticket')
fin['PrixUnitaire'] = fin['Fare'] / fin['TicketCount'].fillna(1)


# Passager sans Prix !

Attention, car nous avons aussi un passager qui n’a pas de Prix. Regardons de qui il s’agit :
import numpy as np
test.loc[np.isnan(test['Fare'])]

Il s’agit d’un passager de 3ème classe, calculons donc le prix moyen de ce type de billet:
test.loc[test['Pclass'] == 3]['Fare'].mean()
12.459677880184334
# Nous affecterons ce prix à ce passager.












Le Titre

De la même manière que le nom de famille, nous devons extraire le titre en parsant la caractéristique Name. Regardons les titres sur l’ensemble du jeu de données (full) :
full['Titre'] = full.Name.map(lambda x : x.split(",")[1].split(".")[0])
full['NomFamille'] = full.Name.map(lambda x : x.split(",")[0])
titre = pd.DataFrame(full['Titre'])
full['Titre'].value_counts() # affiche tous les titres possible

# Regroupement des titres
#Créons donc 3 catégories : Femme et enfant, VIP et les autres :
X = test
X['Rang'] = 0
X['Titre'] = X.Name.map(lambda x : x.split(",")[1].split(".")[0])
vip = ['Don','Sir', 'Major', 'Col', 'Jonkheer', 'Dr', 'Rev']
femmeenfant = ['Miss', 'Mrs', 'Lady', 'Mlle', 'the Countess', 'Ms', 'Mme', 'Dona', 'Master']
for idx, titre in enumerate(X['Titre']):
    if (titre.strip() in femmeenfant) :
        X.loc[idx, 'Rang'] = 'FE'
    elif (titre.strip() in vip) :
        X.loc[idx, 'Rang'] = 'VIP'
    else :
        X.loc[idx, 'Rang'] = 'Autres'
X['Rang'].value_counts()


Relation Titre / classe / survivant ?


Calcul du prix unitaire du billet
Pour ce faire nous allons utiliser les capacités de Pandas a effectuer des jointures (gauche) entres DataFrame. Au préalable nous allons constituer un DataFrame qui regroupe les Tickets avec leur nombre d’occurences : TicketCounts. Ensuite nous ferons une jointure gauche entre le jeu de données et ce nouveau DataFrame. Nous n’aurons ensuite plus qu’à ajouter une colonne PrixUnitaire qui divise le prix total par le nombre de personne sur le Ticket. Attention ici de bien utiliser la fonction fillna() sur le nombre de ticket.

# Prépartion d'un DF (TicketCounts) contenant les ticket avec leur nb d'occurence
TicketCounts = pd.DataFrame(test['Ticket'].value_counts().head())
TicketCounts['TicketCount'] = TicketCounts['Ticket'] # renomme la colonne Ticket
TicketCounts['Ticket'] = TicketCounts.index # rajoute une colonne Ticket pour le merge (jointure)

# Reporte le résultat dans le dataframe test (jointure des datasets)
fin = pd.merge(test, TicketCounts, how='left', on='Ticket')
fin['PrixUnitaire'] = fin['Fare'] / fin['TicketCount'].fillna(1)



# Catégories d'âge

    Les bébés : de 0 a 3 ans
    Les enfants: de 3 à 15 ans
    Les adultes de 15 à 60 ans
    Les « vieux » de plus de 60 ans

age = X['Age'].fillna(X['Age'].mean())
catAge = []
for i in range(X.shape[0]) :
    if age[i] <= 3:
       catAge.append("bebe")
    elif age[i] > 3 and age[i] >= 15:
        catAge.append("enfant")
    elif age[i] > 15 and age[i] <= 60:
        catAge.append("adulte")
    else:
        catAge.append("vieux")
print(pd.DataFrame(catAge, columns = ['catAge'])['catAge'].value_counts())
cat = pd.get_dummies(pd.DataFrame(catAge, columns = ['catAge']), prefix='catAge')
cat.head(3)




full = pd.concat([train, test]) # Assemble les deux jeux de données


# Tous les passagers ont-ils un ticket ?
noticket = []
full['Ticket'].fillna('X')
for ticketnn in full['Ticket']:
    if (ticketnn == 'X'):
        noticket.append(1)
    else:
        noticket.append(0)
pd.DataFrame(noticket)[0].value_counts()



def dataprep(data):
    # Sexe
   sexe = pd.get_dummies(data['Sex'], prefix='sex')
  ²
    # Cabine, récupération du pont (on remplace le pont T proche du pont A)
    cabin = pd.get_dummies(data['Cabin'].fillna('X').str[0].replace('T', 'A'), prefix='Cabin')
     
    # Age et catégories d'age
    age = data['Age'].fillna(data['Age'].mean())
    catAge = []
    for i in range(data.shape[0]) :
        if age[i] &amp;gt; 3:
           catAge.append("bebe")
        elif age[i] &amp;gt;= 3 and age[i] &amp;lt; 15:
            catAge.append("enfant")
        elif age[i] &amp;gt;= 15 and age[i] &amp;lt; 60:
            catAge.append("adulte")
        else:
            catAge.append("vieux")
    catage = pd.get_dummies(pd.DataFrame(catAge, columns = ['catAge']), prefix='catAge')
	     
    # Titre et Rang
	    data['Titre'] = data.Name.map(lambda x : x.split(",")[1].split(".")[0]).fillna('X')
	    data['Rang'] = 0
	    vip = ['Don','Sir', 'Major', 'Col', 'Jonkheer', 'Dr']
	    femmeenfant = ['Miss', 'Mrs', 'Lady', 'Mlle', 'the Countess', 'Ms', 'Mme', 'Dona', 'Master']
	    for idx, titre in enumerate(data['Titre']):
	        if (titre.strip() in femmeenfant) :
	            data.loc[idx, 'Rang'] = 'FE'
	        elif (titre.strip() in vip) :
	            data.loc[idx, 'Rang'] = 'VIP'
	        else :
	            data.loc[idx, 'Rang'] = 'Autres'
	rg = pd.get_dummies(data['Rang'], prefix='Rang')

	# Embarquement
    emb = pd.get_dummies(data['Embarked'], prefix='emb')
    
    # Prix unitaire - Ticket, Prépartion d'un DF (TicketCounts) contenant les ticket avec leur nb d'occurence
    TicketCounts = pd.DataFrame(data['Ticket'].value_counts())
    TicketCounts['TicketCount'] = TicketCounts['Ticket'] # renomme la colonne Ticket
    TicketCounts['Ticket'] = TicketCounts.index # rajoute une colonne Ticket pour le merge (jointure)
    # reporte le résultat dans le dataframe test (jointure des datasets)
    fin = pd.merge(data, TicketCounts, how='left', on='Ticket')
    fin['PrixUnitaire'] = fin['Fare'] / fin['TicketCount'].fillna(1)
    prxunit = pd.DataFrame(fin['PrixUnitaire'])
    # Prix moyen 3eme classe (pour le passager de 3eme qui n'a pas de prix)
    prx3eme = data.loc[data['Pclass'] == 3]['Fare'].mean()
    prxunit = prxunit['PrixUnitaire'].fillna(prx3eme)
    
    # Classe
    pc = pd.DataFrame(MinMaxScaler().fit_transform(data[['Pclass']]), columns = ['Classe'])
     
   dp = data[['SibSp', 'Parch', 'Name']].join(pc).join(sexe).join(emb).join(prxunit).join(cabin).join(age).join(catage).join(rg)
   addColumnFamilyName(dp)
    del dp['Name']
     
    return dp

#Entrainement du modèle
Xtrain = dataprep(train)
Xtest = dataprep(test)

y = train.Survived
clf = LinearSVC(random_state=4)
clf.fit(Xtrain, y)
p_tr = clf.predict(Xtrain)
print ("Score Train : ", round(clf.score(Xtrain, y) *100,4), " %")

# Algorithme de Random Forest
y = train.Survived
rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(Xtrain, y)
p_tr = rf.predict(Xtrain)
print ("Score Train -- ", round(rf.score(Xtrain, y) *100,2), " %")

# Et sur le fichier de test...
p_test = rf.predict(Xtest) 


