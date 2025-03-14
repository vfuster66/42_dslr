# 🧙‍♂️ dslr - Le Chapeau Magique de Poudlard en Python !

Bienvenue sur le projet **dslr** !  
Ce projet a pour but de prédire dans **quelle maison de Poudlard** seront répartis les élèves, en recréant le **Chapeau Magique** grâce à un **modèle de Machine Learning**, codé **from scratch en Python**.

L'algorithme principal est une **régression logistique multi-classe** utilisant la méthode **One-vs-All**. Ce projet suit une méthodologie complète de **Data Science**, allant de la compréhension des données jusqu'à l'évaluation d'un modèle de classification.

🪄 **Objectif :** prédire dans quelle maison seront répartis les nouveaux élèves parmi :  
➡️ **Gryffondor**, **Poufsouffle**, **Serdaigle**, **Serpentard**.

---

## 🧭 Contexte

Le **Chapeau Magique** de Poudlard est défectueux ! Le professeur McGonagall fait appel à vous, un(e) **data scientist moldu(e)**, pour créer un algorithme capable de trier les élèves dans leur maison respective : **Gryffondor**, **Poufsouffle**, **Serdaigle** ou **Serpentard** .

Pour ce faire, vous utilisez les **données historiques** des élèves : leurs notes dans différentes matières, leurs comportements, etc.

---

## 🎯 Objectifs pédagogiques

Ce projet vise à :
- **Comprendre et manipuler des données réelles**
- **Implémenter une régression logistique multi-classe One-vs-All**
- **Apprendre à entraîner un modèle sans librairie ML**
- **Atteindre une précision d'au moins 98 % sur des données inconnues**

---

# ✅ Partie obligatoire - Étapes détaillées

Chaque étape est essentielle dans le pipeline d'un projet de Machine Learning :

---

## 1. 📂 Analyse des données

### Script : `describe.py`

🔍 **Ce que ça fait :**  
- **Charger** le dataset d'entraînement (`dataset_train.csv`).
- **Observer** la structure des données : colonnes, types de valeurs, valeurs manquantes.

### Concepts expliqués :
- **Count** : nombre de valeurs valides dans chaque colonne  
- **Mean (moyenne)** : la valeur moyenne de la colonne  
- **Standard Deviation (écart-type)** : mesure la dispersion des valeurs  
- **Min/Max** : valeur minimale et maximale  
- **Quartiles (25%, 50%, 75%)** : indiquent la distribution des données  
- **Skewness** : asymétrie de la distribution (bonus)  
- **Kurtosis** : aplatissement ou concentration de la distribution (bonus)

### Pourquoi c'est important :
Avant de modéliser quoi que ce soit, il faut **comprendre ses données**. Cette étape permet d'anticiper les problèmes (valeurs manquantes, colonnes inutiles, etc., données bien réparties et si certaines colonnes sont inutiles ou biaisées).

### Implémentation : `app/describe.py`
- Calcul **manuel** de statistiques descriptives :
  - Nombre d'échantillons
  - Moyenne
  - Écart-type
  - Minimum, maximum
  - Quartiles (25%, 50%, 75%)

🔧 **À quoi ça sert ?**
- Identifier les **variables discriminantes**.
- Détecter les **anomalies** (valeurs aberrantes, outliers).

#### Commande :
```bash
make describe
```

#### Résultats :
- Statistiques exportées en `.csv` et `.json` dans `data/describe/`.

---

## 2. 📊 Visualisation des données

La **visualisation** aide à comprendre les **relations entre variables** et à **choisir les features** pertinentes pour le modèle.

### 2.1. Histogrammes - `app/histogram.py`
🎨 **Ce que ça fait :**  
- Affiche la distribution des scores par matière, maison par maison.
- Affiche un **histogramme** des notes par matière.
- Cherche les cours où la répartition est **homogène** entre les maisons.

#### Concept :
- **Histogramme** : montre comment les données sont réparties sur une échelle (valeurs fréquentes ou rares).

🔧 **Pourquoi ?**
➡️ Déterminer si une matière est **pertinente pour différencier les maisons**.

#### Commande :
```bash
make histogram
```

### 2.2. Scatter Plots - `app/scatter_plot.py`

🎨 **Ce que ça fait :**
- Affiche des **nuages de points** entre deux variables.
- Cherche les deux features **les plus similaires** (corrélation).

#### Concept :
- **Scatter Plot** : montre la relation entre deux variables → permet de voir si elles sont **corrélées**.

🔧 **Pourquoi ?**
➡️ Identifier les **variables similaires** pour éviter la redondance dans l'entraînement du modèle.

#### Commande :
```bash
make scatter
```

### 2.3. Pair Plot - `app/pair_plot.py`
- Affiche une matrice complète de Scatter Plots et d'histogrammes pour **toutes les paires de variables**.
- Sélectionne les **features à utiliser pour l'entraînement**.

🔧 **Pourquoi ?**
➡️ Pour **sélectionner les features** les plus pertinentes pour l’entraînement.

#### Commande :
```bash
make pairplot
```

---

## 3. 🤖 Régression Logistique (One-vs-All)

### Qu’est-ce que la **régression logistique** ?  
➡️ Un algorithme **de classification** : il prédit une probabilité d'appartenance à une classe.  
➡️ Ici, il sert à prédire dans **quelle maison** ira l'élève.

---

### Qu’est-ce que le **One-vs-All** ?  
➡️ On transforme un problème **multi-classe** en **plusieurs problèmes binaires** :  
  - Le modèle apprend à dire "Maison Gryffondor : Oui ou Non ?", puis "Poufsouffle : Oui ou Non ?", etc.

---

### 3.1. Entraînement - `app/logreg_train.py`

🛠️ **Ce que ça fait :**  
- Entraîne **un modèle par maison** via **descente de gradient**  
- Minimise l'erreur entre prédiction et réalité  
- **Régularisation L2** : limite l’amplitude des poids pour éviter le **sur-apprentissage**
- Calcul des **poids** pour chaque classe.
- Sauvegarde des poids dans `data/logreg_model.npy`.

### Concepts expliqués :
- **Descente de gradient** : algorithme qui ajuste progressivement les poids pour **minimiser l'erreur**.  
- **Régularisation L2** : pénalise les poids trop grands pour **améliorer la généralisation**.

🔧 **Pourquoi ?**
- La descente de gradient ajuste les **paramètres** pour minimiser l'erreur entre les **prédictions** et la **réalité**.

#### Commande :
```bash
make train
```

### 3.2. Prédiction - `app/logreg_predict.py`

🛠️ **Ce que ça fait :**  
- Charge le modèle entraîné  
- Prédit la **maison** de chaque élève du `dataset_test.csv`

🔧 **Pourquoi ?**
- Générer des **prédictions** qu’on pourra **évaluer** ensuite.

#### Commande :
```bash
make predict
```

---

## 4. 🧮 Évaluation du modèle

📈 **Ce que ça fait :**  
- Calcule la **précision globale** (% de bonnes réponses)  
- Crée une **matrice de confusion** : montre où se trouvent les erreurs  
- Génère un **rapport de classification** : précision, rappel, F1-score

🔧 **Pourquoi ?**
- Évaluer objectivement les **performances** du modèle.
- Voir si le modèle est **équilibré** entre les différentes classes.

#### Commande :
```bash
make evaluate
```

---

# 🎁 Partie bonus - Fonctionnalités avancées

---

## 1. 🔹 Sélection manuelle des meilleures features

### Pourquoi ?
- Réduire la **dimensionnalité** des données pour éviter le **surapprentissage**.

### Ce qu’on a fait :
- Analyse des **pair plots** pour repérer les features **les plus discriminantes**.
- Stockage des meilleures features dans `data/best_features.txt`.

#### Commande :
```bash
make pairplot
```

---

## 2. 🔹 Statistiques avancées dans `describe.py`

### Pourquoi ?
- Comprendre plus finement la **distribution** des variables.

### Ce qu’on a ajouté :
- **Skewness** : asymétrie de la distribution.
- **Kurtosis** : aplatissement de la distribution.

#### Commande :
```bash
make describe
```

---

## 3. 🔹 Courbes ROC détaillées et Super ROC

### Pourquoi ?
- Visualiser les performances de chaque classifieur **One-vs-All**.

### Ce qu’on a ajouté :
- Courbes ROC pour chaque maison.
- Super ROC : toutes les courbes affichées sur le même graphe.

#### Commande :
```bash
make roc
make roc-all
```

---

## 4. 🔹 Mini-Batch Gradient Descent (MBGD)

### Pourquoi ?
- Accélérer la **descente de gradient** sur des jeux de données plus volumineux.

### Ce qu’on a fait :
- Implémentation d'une **descente en mini-lots** dans `logreg_train.py`.
- Gestion de la **taille des batchs** paramétrable (par défaut 32).

#### Commande :
```bash
make train-mbgd
```

---

## 🐳 Docker & Makefile intégrés

### Pourquoi ?
- Simplifier l'exécution des scripts.
- Garantir la **reproductibilité** sur tout environnement.

### Ce qu’on a :
- `Dockerfile` : environnement Python 3.10 avec dépendances.
- `Makefile` : automatisation des commandes Docker.

#### Commande :
Principales commandes :
```bash
make build
make shell
make describe
make train
```

---

## 📚 Ressources complémentaires

- [Coursera - Machine Learning par Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Khan Academy - Probabilités et Statistiques](https://fr.khanacademy.org/math/statistics-probability)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

