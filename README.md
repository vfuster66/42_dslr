# 🧙‍♂️ dslr - Le Chapeau Magique de Poudlard en Python !

Bienvenue sur le projet **dslr** !  
Ce projet a pour but de prédire dans **quelle maison de Poudlard** seront répartis les élèves, en recréant le **Chapeau Magique** grâce à un **modèle de Machine Learning**, codé **from scratch en Python**.

L'algorithme principal est une **régression logistique multi-classe** utilisant la méthode **One-vs-All**. Ce projet suit une méthodologie complète de **Data Science**, allant de la compréhension des données jusqu'à l'évaluation d'un modèle de classification.

---

## 🧭 Contexte

Le **Chapeau Magique** de Poudlard est défectueux ! Le professeur McGonagall fait appel à vous, un(e) **data scientist moldu(e)**, pour créer un algorithme capable de trier les élèves dans leur maison respective : **Gryffondor**, **Poufsouffle**, **Serdaigle** ou **Serpentard**.

Pour ce faire, vous utilisez les **données historiques** des élèves : leurs notes dans différentes matières, leurs comportements, etc.

---

## 🎯 Objectifs pédagogiques

Ce projet vise à :
- Implémenter une **régression logistique multi-classe** sans bibliothèque de Machine Learning haut niveau.
- Apprendre à **analyser**, **nettoyer**, **visualiser** et **modéliser** des données.
- Comprendre le cycle complet d'un projet de **classification supervisée**.

---

# ✅ Partie obligatoire - Étapes détaillées

Chaque étape est essentielle dans le pipeline d'un projet de Machine Learning :

---

## 1. 📂 Analyse des données

### Ce qu’on fait :
- **Charger** le dataset d'entraînement (`dataset_train.csv`).
- **Observer** la structure des données : colonnes, types de valeurs, valeurs manquantes.

### Pourquoi c'est important :
Avant de modéliser quoi que ce soit, il faut **comprendre ses données**. Cette étape permet d'anticiper les problèmes (valeurs manquantes, colonnes inutiles, etc.).

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

#### Exemple d'utilisation :
```bash
make describe
```

#### Résultats :
- Statistiques exportées en `.csv` et `.json` dans `data/describe/`.

---

## 2. 📊 Visualisation des données

La **visualisation** aide à comprendre les **relations entre variables** et à **choisir les features** pertinentes pour le modèle.

### 2.1. Histogrammes - `app/histogram.py`
- Affiche un **histogramme** des notes par matière.
- Cherche les cours où la répartition est **homogène** entre les maisons.

🔧 **Pourquoi ?**
- Permet de détecter si certaines matières sont **plus ou moins discriminantes**.

```bash
make histogram
```

### 2.2. Scatter Plots - `app/scatter_plot.py`
- Affiche des **nuages de points** entre deux variables.
- Cherche les deux features **les plus similaires** (corrélation).

🔧 **Pourquoi ?**
- Comprendre quelles features sont **corrélées**, et ainsi éviter la **redondance** dans les variables.

```bash
make scatter
```

### 2.3. Pair Plot - `app/pair_plot.py`
- Génère une **matrice de scatter plots** pour observer toutes les relations.
- Sélectionne les **features à utiliser pour l'entraînement**.

🔧 **Pourquoi ?**
- Aide à faire une **sélection de features** basée sur l'observation visuelle des distributions et relations.

```bash
make pairplot
```

---

## 3. 🤖 Régression Logistique (One-vs-All)

### Ce qu’on fait :
- Implémenter un **classifieur multi-classe** en One-vs-All avec **régression logistique**.
- Chaque maison devient une **classe**, et le modèle apprend à **prédire la probabilité** qu'un élève appartienne à cette maison.

### 3.1. Entraînement - `app/logreg_train.py`
- Implémentation de la **descente de gradient** classique.
- Calcul des **poids** pour chaque classe.
- Sauvegarde des poids dans `data/logreg_model.npy`.

🔧 **Pourquoi ?**
- La descente de gradient ajuste les **paramètres** pour minimiser l'erreur entre les **prédictions** et la **réalité**.

```bash
make train
```

### 3.2. Prédiction - `app/logreg_predict.py`
- Utilise les poids appris pour **prédire** la maison de chaque élève du `dataset_test.csv`.
- Produit le fichier `houses.csv`.

🔧 **Pourquoi ?**
- Générer des **prédictions** qu’on pourra **évaluer** ensuite.

```bash
make predict
```

---

## 4. 🧮 Évaluation du modèle

### Ce qu’on fait :
- Évaluer la qualité de nos **prédictions** sur les données de test.

### Implémentation - `app/logreg_evaluate.py`
- Génération de **matrices de confusion**, **rapports de classification** et **courbes ROC** pour chaque maison.
- Vérification que le modèle atteint **98% de précision** (critère du sujet).

🔧 **Pourquoi ?**
- Évaluer objectivement les **performances** du modèle.
- Voir si le modèle est **équilibré** entre les différentes classes.

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

## 🙌 Auteurs

Projet réalisé dans le cadre de l'école **42**, module **dslr**.

- Auteur : [Ton nom ici]
- Année : 2024

---

