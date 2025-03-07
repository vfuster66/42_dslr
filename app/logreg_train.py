#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
import os


def load_best_features(filepath="data/best_features.txt"):
    """Charge dynamiquement les meilleures
    caractéristiques identifiées précédemment."""
    try:
        with open(filepath, "r") as f:
            features = [line.strip() for line in f.readlines()]
        print(f"🔍 Caractéristiques sélectionnées : {features}")
        return features
    except Exception as e:
        print(f"❌ Erreur lors du chargement des meilleures "
              f"caractéristiques : {e}")
        sys.exit(1)


def load_data(filepath):
    """
    Charge les données, sélectionne les colonnes pertinentes et normalise
    les valeurs numériques.
    Retourne le DataFrame, la liste des caractéristiques sélectionnées,
    ainsi que la moyenne et l’écart-type calculés sur le jeu d’entraînement.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"📊 Train -> Fichier chargé avec {len(df)} lignes "
              f"et {len(df.columns)} colonnes.")
        print(f"📊 Train -> Colonnes chargées : {df.columns.tolist()}")

        if "Hogwarts House" not in df.columns:
            raise ValueError(
                "Le dataset doit contenir une colonne 'Hogwarts House'."
                )

        selected_features = load_best_features()

        df = df[["Hogwarts House"] + selected_features].dropna()

        df[selected_features] = df[selected_features].apply(
            pd.to_numeric, errors='coerce')

        mean_train = df[selected_features].mean()
        std_train = df[selected_features].std()

        df[selected_features] = (df[selected_features] - mean_train) / \
            std_train

        return df, selected_features, mean_train, std_train
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        sys.exit(1)


def encode_labels(df):
    """Encode les maisons de Poudlard en labels numériques One-vs-All."""
    houses = sorted(df["Hogwarts House"].unique())
    label_dict = {house: i for i, house in enumerate(houses)}
    df["House Label"] = df["Hogwarts House"].map(label_dict)
    return df, label_dict


def sigmoid(z):
    """Fonction sigmoïde."""
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, lambda_):
    """Calcule le coût avec régularisation."""
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg


def gradient_descent(X, y, alpha=0.01, lambda_=0.1, iterations=7000):
    """Effectue la descente de gradient pour ajuster theta."""
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (X.T @ (h - y)) / m
        gradient[1:] += (lambda_ / m) * theta[1:]
        theta -= alpha * gradient

        if i % 100 == 0:
            cost = cost_function(theta, X, y, lambda_)
            cost_history.append(cost)
            print(f"📉 Itération {i}/{iterations} - Coût : {cost:.4f}")

    return theta, cost_history


def train_one_vs_all(X, y, num_labels, alpha=0.01, lambda_=0.1,
                     iterations=7000):
    """Entraîne un modèle de régression logistique One-vs-All."""
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))
    for i in range(num_labels):
        print(f"🚀 Entraînement du modèle pour la maison {i}...")
        y_i = (y == i).astype(int)
        all_theta[i], _ = gradient_descent(X, y_i, alpha, lambda_, iterations)
    return all_theta


def save_model(theta, label_dict, mean_train, std_train,
               filepath="data/logreg_model.npy"):
    """Sauvegarde les poids du modèle et les paramètres de
    normalisation dans un fichier numpy."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, {
        "theta": theta,
        "labels": label_dict,
        "mean": mean_train,
        "std": std_train
    })
    print(f"✅ Modèle sauvegardé : {filepath}")


def evaluate_model(X, y, theta):
    """Évalue la performance du modèle sur les données d'entraînement."""
    predictions = np.argmax(sigmoid(X @ theta.T), axis=1)
    accuracy = np.mean(predictions == y) * 100
    print(f"🎯 Précision du modèle sur les données d'entraînement : "
          f"{accuracy:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df, selected_features, mean_train, std_train = load_data(file_path)
    df, label_dict = encode_labels(df)

    X = df[selected_features].values

    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = df["House Label"].values

    num_labels = len(label_dict)
    theta = train_one_vs_all(X, y, num_labels, iterations=7000)

    save_model(theta, label_dict, mean_train, std_train)
    evaluate_model(X, y, theta)

    print("🎯 Entraînement terminé !")
