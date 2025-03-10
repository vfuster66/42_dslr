#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
import os


def load_best_features(filepath="data/best_features.txt"):
    try:
        with open(filepath, "r") as f:
            features = [line.strip() for line in f.readlines()]
        print(f"🔍 Caractéristiques sélectionnées : {features}")
        return features
    except Exception as e:
        print(
            f"❌ Erreur lors du chargement des meilleures caractéristiques : "
            f"{e}"
            )
        sys.exit(1)


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"📊 Train -> Fichier chargé avec {len(df)} lignes et "
              f"{len(df.columns)} colonnes.")
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

        df[selected_features] = (
            df[selected_features] - mean_train) / std_train

        return df, selected_features, mean_train, std_train
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        sys.exit(1)


def encode_labels(df):
    houses = sorted(df["Hogwarts House"].unique())
    label_dict = {house: i for i, house in enumerate(houses)}
    df["House Label"] = df["Hogwarts House"].map(label_dict)
    return df, label_dict


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg


def gradient_descent(X, y, alpha=0.01, lambda_=0.1, iterations=7000):
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


def mini_batch_gradient_descent(X, y, alpha=0.01, lambda_=0.1,
                                iterations=7000, batch_size=32):
    """Effectue la descente de gradient en mini-batch."""
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for it in range(iterations):
        indices = np.arange(m)
        np.random.shuffle(indices)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            h = sigmoid(X_batch @ theta)
            gradient = (X_batch.T @ (h - y_batch)) / len(y_batch)
            gradient[1:] += (lambda_ / len(y_batch)) * theta[1:]
            theta -= alpha * gradient

        if it % 100 == 0:
            cost = cost_function(theta, X, y, lambda_)
            cost_history.append(cost)
            print(f"📉 MBGD Itération {it}/{iterations} - Coût : {cost:.4f}")

    return theta, cost_history


def train_one_vs_all(X, y, num_labels, alpha=0.01, lambda_=0.1,
                     iterations=7000, batch_size=None):
    """Entraîne un modèle de régression logistique One-vs-All."""
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))

    for i in range(num_labels):
        print(f"\n🚀 Entraînement du modèle pour la maison {i}...")

        y_i = (y == i).astype(int)

        if batch_size:
            all_theta[i], _ = mini_batch_gradient_descent(
                X, y_i, alpha, lambda_, iterations, batch_size
            )
        else:
            all_theta[i], _ = gradient_descent(
                X, y_i, alpha, lambda_, iterations
            )

    return all_theta


def save_model(theta, label_dict, mean_train, std_train,
               filepath="data/logreg_model.npy"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, {
        "theta": theta,
        "labels": label_dict,
        "mean": mean_train,
        "std": std_train
    })
    print(f"✅ Modèle sauvegardé : {filepath}")


def evaluate_model(X, y, theta):
    predictions = np.argmax(sigmoid(X @ theta.T), axis=1)
    accuracy = np.mean(predictions == y) * 100
    print("🎯 Précision du modèle sur les données d'entraînement : "
          f"{accuracy:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("📌 Usage : python logreg_train.py <dataset_train.csv> "
              "[batch_size]")
        sys.exit(1)

    file_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df, selected_features, mean_train, std_train = load_data(file_path)
    df, label_dict = encode_labels(df)

    X = df[selected_features].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = df["House Label"].values

    num_labels = len(label_dict)

    theta = train_one_vs_all(
        X, y, num_labels, iterations=7000, batch_size=batch_size
    )

    save_model(theta, label_dict, mean_train, std_train)
    evaluate_model(X, y, theta)

    print("🎯 Entraînement terminé !")
