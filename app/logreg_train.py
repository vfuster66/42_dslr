#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
import os


def load_best_features(filepath="data/best_features.txt"):
    try:
        with open(filepath, "r") as f:
            features = [line.strip() for line in f.readlines()]
        return features
    except Exception as e:
        print(f"❌ Erreur lors du chargement des meilleures "
              f"caractéristiques : {e}")
        sys.exit(1)


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)

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


def stochastic_gradient_descent(X, y, alpha=0.05, lambda_=0.1,
                                iterations=3000):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for it in range(iterations):
        indices = np.arange(m)
        np.random.shuffle(indices)

        for i in indices:
            xi = X[i, :].reshape(1, -1)
            yi = y[i]

            h = sigmoid(xi @ theta)
            gradient = (xi.T * (h - yi)).flatten()
            gradient[1:] += (lambda_ / m) * theta[1:]

            theta -= alpha * gradient

        if it % 100 == 0:
            cost = cost_function(theta, X, y, lambda_)
            cost_history.append(cost)
            print(f"📉 SGD Itération {it}/{iterations} - Coût : {cost:.4f}")

    return theta, cost_history


def train_one_vs_all(X, y, num_labels, method="batch", alpha=0.01,
                     lambda_=0.1, iterations=7000, batch_size=32):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))

    print(f"\n🚀 Entraînement avec la méthode : {method.upper()}")
    for i in range(num_labels):
        print(f"\n🏠 Entraînement du modèle pour la maison {i}...")

        y_i = (y == i).astype(int)

        if method == "batch":
            all_theta[i], _ = gradient_descent(
                X, y_i, alpha, lambda_, iterations
            )
        elif method == "mini-batch":
            all_theta[i], _ = mini_batch_gradient_descent(
                X, y_i, alpha, lambda_, iterations, batch_size
            )
        elif method == "sgd":
            all_theta[i], _ = stochastic_gradient_descent(
                X, y_i, alpha, lambda_, iterations
            )
        else:
            raise ValueError("Méthode d'entraînement non reconnue.")

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
    print(f"\n🎯 Précision du modèle sur les données d'entraînement : "
          f"{accuracy:.2f}%")


def get_user_choice():

    methods = ["batch", "mini-batch", "sgd"]
    print("Méthodes disponibles :")
    for i, method in enumerate(methods):
        print(f"{i+1}. {method}")

    method_choice = input(
        "\n➡️  Entrez le numéro de la méthode choisie (1-3) : "
    )
    method = methods[int(method_choice)-1]

    # Paramètres prédéfinis
    if method == "batch":
        lr = 0.01
        epochs = 6000
        batch_size = None
    elif method == "mini-batch":
        lr = 0.01
        epochs = 1500
        batch_size = 32
    elif method == "sgd":
        lr = 0.05
        epochs = 3000
        batch_size = None

    print(f"\n✅ Méthode : {method.upper()}")
    print(f"✅ Learning rate : {lr}")
    print(f"✅ Epochs : {epochs}")
    if batch_size:
        print(f"✅ Batch size : {batch_size}")

    return method, lr, epochs, batch_size


if __name__ == "__main__":
    # Dataset par défaut
    dataset_path = "data/dataset_train.csv"

    # Chargement des données
    df, selected_features, mean_train, std_train = load_data(dataset_path)
    df, label_dict = encode_labels(df)

    X = df[selected_features].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = df["House Label"].values
    num_labels = len(label_dict)

    # Choix de l'utilisateur
    method, lr, epochs, batch_size = get_user_choice()

    # Entraînement du modèle
    theta = train_one_vs_all(
        X, y, num_labels,
        method=method,
        alpha=lr,
        iterations=epochs,
        batch_size=batch_size
    )

    # Sauvegarde et évaluation
    save_model(theta, label_dict, mean_train, std_train)
    evaluate_model(X, y, theta)

    print("\n🎯 Entraînement terminé !")
