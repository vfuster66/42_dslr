#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys


def load_best_features(filepath="data/best_features.txt"):
    """Charge dynamiquement les meilleures
    caractéristiques identifiées précédemment."""
    try:
        with open(filepath, "r") as f:
            features = [line.strip() for line in f.readlines()]
        print(f"🔍 Caractéristiques sélectionnées : {features}")
        return features
    except Exception as e:
        print("❌ Erreur lors du chargement des meilleures "
              f"caractéristiques : {e}")
        sys.exit(1)


def load_data(filepath):
    """
    Charge les données en sélectionnant la colonne 'Hogwarts House'
    et les caractéristiques choisies,
    puis supprime les lignes contenant des NaN.
    """
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
        return df, selected_features
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        sys.exit(1)


def sigmoid(z):
    """Fonction sigmoïde."""
    return 1 / (1 + np.exp(-z))


def predict(X, theta):
    """Prédit la classe pour chaque instance."""
    probabilities = sigmoid(X @ theta.T)
    return np.argmax(probabilities, axis=1)


def evaluate_model(y_pred, y_true):
    """Calcule et affiche la précision du modèle."""
    accuracy = np.mean(y_pred == y_true) * 100
    print("🎯 Précision du modèle sur les données d'entraînement : "
          f"{accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_evaluate.py <dataset_train.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df, selected_features = load_data(file_path)

    try:
        model_data = np.load("data/logreg_model.npy", allow_pickle=True).item()
        theta = model_data["theta"]
        labels = model_data["labels"]
        mean_train = model_data["mean"]
        std_train = model_data["std"]
        print(f"📜 Labels du modèle : {labels}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    df[selected_features] = (df[selected_features] - mean_train) / std_train

    df["House Label"] = df["Hogwarts House"].map(labels)

    df = df.dropna(subset=["House Label"])
    y_true = df["House Label"].astype(int).values
    X = df[selected_features].values

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    y_pred = predict(X, theta)
    evaluate_model(y_pred, y_true)
