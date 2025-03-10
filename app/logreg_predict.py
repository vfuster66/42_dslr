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
        print(f"\n🔍 Caractéristiques sélectionnées : {features}")
        return features
    except Exception as e:
        print(f"❌ Erreur lors du chargement des meilleures "
              f"caractéristiques : {e}")
        sys.exit(1)


def load_data(filepath):
    """
    Charge le jeu de test et remplit les valeurs manquantes.
    Retourne le DataFrame et la liste des caractéristiques sélectionnées.
    """
    try:
        df = pd.read_csv(filepath)

        if "Hogwarts House" not in df.columns:
            raise ValueError(
                "Le dataset doit contenir une colonne 'Hogwarts House'."
            )

        selected_features = load_best_features()

        df = df[["Hogwarts House"] + selected_features]
        df[selected_features] = df[selected_features].apply(
            pd.to_numeric, errors='coerce')
        df[selected_features] = df[selected_features].fillna(
            df[selected_features].mean())

        return df, selected_features
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        sys.exit(1)


def sigmoid(z):
    """Fonction sigmoïde."""
    return 1 / (1 + np.exp(-z))


def predict(X, theta):
    """Prédit la classe pour chaque élève."""
    probabilities = sigmoid(X @ theta.T)
    return np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py <dataset_test.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df, selected_features = load_data(file_path)

    try:
        model_data = np.load("data/logreg_model.npy", allow_pickle=True).item()
        theta = model_data["theta"]
        labels = model_data["labels"]
        mean_train = model_data["mean"]
        std_train = model_data["std"]

    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    df[selected_features] = (df[selected_features] - mean_train) / std_train

    X = df[selected_features].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y_pred = predict(X, theta)

    inv_labels = {v: k for k, v in labels.items()}
    predictions = [inv_labels[p] for p in y_pred]

    results = pd.DataFrame({
        "Index": range(len(predictions)),
        "Hogwarts House": predictions
    })

    output_path = "data/houses.csv"
    results.to_csv(output_path, index=False)
    print(f"\n✅ Prédictions sauvegardées dans {output_path}")
