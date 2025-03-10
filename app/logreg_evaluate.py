#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt


def load_best_features(filepath="data/best_features.txt"):
    """Charge la liste des meilleures caractéristiques sélectionnées."""
    try:
        with open(filepath, "r") as f:
            features = [line.strip() for line in f.readlines()]
        return features
    except Exception as e:
        print(f"❌ Erreur chargement des meilleures caractéristiques : {e}")
        sys.exit(1)


def load_data(filepath):
    """Charge les données et sélectionne les colonnes pertinentes."""
    try:
        df = pd.read_csv(filepath)

        if "Hogwarts House" not in df.columns:
            raise ValueError(
                "Le dataset doit contenir la colonne 'Hogwarts House'."
                )

        selected_features = load_best_features()

        df = df[["Hogwarts House"] + selected_features].dropna()
        df[selected_features] = df[selected_features].apply(
            pd.to_numeric, errors='coerce'
        )

        return df, selected_features
    except Exception as e:
        print(f"❌ Erreur chargement des données : {e}")
        sys.exit(1)


def sigmoid(z):
    """Fonction sigmoïde."""
    return 1 / (1 + np.exp(-z))


def predict(X, theta):
    """Prédit la classe pour chaque observation."""
    probabilities = sigmoid(X @ theta.T)
    return np.argmax(probabilities, axis=1)


def evaluate_accuracy(y_pred, y_true):
    """Calcule et affiche la précision globale du modèle."""
    accuracy = np.mean(y_pred == y_true) * 100
    print(f"\n🎯 Précision globale : {accuracy:.2f}%")
    return accuracy


def confusion_matrix(y_true, y_pred, num_classes):
    """Calcule la matrice de confusion."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def classification_report(matrix, label_names, output_file=None):
    """Affiche et enregistre un rapport de classification."""
    lines = []
    header = f"{'Classe':<15}{'Precision':>10}{'Recall':>10}{'F1-Score':>12}"
    print("\n📋 Rapport de classification :\n")
    print(header)
    lines.append(header)

    for idx, label in enumerate(label_names):
        TP = matrix[idx, idx]
        FP = sum(matrix[:, idx]) - TP
        FN = sum(matrix[idx, :]) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0

        row = f"{label:<15}{precision:>10.2f}{recall:>10.2f}{f1:>12.2f}"
        print(row)
        lines.append(row)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        print(f"\n✅ Rapport de classification sauvegardé dans {output_file}")


def save_confusion_matrix(matrix, label_names, output_file):
    """Sauvegarde la matrice de confusion dans un CSV."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df_cm = pd.DataFrame(matrix, index=label_names, columns=label_names)
    df_cm.to_csv(output_file)

    print(f"\n✅ Matrice de confusion sauvegardée dans {output_file}")


def plot_confusion_matrix_heatmap(matrix, label_names, output_file):
    """
    Génère et sauvegarde une heatmap de la matrice de confusion.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_names,
                yticklabels=label_names,
                cbar=False)

    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")

    plt.savefig(output_file)
    plt.close()

    print(
        f"\n✅ Heatmap de la matrice de confusion sauvegardée dans "
        f"{output_file}"
        )


def main():
    if len(sys.argv) != 2:
        print("📌 Usage : python logreg_evaluate.py <dataset_train.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    # === Chargement des données ===
    df, selected_features = load_data(file_path)

    # === Chargement du modèle ===
    try:
        model_data = np.load("data/logreg_model.npy", allow_pickle=True).item()
        theta = model_data["theta"]
        labels = model_data["labels"]
        mean_train = model_data["mean"]
        std_train = model_data["std"]

    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        sys.exit(1)

    # === Préparation des données ===
    df[selected_features] = (df[selected_features] - mean_train) / std_train
    df["House Label"] = df["Hogwarts House"].map(labels)
    df = df.dropna(subset=["House Label"])

    y_true = df["House Label"].astype(int).values
    X = df[selected_features].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # === Prédictions ===
    y_pred = predict(X, theta)

    # === Évaluation ===
    accuracy = evaluate_accuracy(y_pred, y_true)

    label_names = [
        k for k, v in sorted(labels.items(), key=lambda item: item[1])
    ]
    matrix = confusion_matrix(y_true, y_pred, len(label_names))

    # === Rapport et Sauvegardes ===
    classification_report(
        matrix, label_names, "data/evaluate/classification_report.txt"
    )
    save_confusion_matrix(
        matrix, label_names, "data/evaluate/confusion_matrix.csv"
    )

    # === Nouvelle Heatmap ===
    plot_confusion_matrix_heatmap(
        matrix, label_names, "data/evaluate/confusion_matrix.png"
    )

    print(f"\n✅ Évaluation terminée avec une précision de {accuracy:.2f}%\n")


if __name__ == "__main__":
    main()
