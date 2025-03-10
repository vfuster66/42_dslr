#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt


def load_best_features(filepath="data/best_features.txt"):
    try:
        with open(filepath, "r") as f:
            features = [line.strip() for line in f.readlines()]
        print(f"🔍 Caractéristiques sélectionnées : {features}")
        return features
    except Exception as e:
        print(f"❌ Erreur chargement best features : {e}")
        sys.exit(1)


def load_data(filepath, selected_features):
    try:
        df = pd.read_csv(filepath)

        if "Hogwarts House" not in df.columns:
            raise ValueError(
                f"Le dataset {filepath} doit contenir 'Hogwarts House'."
            )

        df = df[["Hogwarts House"] + selected_features].dropna()
        df[selected_features] = df[selected_features].apply(
            pd.to_numeric, errors='coerce')

        return df
    except Exception as e:
        print(f"❌ Erreur chargement données : {e}")
        sys.exit(1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_probabilities(X, theta):
    return sigmoid(X @ theta.T)


def compute_roc_auc(y_true_bin, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr_list, fpr_list = [], []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        TP = np.sum((y_pred == 1) & (y_true_bin == 1))
        FP = np.sum((y_pred == 1) & (y_true_bin == 0))
        FN = np.sum((y_pred == 0) & (y_true_bin == 1))
        TN = np.sum((y_pred == 0) & (y_true_bin == 0))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    tpr_list = [0.0] + tpr_list + [1.0]
    fpr_list = [0.0] + fpr_list + [1.0]

    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (
            (fpr_list[i] - fpr_list[i - 1]) *
            (tpr_list[i] + tpr_list[i - 1]) / 2
        )

    return np.array(fpr_list), np.array(tpr_list), auc, thresholds


def find_optimal_threshold(fpr, tpr, thresholds):
    distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    idx = np.argmin(distances)
    return fpr[idx], tpr[idx], thresholds[idx]


def plot_roc(fpr, tpr, auc_value, label_name, output_path, optimal_point=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_value:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    if optimal_point:
        plt.scatter(
            optimal_point[0], optimal_point[1],
            color='red', label='Seuil optimal'
        )

    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Courbe ROC - {label_name}')
    plt.legend(loc="lower right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"✅ ROC sauvegardé : {output_path}")


def plot_super_roc(roc_data, output_path, dataset_name="Train"):
    plt.figure(figsize=(10, 8))
    for label_name, fpr, tpr, auc_value in roc_data:
        plt.plot(fpr, tpr, lw=2, label=f'{label_name} (AUC = {auc_value:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Super ROC - {dataset_name}')
    plt.legend(loc="lower right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Super ROC sauvegardé : {output_path}")


def export_roc_data(roc_data, output_csv):
    """Sauvegarde des TPR/FPR/AUC dans un fichier CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows = []
    for label_name, fpr, tpr, auc_value in roc_data:
        for f, t in zip(fpr, tpr):
            rows.append([label_name, f, t, auc_value])

    df_roc = pd.DataFrame(rows, columns=["Classe", "FPR", "TPR", "AUC"])
    df_roc.to_csv(output_csv, index=False)
    print(f"✅ Données ROC sauvegardées dans {output_csv}")


def process_dataset(
    df, selected_features, mean_train, std_train, theta, labels, output_prefix
):
    df[selected_features] = (df[selected_features] - mean_train) / std_train
    df["House Label"] = df["Hogwarts House"].map(labels)
    df = df.dropna(subset=["House Label"])

    y_true = df["House Label"].astype(int).values
    X = df[selected_features].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    y_probs = compute_probabilities(X, theta)

    label_names = [
        k for k, v in sorted(labels.items(), key=lambda item: item[1])
    ]
    roc_data = []

    for idx, label in enumerate(label_names):
        y_true_bin = (y_true == idx).astype(int)
        y_prob = y_probs[:, idx]

        fpr, tpr, auc_value, thresholds = compute_roc_auc(y_true_bin, y_prob)

        opt_fpr, opt_tpr, opt_threshold = find_optimal_threshold(
            fpr, tpr, thresholds
        )

        print(f"➡️  {label}: Seuil optimal = {opt_threshold:.4f}, "
              f"AUC = {auc_value:.4f}")

        plot_roc(
            fpr, tpr, auc_value, label,
            f"{output_prefix}_roc_{label}.png",
            optimal_point=(opt_fpr, opt_tpr)
        )

        roc_data.append((label, fpr, tpr, auc_value))

    plot_super_roc(
        roc_data,
        f"{output_prefix}_super.png",
        dataset_name=os.path.basename(output_prefix)
    )
    export_roc_data(roc_data, f"{output_prefix}_roc_data.csv")


def merge_test_with_predictions(test_file, predictions_file):
    try:
        df_test = pd.read_csv(test_file)
        df_pred = pd.read_csv(predictions_file)

        df_merged = df_test.copy()
        df_merged["Hogwarts House"] = df_pred["Hogwarts House"]

        print("✅ Fusion test + prédictions OK")
        return df_merged
    except Exception as e:
        print(f"❌ Erreur fusion test + prédictions : {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("📌 Usage : python logreg_roc.py <dataset_train.csv> "
              "[dataset_test.csv]")
        sys.exit(1)

    train_dataset_path = sys.argv[1]
    test_dataset_path = sys.argv[2] if len(sys.argv) == 3 else None

    try:
        model_data = np.load("data/logreg_model.npy", allow_pickle=True).item()
        theta = model_data["theta"]
        labels = model_data["labels"]
        mean_train = model_data["mean"]
        std_train = model_data["std"]
        print(f"\n📜 Labels du modèle : {labels}")
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        sys.exit(1)

    selected_features = load_best_features()
    df_train = load_data(train_dataset_path, selected_features)

    print("⚙️  Génération des courbes ROC sur le dataset d'entraînement...")
    process_dataset(
        df_train,
        selected_features,
        mean_train,
        std_train,
        theta,
        labels,
        output_prefix="data/evaluate/train"
    )

    if test_dataset_path:
        df_test = merge_test_with_predictions(
            test_dataset_path, "data/houses.csv")
        df_test = df_test[["Hogwarts House"] + selected_features].dropna()

        print("\n⚙️  Génération des courbes ROC sur le dataset de test...")
        process_dataset(
            df_test,
            selected_features,
            mean_train,
            std_train,
            theta,
            labels,
            output_prefix="data/evaluate/test"
        )
