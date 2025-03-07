import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import numpy as np


def load_data(filepath):
    """Charge les données et filtre les colonnes numériques."""
    try:
        df = pd.read_csv(filepath)

        if "Hogwarts House" not in df.columns:
            raise ValueError(
                "Le dataset doit contenir une colonne 'Hogwarts House'.")

        numeric_cols = [
            col for col in df.columns
            if col not in ["Index", "Hogwarts House", "First Name",
                           "Last Name", "Birthday", "Best Hand"]
        ]

        df = df[["Hogwarts House"] + numeric_cols].dropna()
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        sys.exit(1)


def find_best_separating_features(df):
    """Identifie les variables ayant une bonne séparation entre les maisons."""
    best_features = []
    for col in df.columns[1:]:
        grouped = df.groupby("Hogwarts House")[col].mean()
        std = df[col].std()
        diff_max_min = grouped.max() - grouped.min()

        if diff_max_min > 0.8 * std:
            best_features.append((col, diff_max_min / std))

    best_features.sort(key=lambda x: x[1], reverse=True)

    print("\n📊 **Variables avec la meilleure séparation entre les maisons :**")
    for feature, score in best_features[:5]:  # Afficher les 5 meilleures
        print(f"✅ {feature} (Ratio de séparation : {score:.3f})")


def find_best_correlated_features(df):
    """Trouve les meilleures paires de variables corrélées."""
    correlation_matrix = df.iloc[:, 1:].corr().abs()
    np.fill_diagonal(correlation_matrix.values, 0)

    correlations = correlation_matrix.unstack().reset_index()
    correlations.columns = ["Feature1", "Feature2", "Correlation"]

    top_correlations = (
        correlations.sort_values(by="Correlation", ascending=False)
        .drop_duplicates(subset=["Correlation"])
        .head(5)
    )

    print("\n📊 **Meilleures relations entre deux variables :**")
    for _, row in top_correlations.iterrows():
        print(f"✅ {row['Feature1']} ↔ {row['Feature2']} "
              f"(Corrélation : {row['Correlation']:.3f})")


def plot_pairplot(df, output_path="data/pair_plot.png"):
    """Génère un pair plot des caractéristiques
    numériques et sauvegarde l'image."""
    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(df, hue="Hogwarts House", palette="Set2")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pairplot.savefig(output_path)
    plt.close()
    print(f"📊 Pair plot sauvegardé : {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_data(file_path)

    find_best_separating_features(df)

    find_best_correlated_features(df)

    plot_pairplot(df)
