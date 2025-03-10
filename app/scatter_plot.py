import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


def load_data(filepath):
    """Charge les données du fichier CSV et retourne un DataFrame filtré
    sur les colonnes numériques."""
    try:
        df = pd.read_csv(filepath)

        if "Hogwarts House" not in df.columns:
            raise ValueError(
                "Le dataset doit contenir une colonne 'Hogwarts House'."
                )

        numeric_cols = [
            col for col in df.columns
            if col not in [
                "Index", "Hogwarts House", "First Name",
                "Last Name", "Birthday", "Best Hand"
            ]
        ]

        df = df[["Hogwarts House"] + numeric_cols].dropna()

        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        sys.exit(1)


def find_most_correlated_features(df, top_n=5):
    """Trouve les `top_n` paires de caractéristiques les plus corrélées."""

    numeric_df = df.select_dtypes(include=[np.number])

    correlation_matrix = numeric_df.corr().abs()
    np.fill_diagonal(correlation_matrix.values, 0)

    correlations = (
        correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        .unstack()
        .reset_index()
        .dropna()
    )
    correlations.columns = ["Feature1", "Feature2", "Correlation"]

    top_correlations = correlations.sort_values(
        by="Correlation", ascending=False
    ).head(top_n)

    top_correlations_list = list(
        top_correlations.itertuples(index=False, name=None)
    )

    print("\n📊 **Top {} des corrélations les plus élevées :**".format(top_n))
    for feature1, feature2, corr_value in top_correlations_list:
        print(f"✅ {feature1} ↔ {feature2} : {corr_value:.3f}")

    return top_correlations_list


def plot_scatter(df, feature1, feature2, output_path):
    """Génère un scatter plot des caractéristiques les plus corrélées."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x=feature1, y=feature2, hue="Hogwarts House", palette="Set2"
    )

    plt.title(f"Relation entre {feature1} et {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(title="Maisons")
    plt.grid(True)

    # Sauvegarde du fichier .png
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"\n📊 Scatter plot sauvegardé : {output_path}")


def plot_top_correlations(df, output_dir="data/scatter_plots", top_n=3):
    """Génère plusieurs scatter plots pour les paires les plus corrélées."""
    os.makedirs(output_dir, exist_ok=True)

    correlated_pairs = find_most_correlated_features(df, top_n)

    for i, (feat1, feat2, _) in enumerate(correlated_pairs):
        output_path = f"{output_dir}/scatter_{i+1}_{feat1}_vs_{feat2}.png"
        plot_scatter(df, feat1, feat2, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_data(file_path)

    # Affichage des 5 meilleures corrélations
    find_most_correlated_features(df, top_n=5)

    # Génération des 3 meilleurs scatter plots
    plot_top_correlations(df, top_n=3)
