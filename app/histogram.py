import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


def load_data(filepath):
    """Charge les données du fichier CSV et retourne un DataFrame
    filtré sur les colonnes numériques."""
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


def plot_histograms(df, output_dir="data/histograms"):
    """Génère un histogramme pour chaque cours et l'enregistre en .png."""
    os.makedirs(output_dir, exist_ok=True)

    courses = [col for col in df.columns if col != "Hogwarts House"]

    houses = df["Hogwarts House"].unique()

    for course in courses:
        plt.figure(figsize=(8, 6))

        ax = sns.histplot(
            data=df, x=course, hue="Hogwarts House", kde=True,
            bins=30, palette="Set2", legend=False
        )

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, houses, title="Maisons")

        plt.title(f"Distribution des scores pour {course}")
        plt.xlabel("Score")
        plt.ylabel("Nombre d'élèves")
        plt.grid(True)

        # Sauvegarde du fichier .png
        file_path = os.path.join(output_dir, f"{course.replace(' ', '_')}.png")
        plt.savefig(file_path)
        plt.close()  # Ferme la figure pour éviter la surcharge mémoire

        print(f"📊 Histogramme sauvegardé : {file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_data(file_path)
    plot_histograms(df)
