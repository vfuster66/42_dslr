import sys
import csv
import numpy as np
import json
import os


def load_data(filepath):
    """Charge le dataset CSV et extrait
    uniquement les colonnes numériques utiles."""
    try:
        data = {}
        header = []

        with open(filepath, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)

            for col in header:
                data[col] = []

            for row in csv_reader:
                for i, value in enumerate(row):
                    if i < len(header):
                        try:
                            float_val = float(value) if value.strip() else None
                            data[header[i]].append(float_val)
                        except ValueError:
                            data[header[i]].append(None)

        numeric_data = {}
        for col in header:
            if col.lower() in [
                "index", "hogwarts house", "first name",
                "last name", "birthday", "best hand"
            ]:
                continue  # Exclure explicitement les colonnes non numériques

            valid_values = [
                val for val in data[col]
                if isinstance(val, (int, float)) and val is not None
            ]

            if len(valid_values) >= 0.5 * len(data[col]):
                numeric_data[col] = valid_values

        return numeric_data

    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier : {e}")
        sys.exit(1)


def compute_statistics(data):
    """Calcule les statistiques descriptives pour chaque colonne numérique."""
    stats = {}
    for column, values in data.items():
        values = sorted(values)
        n = len(values)
        if n == 0:
            continue

        count = n
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5

        def quartile(sorted_values, q):
            pos = (len(sorted_values) - 1) * q
            lower = int(np.floor(pos))
            upper = int(np.ceil(pos))
            if lower == upper:
                return sorted_values[lower]
            return (sorted_values[lower] +
                    (sorted_values[upper] - sorted_values[lower]) *
                    (pos - lower))

        q25 = quartile(values, 0.25)
        median = quartile(values, 0.5)
        q75 = quartile(values, 0.75)

        skewness = (sum((x - mean) ** 3 for x in values) / n) / (std ** 3) \
            if std != 0 else 0
        kurtosis = (sum((x - mean) ** 4 for x in values) / n) / (std ** 4) \
            if std != 0 else 0

        stats[column] = {
            "count": count,
            "mean": mean,
            "std": std,
            "min": values[0],
            "25%": q25,
            "50%": median,
            "75%": q75,
            "max": values[-1],
            "skewness": skewness,
            "kurtosis": kurtosis
        }

    return stats


def print_statistics(stats):
    """Affiche les statistiques dans un format tabulaire bien aligné."""
    max_feature_len = max(len(feature) for feature in stats.keys())
    feature_width = max(max_feature_len, 15)
    column_width = 15

    print(f"{'':^{feature_width}}", end="")
    features = list(stats.keys())

    for feature in features:
        print(f"{feature:^{column_width}}", end="")
    print()

    metrics = [
        "count", "mean", "std", "min", "25%",
        "50%", "75%", "max", "skewness", "kurtosis"
    ]

    for metric in metrics:
        print(f"{metric:^{feature_width}}", end="")
        for feature in features:
            value = stats[feature].get(metric, None)
            if value is None:
                print("N/A".rjust(column_width), end="")
            else:
                print(f"{value:.6f}".rjust(column_width), end="")
        print()


def export_json(stats, output_dir):
    """Exporte les statistiques en JSON."""
    json_path = os.path.join(output_dir, 'describe_stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\n✅ Statistiques JSON exportées dans {json_path}")


def export_csv(stats, output_dir):
    """Exporte les statistiques en CSV."""
    csv_path = os.path.join(output_dir, 'describe_stats.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Feature'] + list(next(iter(stats.values())).keys())
        writer.writerow(header)

        for feature, values in stats.items():
            row = [feature] + [values[k] for k in header[1:]]
            writer.writerow(row)

    print(f"✅ Statistiques CSV exportées dans {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("📌 Usage : python describe.py <dataset_train.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_data(file_path)
    stats = compute_statistics(data)

    print_statistics(stats)

    output_dir = 'data/describe'
    os.makedirs(output_dir, exist_ok=True)

    export_json(stats, output_dir)
    export_csv(stats, output_dir)
