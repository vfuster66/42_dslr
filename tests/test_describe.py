import unittest
import pandas as pd
import numpy as np
from app.describe import load_data, compute_statistics, export_json, export_csv
from colorama import Fore, init
import json
import csv as csv_module
import tempfile
import os

init(autoreset=True)


class TestDescribe(unittest.TestCase):
    """🧪 Tests unitaires pour vérifier le fonctionnement de describe.py"""

    @classmethod
    def setUpClass(cls):
        """🔍 Charge le dataset et prépare les données pour les tests"""
        print(Fore.BLUE + "\n🔍 Chargement des données pour les tests...")
        cls.filepath = "data/dataset_train.csv"
        cls.data = load_data(cls.filepath)
        cls.stats = compute_statistics(cls.data)

        cls.df = pd.read_csv(cls.filepath)
        cls.df_numeric = cls.df.select_dtypes(include=[np.number])
        cls.df_numeric = cls.df_numeric.drop(columns=["Index"])

    def test_columns_loaded_correctly(self):
        """✅ Vérifie que seules les colonnes numériques sont chargées"""
        expected_columns = set(self.df_numeric.columns)
        loaded_columns = set(self.data.keys())

        self.assertEqual(loaded_columns, expected_columns, Fore.RED +
                         "❌ Les colonnes chargées ne correspondent pas.")

    def test_mean_calculation(self):
        """✅ Vérifie que la moyenne est correcte pour chaque colonne"""
        print(Fore.CYAN + "📊 Vérification des moyennes...")

        for column in self.data.keys():
            expected_mean = self.df_numeric[column].mean()
            computed_mean = self.stats[column]['mean']

            self.assertAlmostEqual(
                computed_mean, expected_mean, places=5,
                msg=Fore.RED +
                f"❌ Moyenne incorrecte pour {column}"
            )

        print(Fore.GREEN + "✅ Moyennes validées !")

    def test_std_calculation(self):
        """✅ Vérifie que l'écart-type est correct pour chaque colonne"""
        print(Fore.CYAN + "📊 Vérification des écarts-types...")

        for column in self.data.keys():
            expected_std = self.df_numeric[column].std(ddof=0)
            computed_std = self.stats[column]['std']

            self.assertAlmostEqual(
                computed_std, expected_std, places=5,
                msg=Fore.RED +
                f"❌ Écart-type incorrect pour {column}")

        print(Fore.GREEN + "✅ Écarts-types validés !")

    def test_quartile_calculation(self):
        """✅ Vérifie que les quartiles sont corrects"""
        print(Fore.CYAN + "📊 Vérification des quartiles...")

        for column in self.data.keys():
            expected_q25 = self.df_numeric[column].quantile(0.25)
            expected_q50 = self.df_numeric[column].median()
            expected_q75 = self.df_numeric[column].quantile(0.75)

            computed_q25 = self.stats[column]['25%']
            computed_q50 = self.stats[column]['50%']
            computed_q75 = self.stats[column]['75%']

            self.assertAlmostEqual(
                computed_q25, expected_q25, places=5,
                msg=Fore.RED +
                f"❌ 25% incorrect pour {column}")
            self.assertAlmostEqual(
                computed_q50, expected_q50, places=5,
                msg=Fore.RED +
                f"❌ 50% incorrect pour {column}")
            self.assertAlmostEqual(
                computed_q75, expected_q75, places=5,
                msg=Fore.RED +
                f"❌ 75% incorrect pour {column}")

        print(Fore.GREEN + "✅ Quartiles validés !")

    def test_min_max(self):
        """✅ Vérifie que le min et le max sont corrects"""
        print(Fore.CYAN + "📊 Vérification des min et max...")

        for column in self.data.keys():
            expected_min = self.df_numeric[column].min()
            expected_max = self.df_numeric[column].max()

            computed_min = self.stats[column]['min']
            computed_max = self.stats[column]['max']

            self.assertAlmostEqual(
                computed_min, expected_min, places=5,
                msg=Fore.RED +
                f"❌ Min incorrect pour {column}")
            self.assertAlmostEqual(
                computed_max, expected_max, places=5,
                msg=Fore.RED +
                f"❌ Max incorrect pour {column}")

        print(Fore.GREEN + "✅ Min et Max validés !")

    def test_skewness_kurtosis(self):
        """✅ Vérifie que skewness et kurtosis sont corrects"""
        print(Fore.CYAN + "📊 Vérification de skewness et kurtosis...")

        for column in self.data.keys():
            std = self.df_numeric[column].std(ddof=0)

            if std == 0:
                expected_skewness = 0
                expected_kurtosis = 0
            else:
                expected_skewness = self.df_numeric[column].skew()
                expected_kurtosis = self.df_numeric[column].kurt() + 3

            computed_skewness = self.stats[column]['skewness']
            computed_kurtosis = self.stats[column]['kurtosis']

            print(
                f"🧐 {column} | std: {std:.6f} | "
                f"expected_skew: {expected_skewness:.6f} / "
                f"computed_skew: {computed_skewness:.6f}\n"
                f"    expected_kurt: {expected_kurtosis:.6f} / "
                f"computed_kurt: {computed_kurtosis:.6f}"
            )

            self.assertAlmostEqual(
                computed_skewness, expected_skewness, delta=0.01,
                msg=Fore.RED +
                f"❌ Skewness incorrecte pour {column} : attendu "
                f"{expected_skewness}, obtenu {computed_skewness}"
            )

            self.assertAlmostEqual(
                computed_kurtosis, expected_kurtosis, delta=0.01,
                msg=Fore.RED +
                f"❌ Kurtosis incorrecte pour {column} : attendu "
                f"{expected_kurtosis}, obtenu {computed_kurtosis}"
            )

        print(Fore.GREEN + "✅ Skewness et Kurtosis validés !")

    def test_empty_column_handling(self):
        """✅ Vérifie que les colonnes vides sont ignorées sans erreur"""
        print(Fore.CYAN + "📊 Vérification de la gestion des colonnes vides...")

        dummy_data = {
            "Valid Column": [1, 2, 3, 4, 5],
            "Empty Column": []
        }

        stats = compute_statistics(dummy_data)

        self.assertNotIn("Empty Column", stats, Fore.RED +
                         "❌ La colonne vide a été traitée alors qu'elle devait"
                         " être ignorée.")
        self.assertIn("Valid Column", stats, Fore.RED +
                      "❌ La colonne valide est absente des statistiques.")

        print(Fore.GREEN + "✅ Colonnes vides gérées correctement !")

    def test_export_json_csv(self):
        """✅ Vérifie l'export des statistiques en JSON et CSV"""
        print(Fore.CYAN + "📊 Vérification de l'export JSON et CSV...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_json(self.stats, tmp_dir)
            export_csv(self.stats, tmp_dir)

            json_path = os.path.join(tmp_dir, 'describe_stats.json')
            csv_path = os.path.join(tmp_dir, 'describe_stats.csv')

            self.assertTrue(os.path.exists(json_path), Fore.RED +
                            "❌ JSON non créé !")
            self.assertTrue(os.path.exists(csv_path), Fore.RED +
                            "❌ CSV non créé !")

            with open(json_path, 'r') as jf:
                json_data = json.load(jf)
            self.assertEqual(json_data, self.stats, Fore.RED +
                             "❌ Contenu JSON incorrect !")

            with open(csv_path, 'r', newline='') as cf:
                reader = csv_module.reader(cf)
                headers = next(reader)

                expected_headers = ['Feature'] + list(
                    next(iter(self.stats.values())).keys()
                )
                self.assertEqual(headers, expected_headers, Fore.RED +
                                 "❌ Les en-têtes CSV sont incorrects !")

                for feature, values in self.stats.items():
                    expected_row = [feature] + [
                        values[k] for k in expected_headers[1:]
                    ]
                    actual_row = next(reader)

                    self.assertEqual(actual_row[0], expected_row[0])

                    for i in range(1, len(actual_row)):
                        self.assertAlmostEqual(
                            float(actual_row[i]), expected_row[i], places=5)

                    break

            print(Fore.GREEN + "✅ Export JSON et CSV validés !")


if __name__ == "__main__":
    unittest.main()
