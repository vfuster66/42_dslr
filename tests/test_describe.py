import unittest
import pandas as pd
import numpy as np
from app.describe import load_data, compute_statistics
from colorama import Fore, init

init(autoreset=True)


class TestDescribe(unittest.TestCase):
    """Test unitaires pour vérifier le fonctionnement de describe.py"""

    @classmethod
    def setUpClass(cls):
        """Charge le dataset et prépare les données pour les tests"""
        print(Fore.BLUE + "\n🔍 Chargement des données pour les tests...")
        cls.filepath = "data/dataset_train.csv"
        cls.data = load_data(cls.filepath)
        cls.stats = compute_statistics(cls.data)

        cls.df = pd.read_csv(cls.filepath)

    def test_columns_loaded_correctly(self):
        """✅ Vérifie que seules les colonnes numériques sont chargées"""
        expected_columns = set(
            self.df.select_dtypes(include=[np.number]).columns
        ) - {"Index"}
        loaded_columns = set(self.data.keys())

        self.assertEqual(
            loaded_columns, expected_columns,
            Fore.RED + "❌ Les colonnes chargées ne correspondent pas."
        )

    def test_mean_calculation(self):
        """✅ Vérifie que la moyenne est correcte pour chaque colonne"""
        print(Fore.CYAN + "📊 Vérification des moyennes...")

        for column in self.data.keys():
            expected_mean = self.df[column].mean()
            computed_mean = self.stats[column]['mean']

            self.assertAlmostEqual(
                computed_mean, expected_mean, places=5,
                msg=Fore.RED + f"❌ Moyenne incorrecte pour {column}"
            )
        print(Fore.GREEN + "✅ Moyennes validées !")

    def test_std_calculation(self):
        """✅ Vérifie que l'écart-type est correct pour chaque colonne"""
        print(Fore.CYAN + "📊 Vérification des écarts-types...")

        for column in self.data.keys():
            expected_std = self.df[column].std(ddof=0)
            computed_std = self.stats[column]['std']

            self.assertAlmostEqual(
                computed_std, expected_std, places=5,
                msg=Fore.RED + f"❌ Écart-type incorrect pour {column}"
            )
        print(Fore.GREEN + "✅ Écarts-types validés !")

    def test_quartile_calculation(self):
        """✅ Vérifie que les quartiles sont corrects"""
        print(Fore.CYAN + "📊 Vérification des quartiles...")

        for column in self.data.keys():
            expected_q25 = self.df[column].quantile(0.25)
            expected_q50 = self.df[column].median()
            expected_q75 = self.df[column].quantile(0.75)

            computed_q25 = self.stats[column]['25%']
            computed_q50 = self.stats[column]['50% (median)']
            computed_q75 = self.stats[column]['75%']

            self.assertAlmostEqual(
                computed_q25, expected_q25, places=5,
                msg=Fore.RED + f"❌ 25% incorrect pour {column}"
            )
            self.assertAlmostEqual(
                computed_q50, expected_q50, places=5,
                msg=Fore.RED + f"❌ 50% (median) incorrect pour {column}"
            )
            self.assertAlmostEqual(
                computed_q75, expected_q75, places=5,
                msg=Fore.RED + f"❌ 75% incorrect pour {column}"
            )
        print(Fore.GREEN + "✅ Quartiles validés !")

    def test_min_max(self):
        """✅ Vérifie que le min et le max sont corrects"""
        print(Fore.CYAN + "📊 Vérification des min et max...")

        for column in self.data.keys():
            expected_min = self.df[column].min()
            expected_max = self.df[column].max()

            computed_min = self.stats[column]['min']
            computed_max = self.stats[column]['max']

            self.assertAlmostEqual(
                computed_min, expected_min, places=5,
                msg=Fore.RED + f"❌ Min incorrect pour {column}"
            )
            self.assertAlmostEqual(
                computed_max, expected_max, places=5,
                msg=Fore.RED + f"❌ Max incorrect pour {column}"
            )
        print(Fore.GREEN + "✅ Min et Max validés !")


if __name__ == "__main__":
    unittest.main()
