import unittest
import os
from app.histogram import load_data, plot_histograms
from colorama import Fore, init

init(autoreset=True)


class TestHistogram(unittest.TestCase):
    """🧪 Tests unitaires pour histogram.py"""

    @classmethod
    def setUpClass(cls):
        """🔍 Chargement des données pour les tests"""
        print(Fore.BLUE + "\n📥 Chargement du dataset pour les tests...")
        cls.filepath = "data/dataset_train.csv"
        cls.df = load_data(cls.filepath)
        print(Fore.GREEN + "✅ Chargement réussi !\n")

    def test_load_data(self):
        """✅ Vérifie que les données sont bien chargées
        et que 'Hogwarts House' est présente"""
        print(Fore.CYAN + "🔍 Vérification de la structure des données...")

        self.assertIn("Hogwarts House", self.df.columns, Fore.RED +
                      "❌ La colonne 'Hogwarts House' est absente !")
        self.assertGreater(len(self.df), 0, Fore.RED +
                           "❌ Le DataFrame est vide !")

        print(Fore.GREEN + "✅ Données correctement chargées !\n")

    def test_numeric_columns(self):
        """✅ Vérifie que seules les colonnes numériques sont présentes"""
        print(Fore.CYAN + "🔍 Vérification des colonnes numériques...")

        non_numeric_cols = (
            set(self.df.columns)
            - set(self.df.select_dtypes(include=['number']).columns)
            - {"Hogwarts House"}
        )
        self.assertEqual(len(non_numeric_cols), 0, Fore.RED +
                         f"❌ Colonnes non numériques trouvées : "
                         f"{non_numeric_cols}")

        print(Fore.GREEN +
              "✅ Seules les colonnes numériques sont chargées !\n")

    def test_plot_histograms(self):
        """✅ Vérifie que les fichiers .png sont bien créés"""
        print(Fore.CYAN +
              "📊 Vérification de la génération des histogrammes...")

        output_dir = "data/histograms"
        plot_histograms(self.df, output_dir)

        generated_files = os.listdir(output_dir)
        self.assertGreater(
            len(generated_files), 0,
            Fore.RED + "❌ Aucun histogramme n'a été généré !"
        )

        msg = f"✅ {len(generated_files)} histogrammes générés avec succès !\n"
        print(Fore.GREEN + msg)


if __name__ == "__main__":
    unittest.main()
