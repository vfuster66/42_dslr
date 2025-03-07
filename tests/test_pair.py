import unittest
import os
from app.pair_plot import load_data, plot_pairplot
from colorama import Fore, init

init(autoreset=True)


class TestPairPlot(unittest.TestCase):
    """🧪 Tests unitaires pour pair_plot.py"""

    @classmethod
    def setUpClass(cls):
        """🔍 Chargement des données pour les tests"""
        print(Fore.BLUE + "\n📥 Chargement du dataset pour les tests...")
        cls.filepath = "data/dataset_train.csv"
        cls.df = load_data(cls.filepath)
        print(Fore.GREEN + "✅ Chargement réussi !\n")

    def test_numeric_columns(self):
        """✅ Vérifie que seules les colonnes numériques sont utilisées"""
        print(Fore.CYAN + "🔍 Vérification des colonnes numériques...")
        numeric_cols = self.df.select_dtypes(
            include=['number']).columns.tolist()
        self.assertGreater(len(numeric_cols), 0, Fore.RED +
                           "❌ Aucune colonne numérique trouvée !")
        print(Fore.GREEN +
              "✅ Colonnes numériques correctement sélectionnées !\n")

    def test_pairplot_generation(self):
        """✅ Vérifie que le fichier pair_plot.png est bien généré"""
        print(Fore.CYAN + "📊 Vérification de la génération du pair plot...")
        output_path = "data/pair_plot.png"
        plot_pairplot(self.df, output_path)
        self.assertTrue(os.path.exists(output_path), Fore.RED +
                        "❌ Le pair plot n'a pas été généré !")
        print(Fore.GREEN + f"✅ Pair plot généré avec succès : {output_path}\n")


if __name__ == "__main__":
    unittest.main()
