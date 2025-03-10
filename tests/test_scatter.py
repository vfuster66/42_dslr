import unittest
import os
import tempfile
from app.scatter_plot import (
    load_data, find_most_correlated_features, plot_scatter
)
from colorama import Fore, init

init(autoreset=True)


class TestScatterPlot(unittest.TestCase):
    """🧪 Tests unitaires pour scatter_plot.py"""

    @classmethod
    def setUpClass(cls):
        print(Fore.BLUE + "\n📥 Chargement du dataset pour les tests...")
        cls.filepath = "data/dataset_train.csv"
        cls.df = load_data(cls.filepath)
        print(Fore.GREEN + "✅ Chargement réussi !\n")

    def test_find_most_correlated_features(self):
        print(Fore.CYAN +
              "🔍 Recherche des caractéristiques les plus corrélées...")

        top_correlations = find_most_correlated_features(self.df, top_n=5)

        self.assertIsInstance(top_correlations, list)
        self.assertGreater(len(top_correlations), 0)

        for correlation in top_correlations:
            self.assertEqual(len(correlation), 3)
            feature1, feature2, corr_value = correlation
            self.assertIsInstance(feature1, str)
            self.assertIsInstance(feature2, str)
            self.assertNotEqual(feature1, feature2)
            self.assertIsInstance(corr_value, float)

        print(Fore.GREEN + "✅ Corrélations détectées correctement !\n")

    def test_plot_scatter_files(self):
        print(Fore.CYAN +
              "📊 Vérification de la génération des scatter plots...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            top_correlations = find_most_correlated_features(self.df, top_n=3)

            for i, (feature1, feature2, _) in enumerate(top_correlations, 1):
                output_path = os.path.join(
                    tmp_dir, f"scatter_{i}_{feature1}_vs_{feature2}.png"
                )
                plot_scatter(self.df, feature1, feature2, output_path)

                self.assertTrue(os.path.exists(output_path))
                self.assertGreater(os.path.getsize(output_path), 0)

        print(Fore.GREEN + "✅ Scatter plots générés et vérifiés !\n")

    def test_missing_house_column_raises_error(self):
        print(Fore.CYAN +
              "🚨 Vérification du comportement en cas de dataset incomplet...")

        with tempfile.NamedTemporaryFile(
                mode='w+', suffix='.csv', delete=False) as tmpfile:
            tmpfile.write("Course1,Course2\n80,90\n85,95\n")
            tmpfile.flush()
            file_path = tmpfile.name

        # Test de l'exception attendue
        try:
            with self.assertRaises(ValueError, msg=Fore.RED +
                                   "❌ Erreur non levée pour dataset invalide !"
                                   ):
                load_data(file_path)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        print(Fore.GREEN +
              "✅ Erreur levée correctement si 'Hogwarts House' est absente.\n")


if __name__ == "__main__":
    unittest.main()
