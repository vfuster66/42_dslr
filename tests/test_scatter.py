import unittest
import os
from app.scatter_plot import (
    load_data,
    find_most_correlated_features,
    plot_scatter
)
from colorama import Fore, init

init(autoreset=True)


class TestScatterPlot(unittest.TestCase):
    """🧪 Tests unitaires pour scatter_plot.py"""

    @classmethod
    def setUpClass(cls):
        """🔍 Chargement des données pour les tests"""
        print(Fore.BLUE + "\n📥 Chargement du dataset pour les tests...")
        cls.filepath = "data/dataset_train.csv"
        cls.df = load_data(cls.filepath)
        print(Fore.GREEN + "✅ Chargement réussi !\n")

    def test_find_most_correlated_features(self):
        """✅ Vérifie que plusieurs paires de caractéristiques sont
        bien identifiées comme étant les plus corrélées"""
        print(Fore.CYAN +
              "🔍 Recherche des caractéristiques les plus corrélées...")

        top_correlations = find_most_correlated_features(self.df, top_n=5)

        self.assertIsInstance(top_correlations, list)
        self.assertGreater(len(top_correlations), 0, Fore.RED +
                           "❌ Aucune paire de caractéristiques trouvée !")

        for correlation in top_correlations:
            self.assertEqual(len(correlation), 3, Fore.RED +
                             "❌ Les tuples doivent contenir "
                             "(feature1, feature2, corr_value) !")
            feature1, feature2, corr_value = correlation
            self.assertIsInstance(feature1, str)
            self.assertIsInstance(feature2, str)
            self.assertNotEqual(feature1, feature2, Fore.RED +
                                "❌ Les deux caractéristiques ne doivent pas "
                                "être identiques !")
            self.assertIsInstance(corr_value, float)
            print(Fore.GREEN +
                  f"✅ Caractéristiques trouvées : {feature1} ↔ {feature2} "
                  f"avec corrélation {corr_value:.3f}")

    def test_plot_scatter(self):
        """✅ Vérifie que plusieurs fichiers
        scatter_plot.png sont bien générés"""
        print(Fore.CYAN +
              "📊 Vérification de la génération des scatter plots...")

        output_dir = "data/scatter_plots"
        top_correlations = find_most_correlated_features(self.df, top_n=3)

        for i, (feature1, feature2, _) in enumerate(top_correlations, 1):
            output_path = (f"{output_dir}/scatter_{i}_{feature1}_vs_"
                           f"{feature2}.png")
            plot_scatter(self.df, feature1, feature2, output_path)
            self.assertTrue(os.path.exists(output_path), Fore.RED +
                            f"❌ Le scatter plot {output_path} "
                            "n'a pas été généré !")
            print(Fore.GREEN +
                  f"✅ Scatter plot généré avec succès : {output_path}")


if __name__ == "__main__":
    unittest.main()
