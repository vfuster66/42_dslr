import unittest
import os
import tempfile
from app.pair_plot import (
    load_data, plot_pairplot, find_best_separating_features,
    find_best_correlated_features
)
from colorama import Fore, init

init(autoreset=True)


class TestPairPlot(unittest.TestCase):
    """🧪 Tests unitaires pour pair_plot.py"""

    @classmethod
    def setUpClass(cls):
        print(Fore.BLUE + "\n📥 Chargement du dataset pour les tests...")
        cls.filepath = "data/dataset_train.csv"
        cls.df = load_data(cls.filepath)
        print(Fore.GREEN + "✅ Chargement réussi !\n")

    def test_numeric_columns(self):
        print(Fore.CYAN + "🔍 Vérification des colonnes numériques...")
        numeric_cols = self.df.select_dtypes(
            include=['number']).columns.tolist()
        self.assertGreater(len(numeric_cols), 0)
        print(Fore.GREEN +
              "✅ Colonnes numériques correctement sélectionnées !\n")

    def test_pairplot_generation(self):
        print(Fore.CYAN + "📊 Vérification de la génération du pair plot...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "pair_plot.png")
            plot_pairplot(
                self.df, output_dir=tmp_dir, filename="pair_plot.png")

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            print(Fore.GREEN +
                  f"✅ Pair plot généré avec succès : {output_path}\n")

    def test_find_best_separating_features(self):
        print(Fore.CYAN +
              "🔍 Vérification de la recherche des "
              "meilleures variables de séparation...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "best_features.txt")
            find_best_separating_features(self.df, output_file=output_file)

            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                self.assertGreater(len(lines), 0)

            print(Fore.GREEN +
                  f"✅ {len(lines)} meilleures variables sauvegardées dans : "
                  f"{output_file}\n")

    def test_find_best_correlated_features(self):
        print(Fore.CYAN +
              "🔍 Vérification de la recherche des meilleures corrélations...")

        try:
            find_best_correlated_features(self.df)
        except Exception as e:
            self.fail(
                f"❌ La fonction find_best_correlated_features a levé une "
                f"exception : {e}"
                )

        print(Fore.GREEN +
              "✅ Recherche des meilleures corrélations réussie !"
              "\n")

    def test_missing_house_column_raises_error(self):
        print(Fore.CYAN +
              "🚨 Vérification du comportement en cas de dataset incomplet...")

        with tempfile.NamedTemporaryFile(
            mode='w+', suffix='.csv', delete=False
        ) as tmpfile:
            tmpfile.write("Course1,Course2\n80,90\n85,95\n")
            tmpfile.flush()
            file_path = tmpfile.name

        try:
            with self.assertRaises(ValueError, msg=Fore.RED +
                                   "❌ Erreur non levée pour dataset invalide !"
                                   ):
                load_data(file_path)
        finally:
            # Nettoyage propre
            if os.path.exists(file_path):
                os.remove(file_path)

        print(Fore.GREEN +
              "✅ Erreur levée correctement si 'Hogwarts House' est absente.\n")


if __name__ == "__main__":
    unittest.main()
