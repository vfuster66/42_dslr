import unittest
import os
import tempfile
from app.histogram import load_data, plot_histograms
from colorama import Fore, init

init(autoreset=True)


class TestHistogram(unittest.TestCase):
    """🧪 Tests unitaires pour histogram.py"""

    @classmethod
    def setUpClass(cls):
        print(Fore.BLUE + "\n📥 Chargement du dataset pour les tests...")
        cls.filepath = "data/dataset_train.csv"
        cls.df = load_data(cls.filepath)
        print(Fore.GREEN + "✅ Chargement réussi !\n")

    def test_load_data(self):
        print(Fore.CYAN + "🔍 Vérification de la structure des données...")
        self.assertIn("Hogwarts House", self.df.columns)
        self.assertGreater(len(self.df), 0)
        print(Fore.GREEN + "✅ Données correctement chargées !\n")

    def test_numeric_columns(self):
        print(Fore.CYAN + "🔍 Vérification des colonnes numériques...")
        non_numeric_cols = (
            set(self.df.columns)
            - set(self.df.select_dtypes(include=['number']).columns)
            - {"Hogwarts House"}
        )
        self.assertEqual(len(non_numeric_cols), 0)
        print(Fore.GREEN +
              "✅ Seules les colonnes numériques sont chargées !\n")

    def test_plot_histograms(self):
        print(Fore.CYAN +
              "📊 Vérification de la génération des histogrammes...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_histograms(self.df, tmp_dir)
            courses = [
                col for col in self.df.columns if col != "Hogwarts House"
            ]
            generated_files = os.listdir(tmp_dir)

            self.assertEqual(len(generated_files), len(courses))

            for course in courses:
                file_name = f"{course.replace(' ', '_')}.png"
                file_path = os.path.join(tmp_dir, file_name)
                self.assertTrue(os.path.exists(file_path))
                self.assertGreater(os.path.getsize(file_path), 0)

            print(Fore.GREEN +
                  f"✅ {len(generated_files)} histogrammes générés "
                  "avec succès !\n")

    def test_missing_house_column_raises_error(self):
        print(Fore.CYAN +
              "🚨 Vérification du comportement en cas de dataset incomplet...")

        with tempfile.NamedTemporaryFile(
                mode='w+', suffix='.csv', delete=False) as tmpfile:
            tmpfile.write("Course1,Course2\n80,90\n85,95\n")
            tmpfile.flush()
            file_path = tmpfile.name

        try:
            with self.assertRaises(
                ValueError,
                msg=Fore.RED + "❌ Erreur non levée pour dataset invalide !"
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
