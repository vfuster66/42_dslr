import unittest
import numpy as np
import os
import tempfile
from app.logreg_train import (
    load_data, encode_labels, train_one_vs_all, save_model
)
from colorama import Fore, init

init(autoreset=True)


class TestLogRegTrain(unittest.TestCase):
    """🧪 Tests unitaires pour logreg_train.py"""

    @classmethod
    def setUpClass(cls):
        print(Fore.BLUE + "\n📥 Préparation des fichiers pour les tests...")

        cls.best_features_file = "data/best_features.txt"
        os.makedirs("data", exist_ok=True)
        with open(cls.best_features_file, "w") as f:
            f.write(
                "Flying\nCharms\nDivination\nTransfiguration\nHistory of Magic"
                "\n")

        cls.filepath = "data/dataset_train.csv"
        cls.df, cls.selected_features, _, _ = load_data(cls.filepath)
        cls.df, cls.label_dict = encode_labels(cls.df)

        cls.X = cls.df[cls.selected_features].values
        cls.X = np.hstack([np.ones((cls.X.shape[0], 1)), cls.X])
        cls.y = cls.df["House Label"].values
        cls.num_labels = len(cls.label_dict)

        print(Fore.GREEN + "✅ Fichiers et données prêts !\n")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.best_features_file):
            os.remove(cls.best_features_file)
            print(Fore.GREEN +
                  "✅ Fichier best_features.txt supprimé après les tests.\n")

    def test_train_one_vs_all(self):
        print(Fore.CYAN + "🔍 Test d'entraînement (One vs All)...")
        theta = train_one_vs_all(
            self.X, self.y, self.num_labels, iterations=500)

        self.assertFalse(np.all(theta == 0),
                         Fore.RED +
                         "❌ Les poids sont tous à zéro, "
                         "l'entraînement n'a pas fonctionné.")
        print(Fore.GREEN + "✅ Entraînement réussi avec des poids appris !\n")

    def test_save_model(self):
        print(Fore.CYAN + "🔍 Vérification de la sauvegarde du modèle...")

        theta = train_one_vs_all(
            self.X, self.y, self.num_labels, iterations=500)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_logreg_model.npy")
            save_model(theta, self.label_dict, None, None, model_path)

            self.assertTrue(os.path.exists(model_path),
                            Fore.RED +
                            "❌ Le fichier modèle n'a pas été créé !")
            print(Fore.GREEN +
                  "✅ Modèle sauvegardé avec succès dans :", model_path)


if __name__ == "__main__":
    unittest.main()
