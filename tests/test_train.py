import unittest
import numpy as np
import os
from app.logreg_train import (
    load_data,
    encode_labels,
    train_one_vs_all,
    save_model
)


class TestLogRegTrain(unittest.TestCase):
    """🧪 Tests unitaires pour logreg_train.py"""

    @classmethod
    def setUpClass(cls):
        """🔍 Chargement des données pour les tests"""
        cls.filepath = "data/dataset_train.csv"
        cls.df = load_data(cls.filepath)
        cls.df, cls.label_dict = encode_labels(cls.df)

        cls.X = cls.df.iloc[:, 1:-1].values
        cls.y = cls.df["House Label"].values
        cls.X = np.hstack([np.ones((cls.X.shape[0], 1)), cls.X])
        cls.num_labels = len(cls.label_dict)

    def test_train_one_vs_all(self):
        """✅ Vérifie que l'entraînement produit des poids non nuls"""
        print("🔍 Test d'entraînement en cours...")
        theta = train_one_vs_all(
            self.X, self.y, self.num_labels, iterations=500
        )
        self.assertFalse(
            np.all(theta == 0),
            "❌ Les poids sont tous à zéro, l'entraînement n'a pas fonctionné."
        )
        print("✅ Entraînement réussi avec des poids appris !")

    def test_save_model(self):
        """✅ Vérifie que le modèle est bien sauvegardé"""
        theta = train_one_vs_all(
            self.X, self.y, self.num_labels, iterations=500
        )
        filepath = "data/test_logreg_model.npy"
        save_model(theta, filepath)
        self.assertTrue(os.path.exists(filepath),
                        "❌ Le fichier modèle n'a pas été créé !")
        print("✅ Modèle sauvegardé avec succès !")
        os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
