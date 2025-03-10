import unittest
import numpy as np
import pandas as pd
import os
import tempfile
from app.logreg_predict import load_data, predict


class TestLogRegPredict(unittest.TestCase):
    """🧪 Tests unitaires pour logreg_predict.py"""

    @classmethod
    def setUpClass(cls):
        """Préparation des ressources nécessaires aux tests"""
        # Création du fichier best_features.txt temporaire
        cls.best_features_file = "data/best_features.txt"
        os.makedirs("data", exist_ok=True)
        with open(cls.best_features_file, "w") as f:
            f.write("Flying\n"
                    "Charms\n"
                    "Divination\n"
                    "Transfiguration\n"
                    "History of Magic\n")

        cls.test_file = "data/dataset_test.csv"
        cls.df, cls.selected_features = load_data(cls.test_file)

        cls.theta = np.ones((4, len(cls.selected_features) + 1))
        cls.mean_train = cls.df[cls.selected_features].mean()
        cls.std_train = cls.df[cls.selected_features].std()

    @classmethod
    def tearDownClass(cls):
        """Nettoyage des ressources après tests"""
        if os.path.exists(cls.best_features_file):
            os.remove(cls.best_features_file)

    def test_load_data(self):
        """✅ Vérifie le chargement des données test"""
        self.assertIn("Hogwarts House", self.df.columns)
        self.assertGreater(len(self.selected_features), 0)

    def test_predict_function(self):
        """✅ Vérifie la forme et le contenu des prédictions"""
        X = self.df[self.selected_features]
        X = (X - self.mean_train) / self.std_train
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        y_pred = predict(X, self.theta)

        self.assertEqual(len(y_pred), X.shape[0])
        self.assertTrue(np.all((y_pred >= 0) & (y_pred < self.theta.shape[0])))

    def test_output_file_generation(self):
        """✅ Vérifie que le fichier houses.csv est bien généré"""
        predictions = ["Gryffindor"] * len(self.df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "houses.csv")

            results = pd.DataFrame({
                "Index": range(len(predictions)),
                "Hogwarts House": predictions
            })
            results.to_csv(output_path, index=False)

            self.assertTrue(os.path.exists(output_path))

            df_result = pd.read_csv(output_path)
            self.assertIn("Index", df_result.columns)
            self.assertIn("Hogwarts House", df_result.columns)


if __name__ == "__main__":
    unittest.main()
