import unittest
import numpy as np
import os
import tempfile
from app.logreg_roc import (
    compute_roc_auc,
    find_optimal_threshold,
    plot_roc,
    plot_super_roc,
    export_roc_data
)


class TestLogRegROC(unittest.TestCase):
    """🧪 Tests unitaires pour logreg_roc.py"""

    @classmethod
    def setUpClass(cls):
        """Préparation du fichier best_features.txt"""
        cls.best_features_file = "data/best_features.txt"
        os.makedirs("data", exist_ok=True)
        with open(cls.best_features_file, "w") as f:
            f.write("Flying\n"
                    "Charms\n"
                    "Divination\n"
                    "Transfiguration\n"
                    "History of Magic\n")

    @classmethod
    def tearDownClass(cls):
        """Nettoyage du fichier temporaire"""
        if os.path.exists(cls.best_features_file):
            os.remove(cls.best_features_file)

    def setUp(self):
        np.random.seed(0)
        self.y_true_bin = np.random.randint(0, 2, 100)
        self.y_scores = np.random.rand(100)
        self.label_names = ["Gryffindor", "Hufflepuff"]

    def test_compute_roc_auc(self):
        """✅ Vérifie les calculs ROC et AUC"""
        fpr, tpr, auc, thresholds = compute_roc_auc(
            self.y_true_bin, self.y_scores)

        self.assertTrue(len(fpr) > 0)
        self.assertTrue(0 <= auc <= 1)

    def test_find_optimal_threshold(self):
        """✅ Vérifie la recherche du seuil optimal"""
        fpr, tpr, auc, thresholds = compute_roc_auc(
            self.y_true_bin, self.y_scores)
        opt_fpr, opt_tpr, opt_thresh = find_optimal_threshold(
            fpr, tpr, thresholds)

        self.assertTrue(0 <= opt_fpr <= 1)
        self.assertTrue(0 <= opt_tpr <= 1)

    def test_plot_roc_and_super_roc(self):
        """✅ Vérifie la génération des plots ROC"""
        fpr, tpr, auc, _ = compute_roc_auc(self.y_true_bin, self.y_scores)

        with tempfile.TemporaryDirectory() as tmp_dir:
            single_roc_file = os.path.join(tmp_dir, "roc.png")
            plot_roc(fpr, tpr, auc, "Gryffindor", single_roc_file)

            super_roc_file = os.path.join(tmp_dir, "super_roc.png")
            roc_data = [("Gryffindor", fpr, tpr, auc)]
            plot_super_roc(roc_data, super_roc_file)

            self.assertTrue(os.path.exists(single_roc_file))
            self.assertTrue(os.path.exists(super_roc_file))

    def test_export_roc_data(self):
        """✅ Vérifie l'export des données ROC"""
        fpr, tpr, auc, _ = compute_roc_auc(self.y_true_bin, self.y_scores)
        roc_data = [("Gryffindor", fpr, tpr, auc)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_file = os.path.join(tmp_dir, "roc_data.csv")
            export_roc_data(roc_data, csv_file)
            self.assertTrue(os.path.exists(csv_file))


if __name__ == "__main__":
    unittest.main()
