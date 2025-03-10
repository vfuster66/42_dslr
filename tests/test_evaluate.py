import unittest
import numpy as np
import os
import tempfile
from app.logreg_evaluate import (
    evaluate_accuracy,
    confusion_matrix,
    classification_report,
    plot_confusion_matrix_heatmap
)


class TestLogRegEvaluate(unittest.TestCase):
    """🧪 Tests unitaires pour logreg_evaluate.py"""

    @classmethod
    def setUpClass(cls):
        """Préparation du fichier best_features.txt"""
        cls.best_features_file = "data/best_features.txt"
        os.makedirs("data", exist_ok=True)
        with open(cls.best_features_file, "w") as f:
            f.write(
                "Flying\nCharms\nDivination\nTransfiguration\nHistory of Magic"
                "\n")

    @classmethod
    def tearDownClass(cls):
        """Nettoyage du fichier temporaire"""
        if os.path.exists(cls.best_features_file):
            os.remove(cls.best_features_file)

    def setUp(self):
        self.y_true = np.array([0, 1, 2, 1, 0, 2])
        self.y_pred = np.array([0, 1, 2, 0, 0, 2])
        self.label_names = ["Gryffindor", "Hufflepuff", "Ravenclaw"]

    def test_evaluate_accuracy(self):
        """✅ Vérifie la précision"""
        acc = evaluate_accuracy(self.y_pred, self.y_true)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 100)

    def test_confusion_matrix(self):
        """✅ Vérifie la matrice de confusion"""
        matrix = confusion_matrix(
            self.y_true, self.y_pred, len(self.label_names)
        )
        self.assertEqual(
            matrix.shape, (len(self.label_names), len(self.label_names))
        )

    def test_classification_report_and_file(self):
        """✅ Vérifie que le rapport est généré et sauvegardé"""
        matrix = confusion_matrix(
            self.y_true, self.y_pred, len(self.label_names)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = os.path.join(tmp_dir, "classification_report.txt")
            classification_report(matrix, self.label_names, report_path)
            self.assertTrue(os.path.exists(report_path))

    def test_plot_confusion_matrix_heatmap(self):
        """✅ Vérifie que la heatmap de la
        matrice de confusion est sauvegardée"""
        matrix = confusion_matrix(
            self.y_true, self.y_pred, len(self.label_names))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_img = os.path.join(tmp_dir, "heatmap.png")
            plot_confusion_matrix_heatmap(matrix, self.label_names, output_img)
            self.assertTrue(os.path.exists(output_img))


if __name__ == "__main__":
    unittest.main()
