"""Pruebas deterministas de metricas y umbral de incertidumbre."""

import unittest

from papa_disease.config import CLASS_NAMES
from papa_disease.evaluation import compute_metrics


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_confusion_matrix(self):
        early, late, healthy = CLASS_NAMES
        report = compute_metrics(
            [early, early, late, late, healthy, healthy],
            [early, late, late, late, healthy, early],
            [0.9, 0.6, 0.8, 0.9, 0.95, 0.55],
            uncertainty_threshold=0.7,
        )
        self.assertAlmostEqual(report["accuracy"], 4 / 6)
        self.assertAlmostEqual(report["coverage"], 4 / 6)
        self.assertEqual(report["uncertain_samples"], 2)
        self.assertEqual(report["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(report["per_class"][healthy]["recall"], 0.5)

    def test_rejects_unknown_classes(self):
        with self.assertRaisesRegex(ValueError, "desconocidas"):
            compute_metrics(["unknown"], [CLASS_NAMES[0]], [0.5])

    def test_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "umbral"):
            compute_metrics([CLASS_NAMES[0]], [CLASS_NAMES[0]], [0.9], 1.1)


if __name__ == "__main__":
    unittest.main()
