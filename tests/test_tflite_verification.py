"""Pruebas ligeras del criterio de equivalencia TFLite."""

import unittest

from papa_disease.config import CLASS_NAMES
from scripts.verify_tflite import compare_results


def _result(class_id, probabilities):
    return {"class_id": class_id, "probabilities": probabilities}


class TFLiteVerificationTests(unittest.TestCase):
    def setUp(self):
        self.keras_result = _result(
            CLASS_NAMES[0], dict(zip(CLASS_NAMES, (0.80, 0.15, 0.05)))
        )

    def test_accepts_same_class_within_tolerance(self):
        tflite_result = _result(
            CLASS_NAMES[0], dict(zip(CLASS_NAMES, (0.795, 0.153, 0.052)))
        )
        comparison = compare_results(
            self.keras_result, tflite_result, max_difference=0.01
        )
        self.assertTrue(comparison["passed"])
        self.assertAlmostEqual(
            comparison["max_probability_difference"], 0.005
        )

    def test_rejects_changed_class(self):
        tflite_result = _result(
            CLASS_NAMES[1], dict(zip(CLASS_NAMES, (0.40, 0.55, 0.05)))
        )
        comparison = compare_results(
            self.keras_result, tflite_result, max_difference=1.0
        )
        self.assertFalse(comparison["passed"])
        self.assertFalse(comparison["same_class"])

    def test_rejects_difference_over_tolerance(self):
        tflite_result = _result(
            CLASS_NAMES[0], dict(zip(CLASS_NAMES, (0.75, 0.20, 0.05)))
        )
        comparison = compare_results(
            self.keras_result, tflite_result, max_difference=0.01
        )
        self.assertFalse(comparison["passed"])


if __name__ == "__main__":
    unittest.main()
