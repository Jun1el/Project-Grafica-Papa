"""Pruebas sin dependencias pesadas del contrato compartido."""

import tempfile
import unittest
from pathlib import Path

from papa_disease.config import CLASS_NAMES, CLASS_NAMES_ES, IMAGE_SIZE
from papa_disease.data import resolve_class_directory


class ContractTests(unittest.TestCase):
    def test_class_contract_is_complete_and_ordered(self):
        self.assertEqual(IMAGE_SIZE, (224, 224))
        self.assertEqual(
            CLASS_NAMES,
            (
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Potato___healthy",
            ),
        )
        self.assertEqual(set(CLASS_NAMES), set(CLASS_NAMES_ES))

    def test_resolves_nested_kaggle_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "Potato_Disease"
            for name in CLASS_NAMES:
                (nested / name).mkdir(parents=True)
            self.assertEqual(resolve_class_directory(root), nested)

    def test_rejects_incomplete_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                resolve_class_directory(Path(temporary))


if __name__ == "__main__":
    unittest.main()
