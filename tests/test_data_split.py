"""Pruebas del reparto estratificado y de deteccion de duplicados."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from papa_disease.config import CLASS_NAMES
from papa_disease.data import create_split_manifest, load_split_manifest


class DataSplitTests(unittest.TestCase):
    def _dataset(self, root: Path, images_per_class: int = 20) -> Path:
        dataset = root / "dataset"
        for class_index, class_name in enumerate(CLASS_NAMES):
            directory = dataset / class_name
            directory.mkdir(parents=True)
            for image_index in range(images_per_class):
                content = f"{class_index}:{image_index}".encode()
                (directory / f"image_{image_index}.jpg").write_bytes(content)
        return dataset

    def test_split_is_stratified_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            first = create_split_manifest(dataset, root / "first.json")
            second = create_split_manifest(dataset, root / "second.json")

            first_assignments = [(r["path"], r["split"]) for r in first["records"]]
            second_assignments = [(r["path"], r["split"]) for r in second["records"]]
            self.assertEqual(first_assignments, second_assignments)
            for class_name in CLASS_NAMES:
                counts = Counter(
                    record["split"]
                    for record in first["records"]
                    if record["class_id"] == class_name
                )
                self.assertEqual(counts, {"train": 14, "validation": 3, "test": 3})
            self.assertEqual(load_split_manifest(root / "first.json")["version"], 1)

    def test_duplicate_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            duplicate = dataset / CLASS_NAMES[1] / "duplicate.jpg"
            duplicate.write_bytes((dataset / CLASS_NAMES[0] / "image_0.jpg").read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicada"):
                create_split_manifest(dataset, root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
