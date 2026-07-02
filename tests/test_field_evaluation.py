"""Pruebas del recolector de imagenes de campo (sin dependencias pesadas)."""

import tempfile
import unittest
from pathlib import Path

from papa_disease.config import CLASS_NAMES
from papa_disease.evaluation import collect_field_images


class FieldEvaluationTests(unittest.TestCase):
    def _write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")

    def test_collects_subset_of_classes_and_ignores_non_images(self):
        early, late, healthy = CLASS_NAMES
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root / early / "a.jpg")
            self._write(root / early / "b.PNG")
            self._write(root / late / "c.jpeg")
            self._write(root / early / "notas.txt")  # debe ignorarse
            (root / healthy).mkdir()  # carpeta vacia: aporta 0 muestras

            samples = collect_field_images(root)
            classes = [class_name for class_name, _ in samples]

            self.assertEqual(len(samples), 3)
            self.assertEqual(classes.count(early), 2)
            self.assertEqual(classes.count(late), 1)
            self.assertNotIn(healthy, classes)

    def test_tolerates_missing_class_folder(self):
        early = CLASS_NAMES[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root / early / "a.jpg")
            samples = collect_field_images(root)
            self.assertEqual([name for name, _ in samples], [early])

    def test_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "carpetas de clase"):
                collect_field_images(Path(temporary))

    def test_rejects_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            collect_field_images(Path("directorio_de_campo_inexistente"))


if __name__ == "__main__":
    unittest.main()
