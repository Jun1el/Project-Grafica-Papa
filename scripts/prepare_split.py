"""Crea el manifiesto estratificado de entrenamiento, validacion y prueba."""

import argparse
from pathlib import Path

from papa_disease.config import DATASET_PATH, SPLIT_MANIFEST_PATH
from papa_disease.data import create_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara el split 70/15/15")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=SPLIT_MANIFEST_PATH)
    args = parser.parse_args()
    manifest = create_split_manifest(args.dataset, args.output)
    print(f"Manifiesto creado en {args.output}")
    print(manifest["counts"])


if __name__ == "__main__":
    main()
