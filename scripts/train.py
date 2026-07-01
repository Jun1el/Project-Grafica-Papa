"""Reentrena la linea base con el reparto 70/15/15."""

import argparse
from pathlib import Path

from papa_disease.config import DATASET_PATH, RETRAINED_MODEL_PATH, SPLIT_MANIFEST_PATH
from papa_disease.data import build_datasets, create_split_manifest
from papa_disease.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena la linea base ResNet50")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--output-model", type=Path, default=RETRAINED_MODEL_PATH)
    parser.add_argument(
        "--recreate-split",
        action="store_true",
        help="Vuelve a crear el manifiesto incluso si ya existe",
    )
    args = parser.parse_args()

    if args.recreate_split or not args.manifest.exists():
        create_split_manifest(args.dataset, args.manifest)
    train_dataset, validation_dataset, _ = build_datasets(args.manifest)
    train_model(train_dataset, validation_dataset, args.output_model)


if __name__ == "__main__":
    main()
