"""Reproduce el entrenamiento base a partir del dataset descargado."""

from papa_disease.config import DATASET_PATH
from papa_disease.data import build_datasets
from papa_disease.training import train_model


def main() -> None:
    train_dataset, validation_dataset = build_datasets(DATASET_PATH)
    train_model(train_dataset, validation_dataset)


if __name__ == "__main__":
    main()

