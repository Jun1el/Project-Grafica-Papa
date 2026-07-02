"""Ejecuta el fine-tuning del modelo pre-entrenado."""

import argparse
from pathlib import Path

from papa_disease.config import (
    RETRAINED_MODEL_PATH,
    SPLIT_MANIFEST_PATH,
)
from papa_disease.data import build_datasets
from papa_disease.training import do_finetune


def main() -> None:
    parser = argparse.ArgumentParser(description="Aplica fine-tuning al modelo de papas")
    parser.add_argument("--input-model", type=Path, default=RETRAINED_MODEL_PATH)
    parser.add_argument(
        "--output-model", type=Path, default=Path("modelo_papas_finetuned.h5")
    )
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"No se encontro {args.manifest}. Debe correr scripts.train o scripts.prepare_split primero.")

    print(f"Cargando dataset desde {args.manifest}...")
    train_ds, val_ds, _ = build_datasets(args.manifest)

    print("Iniciando fine-tuning...")
    do_finetune(train_ds, val_ds, args.input_model, args.output_model)
    print(f"Fine-tuning completado. Modelo guardado en {args.output_model}.")


if __name__ == "__main__":
    main()
