"""Convierte el modelo H5 a TFLite optimizado para Android."""

import argparse
from pathlib import Path

from papa_disease.inference import load_model


def export_tflite(input_model_path: Path, output_path: Path) -> None:
    """Exporta con cuantizacion dinamica y reemplazo atomico."""
    import tensorflow as tf

    model = load_model(input_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_bytes(tflite_model)
    temporary_path.replace(output_path)

    original_size = input_model_path.stat().st_size / (1024 * 1024)
    tflite_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Modelo TFLite guardado en {output_path}")
    print(f"Tamaño original (H5): {original_size:.2f} MB")
    print(f"Tamaño optimizado (TFLite): {tflite_size:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta el modelo a TFLite")
    parser.add_argument(
        "--input-model", type=Path, default=Path("modelo_papas_finetuned.h5")
    )
    parser.add_argument(
        "--output-model", type=Path, default=Path("modelo_papas.tflite")
    )
    args = parser.parse_args()
    if not args.input_model.is_file():
        raise FileNotFoundError(
            f"No existe {args.input_model}; ejecute scripts.finetune primero"
        )
    export_tflite(args.input_model, args.output_model)


if __name__ == "__main__":
    main()
