"""Convierte el modelo H5 a TFLite optimizado para Android."""

import argparse
from pathlib import Path

from papa_disease.config import IMAGE_SIZE
from papa_disease.inference import load_model


def export_tflite(input_model_path: Path, output_path: Path) -> None:
    import tensorflow as tf
    import keras
    
    # --- MONKEY PATCH PARA KERAS 3 ---
    # Keras 3 guarda los modelos H5 con 'input_axes' y 'output_axes' en los initializers,
    # pero el constructor no los acepta, rompiendo load_model(). 
    # Parcheamos el constructor dinámicamente para que ignore kwargs adicionales.
    
    # Parche para GlorotUniform
    orig_glorot_init = keras.initializers.GlorotUniform.__init__
    def safe_glorot_init(self, seed=None, **kwargs):
        orig_glorot_init(self, seed=seed)
    keras.initializers.GlorotUniform.__init__ = safe_glorot_init
    
    # Parche para Zeros (por si acaso también falla)
    orig_zeros_init = keras.initializers.Zeros.__init__
    def safe_zeros_init(self, **kwargs):
        orig_zeros_init(self)
    keras.initializers.Zeros.__init__ = safe_zeros_init
    # ---------------------------------
    
    print(f"Cargando modelo desde {input_model_path}...")
    model = load_model(input_model_path)
    
    print("Iniciando conversión a TFLite (Dynamic Range Quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Habilitamos cuantización por defecto para reducir el tamaño al ~25%
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    original_size = input_model_path.stat().st_size / (1024 * 1024)
    tflite_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"Modelo TFLite guardado en {output_path}")
    print(f"Tamaño original (H5): {original_size:.2f} MB")
    print(f"Tamaño optimizado (TFLite): {tflite_size:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta el modelo a TFLite")
    parser.add_argument("--input-model", type=Path, default=Path("modelo_papas_finetuned.h5"))
    parser.add_argument("--output-model", type=Path, default=Path("modelo_papas.tflite"))
    args = parser.parse_args()
    
    if not args.input_model.exists():
        raise FileNotFoundError(f"No se encontro {args.input_model}. Ejecute scripts.finetune primero.")
        
    export_tflite(args.input_model, args.output_model)

if __name__ == "__main__":
    main()
