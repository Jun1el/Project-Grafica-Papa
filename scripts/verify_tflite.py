"""Valida que el modelo TFLite genere predicciones similares al H5."""

import argparse
from pathlib import Path
import time

from papa_disease.inference import (
    load_image,
    load_model,
    predict,
    load_tflite_model,
    predict_tflite,
)


def verify_models(h5_path: Path, tflite_path: Path, image_path: Path = None):
    if image_path:
        print(f"\n--- Verificando {image_path.name} ---")
        image_rgb = load_image(image_path)
    else:
        print("\n--- Verificando con imagen sintética (dummy) ---")
        import numpy as np
        image_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # Evaluar H5
    print("Cargando modelo H5...")
    model_h5 = load_model(h5_path)
    
    # Calentar (opcional) para métricas estables
    predict(model_h5, image_rgb)
    
    print("Inferencia H5...")
    start_time = time.time()
    res_h5 = predict(model_h5, image_rgb)
    time_h5 = time.time() - start_time
    
    # Evaluar TFLite
    print("\nCargando modelo TFLite...")
    interpreter = load_tflite_model(tflite_path)
    
    # Calentar TFLite
    predict_tflite(interpreter, image_rgb)
    
    print("Inferencia TFLite...")
    start_time = time.time()
    res_tflite = predict_tflite(interpreter, image_rgb)
    time_tflite = time.time() - start_time
    
    # Comparar resultados
    print(f"\nResultados:")
    print(f"H5     -> Clase: {res_h5['class_id']}, Confianza: {res_h5['confidence']:.4f}, Tiempo: {time_h5*1000:.1f} ms")
    print(f"TFLite -> Clase: {res_tflite['class_id']}, Confianza: {res_tflite['confidence']:.4f}, Tiempo: {time_tflite*1000:.1f} ms")
    
    if res_h5["class_id"] != res_tflite["class_id"]:
        print("ERROR: Los modelos predicen clases diferentes.")
        return False
        
    diff = abs(res_h5["confidence"] - res_tflite["confidence"])
    if diff > 0.05:
        print(f"WARNING: La diferencia de confianza es mayor al 5% ({diff:.4f}).")
    else:
        print(f"\nCoincidencia Exitosa. Diferencia de confianza: {diff:.5f}")
        
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara predicciones entre H5 y TFLite")
    parser.add_argument("--h5-model", type=Path, default=Path("modelo_papas_finetuned.h5"))
    parser.add_argument("--tflite-model", type=Path, default=Path("modelo_papas.tflite"))
    parser.add_argument("--image", type=Path, help="Ruta de la imagen de prueba")
    parser.add_argument("--dummy", action="store_true", help="Usa una imagen de ruido generada")
    args = parser.parse_args()
    
    if not args.h5_model.exists():
        raise FileNotFoundError(f"No existe el modelo H5 {args.h5_model}")
    if not args.tflite_model.exists():
        raise FileNotFoundError(f"No existe el modelo TFLite {args.tflite_model}")
        
    if not args.image and not args.dummy:
        raise ValueError("Debe proporcionar --image o --dummy")
        
    verify_models(args.h5_model, args.tflite_model, args.image)


if __name__ == "__main__":
    main()
