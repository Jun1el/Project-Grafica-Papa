"""Carga segura de imagenes y contrato unico de inferencia."""

from pathlib import Path

from .config import (
    ALLOWED_IMAGE_SUFFIXES,
    CLASS_NAMES,
    CLASS_NAMES_ES,
    IMAGE_SIZE,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MODEL_PATH,
)


def load_model(model_path: Path = MODEL_PATH):
    """Carga el artefacto local confiable sin restaurar el optimizador.

    ``safe_mode=False`` es necesario por la capa Lambda incluida en el H5
    original. Nunca debe usarse esta funcion con modelos recibidos de terceros.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input},
    )
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("El modelo cargado no tiene exactamente tres salidas")
    return model


def load_image(image_path: Path):
    """Valida y decodifica JPG/PNG como RGB."""
    import cv2
    import numpy as np

    image_path = Path(image_path)
    if image_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("Formato no permitido; use JPG, JPEG o PNG")
    if not image_path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("La imagen supera el limite de 10 MB")

    encoded = np.fromfile(image_path, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("El archivo no contiene una imagen valida")
    height, width = image_bgr.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError("La imagen supera el limite de 25 megapixeles")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def predict(model, image_rgb):
    """Clasifica una imagen RGB y devuelve un resultado serializable."""
    import cv2
    import numpy as np

    resized = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype("float32"), axis=0)
    probabilities = np.asarray(model.predict(batch, verbose=0)[0], dtype=float)
    if probabilities.shape != (len(CLASS_NAMES),):
        raise ValueError("La salida del modelo no coincide con las tres clases")

    index = int(np.argmax(probabilities))
    class_name = CLASS_NAMES[index]
    return {
        "class_id": class_name,
        "label": CLASS_NAMES_ES[class_name],
        "confidence": float(probabilities[index]),
        "probabilities": {
            name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "disclaimer": "Resultado educativo; no sustituye una evaluacion agronomica.",
    }
