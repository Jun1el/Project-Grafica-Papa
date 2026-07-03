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
    import keras

    # Parche para el bug de GlorotUniform en Keras 3 al cargar H5 de ResNet
    orig_glorot_init = keras.initializers.GlorotUniform.__init__
    def safe_glorot_init(self, seed=None, **kwargs):
        orig_glorot_init(self, seed=seed)
    keras.initializers.GlorotUniform.__init__ = safe_glorot_init

    orig_zeros_init = keras.initializers.Zeros.__init__
    def safe_zeros_init(self, **kwargs):
        orig_zeros_init(self)
    keras.initializers.Zeros.__init__ = safe_zeros_init

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


def load_tflite_model(model_path: Path):
    """Carga un modelo TFLite y prepara el interprete."""
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo TFLite: {model_path}")
        
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("El modelo TFLite debe tener una entrada y una salida")
    if tuple(input_details[0]["shape"]) != (1, *IMAGE_SIZE, 3):
        raise ValueError("La entrada TFLite no coincide con 224x224 RGB")
    if tuple(output_details[0]["shape"]) != (1, len(CLASS_NAMES)):
        raise ValueError("La salida TFLite no coincide con las tres clases")
    return interpreter


def predict_tflite(interpreter, image_rgb):
    """Clasifica una imagen RGB usando TFLite y devuelve el resultado."""
    import cv2
    import numpy as np

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    resized = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype("float32"), axis=0)

    input_dtype = input_details[0]["dtype"]
    if input_dtype != batch.dtype:
        raise ValueError(
            f"El modelo TFLite espera {input_dtype}, no {batch.dtype}"
        )

    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()
    
    probabilities = np.asarray(
        interpreter.get_tensor(output_details[0]["index"])[0], dtype=float
    )
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
