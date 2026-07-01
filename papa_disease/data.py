"""Creacion reproducible de los conjuntos de entrenamiento y validacion."""

from pathlib import Path

from .config import BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED


def resolve_class_directory(dataset_path: Path) -> Path:
    """Localiza el directorio que contiene las tres carpetas de clases."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    expected = {
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
    }
    directories = {item.name for item in dataset_path.iterdir() if item.is_dir()}
    if expected.issubset(directories):
        return dataset_path

    children = [item for item in dataset_path.iterdir() if item.is_dir()]
    if len(children) == 1:
        nested = {item.name for item in children[0].iterdir() if item.is_dir()}
        if expected.issubset(nested):
            return children[0]

    raise ValueError("El dataset no contiene las tres carpetas de clases esperadas")


def build_datasets(dataset_path: Path):
    """Construye el split 80/20 usado por el modelo original."""
    import tensorflow as tf

    class_directory = resolve_class_directory(dataset_path)
    common = dict(
        directory=str(class_directory),
        validation_split=0.2,
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    train = tf.keras.utils.image_dataset_from_directory(
        subset="training", **common
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        subset="validation", **common
    )

    autotune = tf.data.AUTOTUNE
    train = train.cache().shuffle(1000).prefetch(buffer_size=autotune)
    validation = validation.cache().prefetch(buffer_size=autotune)
    return train, validation

