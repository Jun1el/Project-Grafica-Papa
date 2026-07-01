"""Configuracion compartida por entrenamiento e inferencia."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "modelo_papas_resnet50.h5"
DATASET_PATH = PROJECT_ROOT / "dataset_papa"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 123

# image_dataset_from_directory ordena alfabeticamente los directorios.
# Este orden debe permanecer sincronizado con las tres salidas del modelo.
CLASS_NAMES = (
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
)

CLASS_NAMES_ES = {
    "Potato___Early_blight": "Tizon temprano",
    "Potato___Late_blight": "Tizon tardio",
    "Potato___healthy": "Hoja sana",
}

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

