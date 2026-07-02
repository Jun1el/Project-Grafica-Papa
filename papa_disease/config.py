"""Configuracion compartida por entrenamiento e inferencia."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "modelo_papas_resnet50.h5"
RETRAINED_MODEL_PATH = PROJECT_ROOT / "modelo_papas_resnet50_split70_15_15.h5"
DATASET_PATH = PROJECT_ROOT / "dataset_papa"
FIELD_DATASET_PATH = PROJECT_ROOT / "dataset_campo"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "split_manifest.json"
REPORTS_PATH = PROJECT_ROOT / "outputs" / "evaluation"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 123
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
UNCERTAINTY_THRESHOLD = 0.70

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

