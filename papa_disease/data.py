"""Particion estratificada y carga reproducible del dataset."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .config import (
    ALLOWED_IMAGE_SUFFIXES,
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    RANDOM_SEED,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)


def resolve_class_directory(dataset_path: Path) -> Path:
    """Localiza el directorio que contiene las tres carpetas de clases."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    expected = set(CLASS_NAMES)
    directories = {item.name for item in dataset_path.iterdir() if item.is_dir()}
    if expected.issubset(directories):
        return dataset_path

    children = [item for item in dataset_path.iterdir() if item.is_dir()]
    if len(children) == 1:
        nested = {item.name for item in children[0].iterdir() if item.is_dir()}
        if expected.issubset(nested):
            return children[0]

    raise ValueError("El dataset no contiene las tres carpetas de clases esperadas")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_counts(total: int) -> tuple[int, int, int]:
    train = round(total * TRAIN_FRACTION)
    validation = round(total * VALIDATION_FRACTION)
    test = total - train - validation
    if min(train, validation, test) < 1:
        raise ValueError("Cada clase necesita al menos una imagen por particion")
    return train, validation, test


def create_split_manifest(
    dataset_path: Path,
    output_path: Path,
    seed: int = RANDOM_SEED,
) -> dict:
    """Crea un manifiesto 70/15/15 estratificado y libre de duplicados."""
    class_directory = resolve_class_directory(dataset_path)
    records = []
    hashes = {}

    for class_name in CLASS_NAMES:
        paths = sorted(
            path
            for path in (class_directory / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"La clase {class_name} no contiene imagenes compatibles")

        class_records = []
        for path in paths:
            digest = _sha256(path)
            if digest in hashes:
                raise ValueError(
                    "Imagen duplicada detectada: "
                    f"{path} coincide con {hashes[digest]}"
                )
            hashes[digest] = path
            class_records.append(
                {
                    "path": path.relative_to(class_directory).as_posix(),
                    "class_id": class_name,
                    "sha256": digest,
                }
            )

        random.Random(f"{seed}:{class_name}").shuffle(class_records)
        train_count, validation_count, _ = _split_counts(len(class_records))
        for index, record in enumerate(class_records):
            if index < train_count:
                record["split"] = "train"
            elif index < train_count + validation_count:
                record["split"] = "validation"
            else:
                record["split"] = "test"
        records.extend(class_records)

    manifest = {
        "version": 1,
        "seed": seed,
        "dataset_root": str(class_directory.resolve()),
        "fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "counts": dict(Counter(record["split"] for record in records)),
        "records": records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split_manifest(manifest_path: Path) -> dict:
    """Carga y valida la estructura minima de un manifiesto."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe el manifiesto: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("records"):
        raise ValueError("El manifiesto no tiene un formato compatible")
    if {record.get("split") for record in manifest["records"]} != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("El manifiesto debe contener train, validation y test")
    return manifest


def _build_tf_dataset(manifest: dict, split: str, training: bool):
    import tensorflow as tf

    root = Path(manifest["dataset_root"])
    selected = [record for record in manifest["records"] if record["split"] == split]
    paths = [str(root / record["path"]) for record in selected]
    labels = [CLASS_NAMES.index(record["class_id"]) for record in selected]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(
            len(paths), seed=manifest["seed"], reshuffle_each_iteration=True
        )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_datasets(manifest_path: Path):
    """Construye train, validation y test desde un manifiesto congelado."""
    manifest = load_split_manifest(manifest_path)
    return (
        _build_tf_dataset(manifest, "train", training=True),
        _build_tf_dataset(manifest, "validation", training=False),
        _build_tf_dataset(manifest, "test", training=False),
    )
