"""Metricas por clase y comparacion de inferencia con Watershed."""

import json
from pathlib import Path

from .config import (
    ALLOWED_IMAGE_SUFFIXES,
    CLASS_NAMES,
    CLASS_NAMES_ES,
    UNCERTAINTY_THRESHOLD,
)
from .data import load_split_manifest
from .inference import load_image, predict
from .segmentation import apply_watershed


def compute_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    confidences: list[float],
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
) -> dict:
    """Calcula accuracy, precision, recall, F1 y matriz de confusion."""
    if not true_labels or len(true_labels) != len(predicted_labels):
        raise ValueError("Las etiquetas reales y predichas deben tener igual longitud")
    if len(confidences) != len(true_labels):
        raise ValueError("Debe existir una confianza por prediccion")
    if not 0.0 <= uncertainty_threshold <= 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1")

    unknown = (set(true_labels) | set(predicted_labels)) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Se encontraron clases desconocidas: {sorted(unknown)}")

    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for truth, prediction in zip(true_labels, predicted_labels):
        matrix[CLASS_NAMES.index(truth)][CLASS_NAMES.index(prediction)] += 1

    per_class = {}
    f1_values = []
    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[index][index]
        false_positive = sum(row[index] for row in matrix) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        support = sum(matrix[index])
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[class_name] = {
            "label": CLASS_NAMES_ES[class_name],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    correct = sum(matrix[i][i] for i in range(len(CLASS_NAMES)))
    uncertain_indices = [
        index for index, confidence in enumerate(confidences)
        if confidence < uncertainty_threshold
    ]
    accepted_indices = [
        index for index, confidence in enumerate(confidences)
        if confidence >= uncertainty_threshold
    ]
    accepted_correct = sum(
        true_labels[index] == predicted_labels[index] for index in accepted_indices
    )
    return {
        "samples": len(true_labels),
        "accuracy": correct / len(true_labels),
        "macro_f1": sum(f1_values) / len(f1_values),
        "uncertainty_threshold": uncertainty_threshold,
        "uncertain_samples": len(uncertain_indices),
        "coverage": len(accepted_indices) / len(true_labels),
        "accepted_accuracy": (
            accepted_correct / len(accepted_indices) if accepted_indices else None
        ),
        "class_order": list(CLASS_NAMES),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def _segment_rgb(image_rgb):
    import cv2

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = apply_watershed(image_bgr)
    return cv2.cvtColor(result["isolated"], cv2.COLOR_BGR2RGB)


def evaluate_manifest(
    model,
    manifest_path: Path,
    split: str = "test",
    compare_watershed: bool = True,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
) -> dict:
    """Evalua un split y opcionalmente repite la prueba con segmentacion."""
    if split not in {"validation", "test"}:
        raise ValueError("Solo se permite evaluar validation o test")
    manifest = load_split_manifest(manifest_path)
    root = Path(manifest["dataset_root"])
    records = [record for record in manifest["records"] if record["split"] == split]

    true_labels = []
    raw_predictions = []
    raw_confidences = []
    segmented_predictions = []
    segmented_confidences = []
    samples = []

    for record in records:
        image = load_image(root / record["path"])
        raw = predict(model, image)
        true_labels.append(record["class_id"])
        raw_predictions.append(raw["class_id"])
        raw_confidences.append(raw["confidence"])

        sample = {
            "path": record["path"],
            "truth": record["class_id"],
            "raw_prediction": raw["class_id"],
            "raw_confidence": raw["confidence"],
        }
        if compare_watershed:
            segmented = predict(model, _segment_rgb(image))
            segmented_predictions.append(segmented["class_id"])
            segmented_confidences.append(segmented["confidence"])
            sample.update(
                segmented_prediction=segmented["class_id"],
                segmented_confidence=segmented["confidence"],
            )
        samples.append(sample)

    report = {
        "split": split,
        "manifest_seed": manifest["seed"],
        "raw": compute_metrics(
            true_labels, raw_predictions, raw_confidences, uncertainty_threshold
        ),
        "samples": samples,
    }
    if compare_watershed:
        report["watershed"] = compute_metrics(
            true_labels,
            segmented_predictions,
            segmented_confidences,
            uncertainty_threshold,
        )
        report["comparison"] = {
            "accuracy_delta": report["watershed"]["accuracy"] - report["raw"]["accuracy"],
            "macro_f1_delta": report["watershed"]["macro_f1"] - report["raw"]["macro_f1"],
        }
    return report


def collect_field_images(field_path: Path) -> list[tuple[str, Path]]:
    """Recorre un directorio de campo y devuelve pares (clase, ruta).

    Cada subcarpeta debe llevar el nombre exacto de una clase de ``CLASS_NAMES``.
    A diferencia del manifiesto 70/15/15, aqui se aceptan subconjuntos: si una
    clase no tiene carpeta o esta vacia simplemente no aporta muestras.
    """
    field_path = Path(field_path)
    if not field_path.is_dir():
        raise FileNotFoundError(f"No existe el directorio de campo: {field_path}")

    samples: list[tuple[str, Path]] = []
    for class_name in CLASS_NAMES:
        class_directory = field_path / class_name
        if not class_directory.is_dir():
            continue
        for path in sorted(class_directory.iterdir()):
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
                samples.append((class_name, path))

    if not samples:
        raise ValueError(
            "El directorio de campo no contiene imagenes en carpetas de clase; "
            f"esperaba subcarpetas con nombres de {list(CLASS_NAMES)}"
        )
    return samples


def evaluate_field_directory(
    model,
    field_path: Path,
    compare_watershed: bool = False,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
) -> dict:
    """Evalua un conjunto de campo externo al manifiesto de laboratorio.

    Mide la robustez del modelo fuera del dataset limpio. Tolera que falten
    clases: la matriz de confusion sigue siendo 3x3 y una clase sin muestras
    aparece con soporte 0. ``macro_f1_present`` promedia solo las clases con
    soporte, para no penalizar el resultado por una clase ausente.
    """
    samples_index = collect_field_images(field_path)

    true_labels = []
    raw_predictions = []
    raw_confidences = []
    segmented_predictions = []
    segmented_confidences = []
    samples = []

    for class_name, path in samples_index:
        image = load_image(path)
        raw = predict(model, image)
        true_labels.append(class_name)
        raw_predictions.append(raw["class_id"])
        raw_confidences.append(raw["confidence"])

        sample = {
            "path": path.name,
            "truth": class_name,
            "raw_prediction": raw["class_id"],
            "raw_confidence": raw["confidence"],
        }
        if compare_watershed:
            segmented = predict(model, _segment_rgb(image))
            segmented_predictions.append(segmented["class_id"])
            segmented_confidences.append(segmented["confidence"])
            sample.update(
                segmented_prediction=segmented["class_id"],
                segmented_confidence=segmented["confidence"],
            )
        samples.append(sample)

    raw_metrics = compute_metrics(
        true_labels, raw_predictions, raw_confidences, uncertainty_threshold
    )
    present = sorted(set(true_labels), key=CLASS_NAMES.index)
    missing = [name for name in CLASS_NAMES if name not in present]
    present_f1 = [raw_metrics["per_class"][name]["f1"] for name in present]
    raw_metrics["present_classes"] = present
    raw_metrics["missing_classes"] = missing
    raw_metrics["macro_f1_present"] = (
        sum(present_f1) / len(present_f1) if present_f1 else 0.0
    )

    report = {
        "source": "field",
        "dataset_root": str(Path(field_path).resolve()),
        "raw": raw_metrics,
        "samples": samples,
    }
    if compare_watershed:
        report["watershed"] = compute_metrics(
            true_labels,
            segmented_predictions,
            segmented_confidences,
            uncertainty_threshold,
        )
        report["comparison"] = {
            "accuracy_delta": report["watershed"]["accuracy"] - raw_metrics["accuracy"],
            "macro_f1_delta": report["watershed"]["macro_f1"] - raw_metrics["macro_f1"],
        }
    return report


def save_report(report: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_confusion_matrix(report: dict, output_path: Path, variant: str = "raw") -> None:
    """Guarda la matriz de confusion como PNG."""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = report[variant]
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = [CLASS_NAMES_ES[name] for name in metrics["class_order"]]
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Prediccion",
        ylabel="Clase real",
        title=f"Matriz de confusion: {variant}",
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
