"""Compara de forma reproducible predicciones Keras y TFLite."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from papa_disease.config import CLASS_NAMES, IMAGE_SIZE, REPORTS_PATH
from papa_disease.inference import (
    load_image,
    load_model,
    load_tflite_model,
    predict,
    predict_tflite,
)


def compare_results(keras_result, tflite_result, max_difference: float):
    """Compara clase y probabilidades de dos resultados de inferencia."""
    differences = {
        name: abs(
            keras_result["probabilities"][name]
            - tflite_result["probabilities"][name]
        )
        for name in CLASS_NAMES
    }
    largest_difference = max(differences.values())
    same_class = keras_result["class_id"] == tflite_result["class_id"]
    return {
        "passed": same_class and largest_difference <= max_difference,
        "same_class": same_class,
        "max_probability_difference": largest_difference,
        "probability_differences": differences,
    }


def _timed_prediction(predictor, model, image_rgb, runs: int):
    predictor(model, image_rgb)
    durations = []
    result = None
    for _ in range(runs):
        start = perf_counter()
        result = predictor(model, image_rgb)
        durations.append((perf_counter() - start) * 1000)
    return result, median(durations)


def verify_models(
    h5_path: Path,
    tflite_path: Path,
    image_paths: list[Path] | None = None,
    *,
    max_difference: float = 0.01,
    runs: int = 5,
    seed: int = 123,
):
    """Valida varias entradas y devuelve evidencia serializable."""
    if not 0 <= max_difference <= 1:
        raise ValueError("max_difference debe estar entre 0 y 1")
    if runs < 1:
        raise ValueError("runs debe ser al menos 1")

    if image_paths:
        samples = [(str(path), load_image(path)) for path in image_paths]
    else:
        import numpy as np

        generator = np.random.default_rng(seed)
        dummy = generator.integers(
            0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8
        )
        samples = [(f"dummy-seed-{seed}", dummy)]

    keras_model = load_model(h5_path)
    tflite_model = load_tflite_model(tflite_path)
    sample_reports = []
    for sample_name, image_rgb in samples:
        keras_result, keras_ms = _timed_prediction(
            predict, keras_model, image_rgb, runs
        )
        tflite_result, tflite_ms = _timed_prediction(
            predict_tflite, tflite_model, image_rgb, runs
        )
        comparison = compare_results(
            keras_result, tflite_result, max_difference
        )
        sample_reports.append(
            {
                "sample": sample_name,
                "keras_class": keras_result["class_id"],
                "tflite_class": tflite_result["class_id"],
                "keras_latency_median_ms": keras_ms,
                "tflite_latency_median_ms": tflite_ms,
                **comparison,
            }
        )

    return {
        "passed": all(sample["passed"] for sample in sample_reports),
        "h5_model": str(h5_path),
        "tflite_model": str(tflite_path),
        "max_allowed_probability_difference": max_difference,
        "runs_per_sample": runs,
        "samples": sample_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara predicciones entre H5 y TFLite"
    )
    parser.add_argument(
        "--h5-model", type=Path, default=Path("modelo_papas_finetuned.h5")
    )
    parser.add_argument(
        "--tflite-model", type=Path, default=Path("modelo_papas.tflite")
    )
    parser.add_argument(
        "--image", type=Path, action="append",
        help="Imagen real; puede repetirse para validar varias muestras",
    )
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-difference", type=float, default=0.01)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_PATH / "tflite_verification.json",
    )
    args = parser.parse_args()

    if not args.h5_model.is_file():
        raise FileNotFoundError(f"No existe el modelo H5: {args.h5_model}")
    if not args.tflite_model.is_file():
        raise FileNotFoundError(
            f"No existe el modelo TFLite: {args.tflite_model}"
        )
    if not args.image and not args.dummy:
        raise ValueError("Proporcione --image al menos una vez o use --dummy")

    report = verify_models(
        args.h5_model,
        args.tflite_model,
        args.image,
        max_difference=args.max_difference,
        runs=args.runs,
        seed=args.seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
