"""Evalua el modelo y compara entradas originales con Watershed."""

import argparse
from pathlib import Path

from papa_disease.config import (
    REPORTS_PATH,
    RETRAINED_MODEL_PATH,
    SPLIT_MANIFEST_PATH,
    UNCERTAINTY_THRESHOLD,
)
from papa_disease.evaluation import (
    evaluate_manifest,
    save_confusion_matrix,
    save_report,
)
from papa_disease.inference import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el clasificador")
    parser.add_argument("--model", type=Path, default=RETRAINED_MODEL_PATH)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_PATH)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=UNCERTAINTY_THRESHOLD)
    parser.add_argument("--skip-watershed", action="store_true")
    args = parser.parse_args()

    report = evaluate_manifest(
        load_model(args.model),
        args.manifest,
        split=args.split,
        compare_watershed=not args.skip_watershed,
        uncertainty_threshold=args.threshold,
    )
    report_path = args.output_dir / f"{args.split}_report.json"
    save_report(report, report_path)
    save_confusion_matrix(report, args.output_dir / f"{args.split}_raw.png")
    if "watershed" in report:
        save_confusion_matrix(
            report, args.output_dir / f"{args.split}_watershed.png", "watershed"
        )
    print(f"Reporte guardado en {report_path}")
    print(f"Accuracy sin segmentacion: {report['raw']['accuracy']:.4f}")
    if "watershed" in report:
        print(f"Accuracy con Watershed: {report['watershed']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
