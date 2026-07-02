"""Evalua la robustez del modelo sobre un conjunto de imagenes de campo.

A diferencia de ``scripts.evaluate``, no usa el manifiesto 70/15/15: recorre un
directorio con subcarpetas por clase (por ejemplo ``dataset_campo/``) y compara
el desempeno frente al dataset de laboratorio. Watershed queda desactivado por
defecto porque ya se demostro que degrada las metricas; puede habilitarse para
documentar su efecto sobre fondos de campo.
"""

import argparse
from pathlib import Path

from papa_disease.config import (
    FIELD_DATASET_PATH,
    REPORTS_PATH,
    RETRAINED_MODEL_PATH,
    UNCERTAINTY_THRESHOLD,
)
from papa_disease.evaluation import (
    evaluate_field_directory,
    save_confusion_matrix,
    save_report,
)
from papa_disease.inference import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el modelo con imagenes de campo")
    parser.add_argument("--model", type=Path, default=RETRAINED_MODEL_PATH)
    parser.add_argument("--field-dir", type=Path, default=FIELD_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_PATH)
    parser.add_argument("--threshold", type=float, default=UNCERTAINTY_THRESHOLD)
    parser.add_argument("--with-watershed", action="store_true")
    args = parser.parse_args()

    report = evaluate_field_directory(
        load_model(args.model),
        args.field_dir,
        compare_watershed=args.with_watershed,
        uncertainty_threshold=args.threshold,
    )
    report_path = args.output_dir / "field_report.json"
    save_report(report, report_path)
    save_confusion_matrix(report, args.output_dir / "field_raw.png")
    if "watershed" in report:
        save_confusion_matrix(
            report, args.output_dir / "field_watershed.png", "watershed"
        )

    raw = report["raw"]
    print(f"Reporte guardado en {report_path}")
    print(f"Muestras: {raw['samples']}")
    print(f"Clases presentes: {raw['present_classes']}")
    if raw["missing_classes"]:
        print(f"Clases ausentes (soporte 0): {raw['missing_classes']}")
    print(f"Accuracy de campo: {raw['accuracy']:.4f}")
    print(f"Macro F1 (clases presentes): {raw['macro_f1_present']:.4f}")
    print(f"Cobertura sobre umbral {args.threshold}: {raw['coverage']:.4f}")
    if "watershed" in report:
        print(f"Accuracy con Watershed: {report['watershed']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
