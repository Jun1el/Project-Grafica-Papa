"""Predice una imagen desde la linea de comandos."""

import argparse
import json
from pathlib import Path

from papa_disease.inference import load_image, load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasifica una hoja de papa")
    parser.add_argument("image", type=Path, help="Imagen JPG o PNG")
    args = parser.parse_args()

    result = predict(load_model(), load_image(args.image))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

