# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `papa_disease/`: `data.py` prepares reproducible splits, `training.py` defines ResNet50 training, `inference.py` loads and predicts, `segmentation.py` implements HSV/Watershed, and `evaluation.py` produces metrics. Command-line entry points are in `scripts/`. Tests are in `tests/` and follow the module boundaries. `Dataset.ipynb` is the original exploratory Colab notebook. Team sequencing and acceptance criteria are documented in `docs/PLAN_PROYECTO.md`.

Large or generated artifacts are intentionally ignored: `dataset_papa/`, `.venv/`, `outputs/`, `split_manifest.json`, checkpoints, and the retrained model. Never commit these without explicit team agreement.

## Build, Test, and Development Commands

Create a local environment and install pinned dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Useful commands:

```bash
python -m unittest discover -s tests -v   # Run all unit tests
python -m compileall papa_disease scripts tests  # Check syntax
python -m scripts.prepare_split           # Create the 70/15/15 manifest
python -m scripts.train                   # Train ResNet50 (prefer Colab GPU)
python -m scripts.evaluate --split validation --threshold 0.70
python -m scripts.predict path/to/leaf.jpg
```

Run the final `test` evaluation only after selecting and freezing the confidence threshold on `validation`.

## Coding Style & Naming Conventions

Use Python 3.13, four-space indentation, type hints for public interfaces, and short docstrings describing behavior or safety constraints. Follow PEP 8 naming: `snake_case` for functions/files, `UPPER_CASE` for constants, and descriptive CLI flags such as `--output-model`. Keep heavyweight imports inside functions when modules should remain usable without TensorFlow.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Name files `test_<area>.py` and methods `test_<behavior>`. Add deterministic tests for split logic, class ordering, metrics, input validation, and failure cases. Every change must pass the full test and compile commands above.

## Commit & Pull Request Guidelines

Create one focused branch per change, such as `feat/fase-3-gradio` or `fix/model-loading`. Use Conventional Commit prefixes: `feat:`, `fix:`, `docs:`, `test:`, or `chore:`. Pull requests must explain purpose, affected workflow, validation commands, and model/data implications. Include screenshots for UI changes and metric summaries for training changes. Do not merge with failing tests.

## Security & Model Integrity

Never commit `kaggle.json`, tokens, private field images, or unreviewed model files. Preserve the class order in `papa_disease/config.py`. Treat external `.h5` files as untrusted. Report metrics with the exact model, split, threshold, and commit used.
