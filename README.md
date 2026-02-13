# Palo Analyzer

[![CI](https://github.com/Iv4nLanna/Palo-Analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/Iv4nLanna/Palo-Analyzer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Desktop + CLI system in Python to automate **Palographic Test** analysis using classical Computer Vision, auditable business rules, and optional ML assistance.

## Why this project matters
- Reduces manual correction effort with a reproducible pipeline.
- Keeps analysis auditable (rule IDs and structured outputs in JSON).
- Uses a hybrid strategy: **manual input has priority** over automatic extraction for safety and practical reliability.

## Technical Highlights
- Perspective correction (homography) and ROI cropping.
- Adaptive binarization and morphology-based stroke detection.
- Stroke grouping into lines and extraction of geometric features.
- Rule engine for psychometric dimensions (productivity, rhythm/NOR, spacing, margins, inclination, organization, etc.).
- Optional ML fusion modes (`assist`, `hybrid`, `override`) for classification support.
- Desktop app for non-technical users (`desktop_app.py`).

## Validation Snapshot
Based on local benchmark artifacts (`output/benchmark_real_examples.json`) generated from real examples + controlled augmentations:
- Samples: `35`
- Mean accuracy by mode:
  - `rules`: `1.0000`
  - `ml_assist`: `1.0000`
  - `ml_hybrid`: `0.9893`
  - `ml_override`: `0.9571`
- Best mode in this run: `rules`

Important: this specific benchmark is pseudo-labeled and evaluates internal consistency, not full clinical validity.

## Project Structure
```text
Palo-Analyzer/
  main.py                         # CLI entrypoint
  desktop_app.py                  # Desktop UI (Tkinter)
  config.py                       # Pipeline and detection parameters
  src/
    pipeline.py                   # CV pipeline + metric extraction
    preprocessor.py               # Homography / ROI / binarization
    detector.py                   # Stroke detection and line grouping
    scorer.py                     # Rule engine and interpretations
    ml_models.py                  # ML training/prediction/fusion
    build_ml_dataset.py           # Build feature dataset
    train_ml_models.py            # Train ML models
    benchmark_accuracy.py         # Compare rules vs ML modes
    run_full_process_examples.py  # End-to-end local run with real examples
  tests/                          # Automated tests
```

## Installation
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start
### CLI
```powershell
python main.py --image "C:\path\to\sheet.jpg" --output-dir output
```

With ML and mirrored margin swap:
```powershell
python main.py --image "C:\path\to\sheet.jpg" --ml-model "output\ml_models_real_examples.pkl" --ml-mode hybrid --ml-threshold 0.75 --swap-lr-margins
```

### Desktop
```powershell
python desktop_app.py
```

## Main Outputs
- `output/resultado.json` (CLI automatic flow)
- `output/analise_completa.json` (desktop hybrid flow)
- `output/overlay.jpg`, `output/aligned.jpg`, `output/roi.jpg`, `output/binary.jpg`
- `output/contagem_por_linha.csv`

## ML Workflow (Optional)
```powershell
python src/build_ml_dataset.py --input input/ml_labels_template.csv --output output/ml_dataset.csv
python src/train_ml_models.py --dataset output/ml_dataset.csv --model output/ml_models.pkl --report output/ml_train_report.json
python src/benchmark_accuracy.py --labels input/ml_labels_template.csv --ml-model output/ml_models.pkl --output output/benchmark_report.json
```

End-to-end automated local process:
```powershell
python src/run_full_process_examples.py
```

## Testing and CI
Run local tests:
```powershell
pytest -q
```

CI is configured with GitHub Actions in `.github/workflows/ci.yml`.

## Packaging for Windows
```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --windowed --name "AnalyzerPalo" --add-data "input;input" --add-data "output;output" desktop_app.py
```

Generated app:
`dist/AnalyzerPalo/AnalyzerPalo.exe`

## Responsible Use
This project is a technical decision-support tool. Final psychological interpretation should be reviewed by a qualified professional.

## Author
Ivan Lana
