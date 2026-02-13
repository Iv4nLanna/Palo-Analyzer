import argparse
import csv
import sys
from pathlib import Path

# Permite executar via "python src/build_ml_dataset.py".
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_models import FEATURE_NAMES
from src.pipeline import parse_roi_frac, process_image


def parse_args():
    p = argparse.ArgumentParser(description="Gera dataset de features para treino ML")
    p.add_argument("--input", required=True, help="CSV de entrada com image_path e targets")
    p.add_argument("--output", default="output/ml_dataset.csv", help="CSV de saida com features + targets")
    p.add_argument("--roi-frac", default="", help="ROI em fracoes x1,y1,x2,y2")
    return p.parse_args()


def main():
    args = parse_args()
    roi_frac = parse_roi_frac(args.roi_frac) if args.roi_frac else None

    with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("CSV de entrada vazio")

    target_cols = [c for c in rows[0].keys() if c.startswith("target_")]
    out_rows = []

    for row in rows:
        image_path = row.get("image_path", "").strip()
        if not image_path:
            continue

        result = process_image(
            image_path=image_path,
            errors=int(row.get("errors", 0) or 0),
            roi_frac=roi_frac,
            output_dir=None,
            save_artifacts=False,
        )
        m = result.metrics
        rec = {k: m.get(k) for k in FEATURE_NAMES}
        rec["image_path"] = image_path
        for tc in target_cols:
            rec[tc] = row.get(tc, "")
        out_rows.append(rec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", *FEATURE_NAMES, *target_cols])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Dataset ML gerado: {out_path}")
    print(f"Amostras: {len(out_rows)}")


if __name__ == "__main__":
    main()
