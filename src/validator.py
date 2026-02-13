import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

# Permite executar via "python src/validator.py".
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import parse_roi_frac, process_image


def parse_args():
    parser = argparse.ArgumentParser(description="Valida pipeline do palografico contra gabarito humano")
    parser.add_argument("--ground-truth", required=True, help="CSV com gabarito")
    parser.add_argument("--output", default="output/validation_report.json", help="JSON de saida")
    parser.add_argument("--roi-frac", default="", help="Override de ROI em fracoes: x1,y1,x2,y2")
    return parser.parse_args()


def parse_line_counts(text):
    if not text:
        return []
    return [int(x.strip()) for x in text.split(";") if x.strip()]


def line_mae(gt, pred):
    n = max(len(gt), len(pred))
    if n == 0:
        return 0.0
    gt_pad = gt + [0] * (n - len(gt))
    pred_pad = pred + [0] * (n - len(pred))
    return float(mean([abs(a - b) for a, b in zip(gt_pad, pred_pad)]))


def main():
    args = parse_args()
    roi_frac = parse_roi_frac(args.roi_frac)

    rows = []
    with open(args.ground_truth, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError("CSV de gabarito vazio")

    per_image = []
    abs_errors = []
    pct_errors = []
    exact_total_matches = 0
    line_maes = []

    for row in rows:
        image_path = row["image_path"]
        total_gt = int(row["total_gt"])
        errors = int(row.get("errors", 0) or 0)
        gt_line_counts = parse_line_counts(row.get("line_counts_gt", ""))

        result = process_image(
            image_path=image_path,
            errors=errors,
            roi_frac=roi_frac,
            output_dir=None,
            save_artifacts=False,
        )
        total_pred = int(result.metrics["total"])
        pred_line_counts = result.line_counts

        ae = abs(total_pred - total_gt)
        abs_errors.append(ae)
        pct_errors.append((ae / total_gt) * 100.0 if total_gt > 0 else 0.0)
        if total_pred == total_gt:
            exact_total_matches += 1

        lm = line_mae(gt_line_counts, pred_line_counts) if gt_line_counts else None
        if lm is not None:
            line_maes.append(lm)

        per_image.append(
            {
                "image_path": image_path,
                "total_gt": total_gt,
                "total_pred": total_pred,
                "abs_error_total": ae,
                "pct_error_total": round((ae / total_gt) * 100.0, 4) if total_gt > 0 else 0.0,
                "line_mae": round(lm, 4) if lm is not None else None,
                "metrics_pred": result.metrics,
                "line_counts_pred": pred_line_counts,
            }
        )

    summary = {
        "num_images": len(rows),
        "mae_total": round(float(mean(abs_errors)), 4),
        "mape_total_percent": round(float(mean(pct_errors)), 4),
        "exact_match_total_percent": round((exact_total_matches / len(rows)) * 100.0, 4),
        "line_mae": round(float(mean(line_maes)), 4) if line_maes else None,
    }

    report = {
        "summary": summary,
        "per_image": per_image,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Validacao concluida")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
