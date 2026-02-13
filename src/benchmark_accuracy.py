import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_models import TARGET_MAP, fuse_ml_with_rules, load_ml_model, predict_ml_classes
from src.pipeline import parse_roi_frac, process_image


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark de acuracia: regras x ML")
    p.add_argument("--labels", required=True, help="CSV com image_path e colunas target_*")
    p.add_argument("--ml-model", default="", help="Modelo ML .pkl (opcional)")
    p.add_argument("--roi-frac", default="", help="ROI em fracoes")
    p.add_argument("--output", default="output/benchmark_report.json", help="JSON de saida")
    p.add_argument("--ml-threshold", type=float, default=0.75, help="Threshold para modo hybrid")
    return p.parse_args()


def _extract_pred(metrics, target_col):
    target_info = TARGET_MAP.get(target_col)
    if not target_info:
        return None
    class_key, is_dict = target_info
    c = metrics.get("classificacoes", {}).get(class_key)
    if is_dict:
        if isinstance(c, dict):
            return c.get("nivel")
        return None
    return c


def _acc(trues, preds):
    valid = [(t, p) for t, p in zip(trues, preds) if t and p]
    if not valid:
        return None
    ok = sum(1 for t, p in valid if t == p)
    return ok / len(valid)


def main():
    args = parse_args()
    roi_frac = parse_roi_frac(args.roi_frac) if args.roi_frac else None

    with open(args.labels, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("CSV de labels vazio")

    target_cols = [c for c in rows[0].keys() if c.startswith("target_")]

    ml_payload = load_ml_model(args.ml_model) if args.ml_model else None
    modes = ["rules"]
    if ml_payload:
        modes.extend(["ml_assist", "ml_hybrid", "ml_override"])

    mode_preds = {m: {t: [] for t in target_cols} for m in modes}
    truths = {t: [] for t in target_cols}

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
        base = result.metrics

        for t in target_cols:
            truths[t].append(row.get(t, "").strip())
            mode_preds["rules"][t].append(_extract_pred(base, t))

        if ml_payload:
            preds = predict_ml_classes(base, ml_payload)

            assist = json.loads(json.dumps(base))
            assist = fuse_ml_with_rules(assist, preds, mode="assist", confidence_threshold=args.ml_threshold)
            hybrid = json.loads(json.dumps(base))
            hybrid = fuse_ml_with_rules(hybrid, preds, mode="hybrid", confidence_threshold=args.ml_threshold)
            override = json.loads(json.dumps(base))
            override = fuse_ml_with_rules(override, preds, mode="override", confidence_threshold=args.ml_threshold)

            for t in target_cols:
                mode_preds["ml_assist"][t].append(_extract_pred(assist, t))
                mode_preds["ml_hybrid"][t].append(_extract_pred(hybrid, t))
                mode_preds["ml_override"][t].append(_extract_pred(override, t))

    report = {"summary": {}, "per_target": {}}

    for t in target_cols:
        per_mode = {}
        for m in modes:
            per_mode[m] = _acc(truths[t], mode_preds[m][t])
        best_mode = max(per_mode, key=lambda k: per_mode[k] if per_mode[k] is not None else -1)
        report["per_target"][t] = {
            "accuracy": {k: (round(v, 4) if v is not None else None) for k, v in per_mode.items()},
            "best_mode": best_mode,
        }

    overall = {}
    for m in modes:
        vals = [report["per_target"][t]["accuracy"][m] for t in target_cols if report["per_target"][t]["accuracy"][m] is not None]
        overall[m] = round(mean(vals), 4) if vals else None

    best_overall = max(overall, key=lambda k: overall[k] if overall[k] is not None else -1)
    report["summary"] = {
        "num_images": len(rows),
        "modes": modes,
        "overall_accuracy_mean": overall,
        "best_overall_mode": best_overall,
        "ml_threshold": args.ml_threshold,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Benchmark concluido")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
