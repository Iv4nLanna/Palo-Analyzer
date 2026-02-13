import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_accuracy import _acc as benchmark_acc  # reaproveita regra de acuracia
from src.ml_models import FEATURE_NAMES, TARGET_MAP, fuse_ml_with_rules, predict_ml_classes, train_ml_models
from src.pipeline import process_image


TARGET_COLUMNS = list(TARGET_MAP.keys())


@dataclass
class ExampleResult:
    image_path: str
    total: int
    linhas: int
    score_final: float
    auto_quality: float


def find_real_images() -> List[Path]:
    preferred = [
        Path(r"C:\Users\Ivan\Downloads\7aa30a5c-f71b-499e-ac5a-914d9f3f6e4f.jpg"),
        Path(r"C:\Users\Ivan\Downloads\41315360-2456-453a-8292-09fdfaa4c0e0.jpg"),
        Path(r"C:\Users\Ivan\Downloads\99f07814-ffca-4b7e-b4a2-c7dbbc842f6d.jpg"),
        Path(r"C:\Users\Ivan\Downloads\09ded9d2-febc-42f4-aeaa-e2f2adf4446a.jpeg"),
        Path(r"C:\Users\Ivan\Downloads\69676982-c132-47ff-84bb-625332e0b16c.jpeg"),
    ]
    existing = [p for p in preferred if p.exists()]
    if existing:
        return existing

    fallback: List[Path] = []
    downloads = Path.home() / "Downloads"
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        fallback.extend(downloads.glob(ext))
    return sorted(fallback, key=lambda p: p.stat().st_mtime, reverse=True)[:5]


def augment_image(img: np.ndarray, variant_id: int) -> np.ndarray:
    h, w = img.shape[:2]

    if variant_id == 0:
        out = img.copy()
    elif variant_id == 1:
        out = cv2.convertScaleAbs(img, alpha=1.08, beta=8)
    elif variant_id == 2:
        out = cv2.convertScaleAbs(img, alpha=0.92, beta=-6)
    elif variant_id == 3:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 1.7, 1.0)
        out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    elif variant_id == 4:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -1.7, 1.0)
        out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    elif variant_id == 5:
        out = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        out = cv2.medianBlur(img, 3)

    return out


def extract_target(metrics: Dict, target_col: str) -> str:
    class_key, is_dict = TARGET_MAP[target_col]
    value = metrics.get("classificacoes", {}).get(class_key)
    if is_dict:
        if isinstance(value, dict):
            return str(value.get("nivel", "")).strip()
        return ""
    if value is None:
        return ""
    return str(value).strip()


def generate_examples(base_images: List[Path], out_root: Path) -> Tuple[List[Dict], List[ExampleResult]]:
    aug_dir = out_root / "augmented_images"
    aug_dir.mkdir(parents=True, exist_ok=True)

    label_rows: List[Dict] = []
    summary_rows: List[ExampleResult] = []

    for base_idx, base_path in enumerate(base_images):
        img = cv2.imread(str(base_path))
        if img is None:
            continue

        for variant in range(7):
            aug = augment_image(img, variant)
            out_img = aug_dir / f"real_{base_idx + 1:02d}_v{variant}.jpg"
            cv2.imwrite(str(out_img), aug)

            result = process_image(str(out_img), output_dir=None, save_artifacts=False)
            m = result.metrics

            row = {
                "image_path": str(out_img),
                "errors": 0,
            }
            for tc in TARGET_COLUMNS:
                row[tc] = extract_target(m, tc)
            label_rows.append(row)

            summary_rows.append(
                ExampleResult(
                    image_path=str(out_img),
                    total=int(m.get("total", 0) or 0),
                    linhas=int(m.get("linhas", 0) or 0),
                    score_final=float(m.get("score_final", 0.0) or 0.0),
                    auto_quality=float(m.get("auto_quality", {}).get("score", 0.0) or 0.0),
                )
            )

    return label_rows, summary_rows


def write_labels_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["image_path", "errors", *TARGET_COLUMNS]
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


def build_dataset_from_labels(labels_rows: List[Dict], dataset_path: Path) -> None:
    out_rows = []
    for row in labels_rows:
        image_path = row["image_path"]
        result = process_image(image_path, errors=int(row.get("errors", 0) or 0), output_dir=None, save_artifacts=False)
        m = result.metrics
        rec = {"image_path": image_path}
        for k in FEATURE_NAMES:
            rec[k] = m.get(k)
        for tc in TARGET_COLUMNS:
            rec[tc] = row.get(tc, "")
        out_rows.append(rec)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["image_path", *FEATURE_NAMES, *TARGET_COLUMNS])
        wr.writeheader()
        wr.writerows(out_rows)


def run_local_benchmark(labels_rows: List[Dict], model_path: str, threshold: float = 0.75) -> Dict:
    from src.ml_models import load_ml_model

    ml_payload = load_ml_model(model_path)
    modes = ["rules", "ml_assist", "ml_hybrid", "ml_override"]

    truths = {t: [] for t in TARGET_COLUMNS}
    preds = {m: {t: [] for t in TARGET_COLUMNS} for m in modes}

    for row in labels_rows:
        result = process_image(row["image_path"], errors=int(row.get("errors", 0) or 0), output_dir=None, save_artifacts=False)
        base = result.metrics
        ml = predict_ml_classes(base, ml_payload)

        assist = json.loads(json.dumps(base))
        assist = fuse_ml_with_rules(assist, ml, mode="assist", confidence_threshold=threshold)
        hybrid = json.loads(json.dumps(base))
        hybrid = fuse_ml_with_rules(hybrid, ml, mode="hybrid", confidence_threshold=threshold)
        override = json.loads(json.dumps(base))
        override = fuse_ml_with_rules(override, ml, mode="override", confidence_threshold=threshold)

        for tc in TARGET_COLUMNS:
            truths[tc].append(row.get(tc, ""))

            ckey, is_dict = TARGET_MAP[tc]
            raw = base.get("classificacoes", {}).get(ckey)
            if is_dict and isinstance(raw, dict):
                preds["rules"][tc].append(raw.get("nivel"))
            else:
                preds["rules"][tc].append(raw if not is_dict else "")

            for name, metrics in [("ml_assist", assist), ("ml_hybrid", hybrid), ("ml_override", override)]:
                curr = metrics.get("classificacoes", {}).get(ckey)
                if is_dict and isinstance(curr, dict):
                    preds[name][tc].append(curr.get("nivel"))
                else:
                    preds[name][tc].append(curr if not is_dict else "")

    per_target = {}
    overall = {m: [] for m in modes}

    for tc in TARGET_COLUMNS:
        acc_map = {}
        for mode in modes:
            acc = benchmark_acc(truths[tc], preds[mode][tc])
            acc_map[mode] = None if acc is None else round(float(acc), 4)
            if acc is not None:
                overall[mode].append(float(acc))
        per_target[tc] = acc_map

    overall_mean = {
        m: (round(float(sum(vals) / len(vals)), 4) if vals else None)
        for m, vals in overall.items()
    }
    best_mode = max(overall_mean, key=lambda k: overall_mean[k] if overall_mean[k] is not None else -1)

    return {
        "samples": len(labels_rows),
        "per_target_accuracy": per_target,
        "overall_accuracy_mean": overall_mean,
        "best_mode": best_mode,
        "threshold": threshold,
        "note": "Benchmark sobre labels bootstrap (pseudo-rotulos gerados por regras).",
    }


def write_example_summary(path: Path, rows: List[ExampleResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_path", "total", "linhas", "score_final", "auto_quality"])
        for r in rows:
            wr.writerow([r.image_path, r.total, r.linhas, r.score_final, r.auto_quality])


def write_markdown_report(path: Path, base_images: List[Path], summary_rows: List[ExampleResult], train_report: Dict, bench_report: Dict) -> None:
    totals = [r.total for r in summary_rows]
    linhas = [r.linhas for r in summary_rows]
    aq = [r.auto_quality for r in summary_rows]

    lines = [
        "# Processo Completo - Exemplos Reais",
        "",
        "## Fontes reais usadas",
    ]
    for p in base_images:
        lines.append(f"- `{p}`")

    lines.extend(
        [
            "",
            "## Resumo da execução",
            f"- Amostras processadas (inclui variacoes): {len(summary_rows)}",
            f"- Total de palos (min/med/max): {min(totals)}/{int(np.median(totals))}/{max(totals)}",
            f"- Linhas detectadas (min/med/max): {min(linhas)}/{int(np.median(linhas))}/{max(linhas)}",
            f"- Auto quality media: {round(float(sum(aq) / len(aq)), 4) if aq else 0.0}",
            "",
            "## Treino ML",
            f"- Modelo: `output/ml_models_real_examples.pkl`",
            f"- Relatorio: `{json.dumps(train_report, ensure_ascii=False)}`",
            "",
            "## Benchmark",
            f"- Resultado geral: `{json.dumps(bench_report.get('overall_accuracy_mean', {}), ensure_ascii=False)}`",
            f"- Melhor modo: `{bench_report.get('best_mode')}`",
            f"- Observacao: {bench_report.get('note')}",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_root = ROOT / "output" / "real_examples_run"
    out_root.mkdir(parents=True, exist_ok=True)

    base_images = find_real_images()
    if len(base_images) < 2:
        raise RuntimeError("Nao encontrei imagens reais suficientes para montar exemplos.")

    label_rows, summary_rows = generate_examples(base_images, out_root)
    if len(label_rows) < 20:
        raise RuntimeError("Poucas amostras geradas para treino ML (minimo recomendado: 20).")

    labels_csv = ROOT / "input" / "ml_labels_real_examples_bootstrap.csv"
    dataset_csv = ROOT / "output" / "ml_dataset_real_examples.csv"
    model_path = ROOT / "output" / "ml_models_real_examples.pkl"
    train_report_json = ROOT / "output" / "ml_train_report_real_examples.json"
    benchmark_json = ROOT / "output" / "benchmark_real_examples.json"
    summary_csv = out_root / "summary_examples.csv"
    report_md = ROOT / "output" / "RELATORIO_PROCESSO_COMPLETO_REAIS.md"

    write_labels_csv(labels_csv, label_rows)
    build_dataset_from_labels(label_rows, dataset_csv)
    train_out = train_ml_models(str(dataset_csv), str(model_path))
    train_report_json.write_text(json.dumps(train_out.report, ensure_ascii=False, indent=2), encoding="utf-8")

    bench = run_local_benchmark(label_rows, str(model_path), threshold=0.75)
    benchmark_json.write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")

    write_example_summary(summary_csv, summary_rows)
    write_markdown_report(report_md, base_images, summary_rows, train_out.report, bench)

    print("Processo completo finalizado.")
    print(f"Labels bootstrap: {labels_csv}")
    print(f"Dataset ML: {dataset_csv}")
    print(f"Modelo ML: {model_path}")
    print(f"Benchmark: {benchmark_json}")
    print(f"Relatorio final: {report_md}")


if __name__ == "__main__":
    main()
