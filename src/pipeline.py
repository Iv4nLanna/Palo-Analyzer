import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detector import PaloDetector
from src.preprocessor import DocumentAligner
from src.scorer import compute_metrics


@dataclass
class PipelineResult:
    metrics: Dict
    line_counts: List[int]
    local_lines: List[List[Dict]]
    global_lines: List[List[Dict]]
    roi_rect: Tuple[int, int, int, int]
    aligned: np.ndarray
    roi_img: np.ndarray
    binary: np.ndarray
    overlay: np.ndarray


def parse_roi_frac(roi_text: str) -> Optional[Tuple[float, float, float, float]]:
    if not roi_text:
        return None
    parts = [p.strip() for p in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi-frac precisa de 4 valores: x1,y1,x2,y2")
    return tuple(float(p) for p in parts)


def to_global_lines(lines, offset_x, offset_y):
    global_lines = []
    for line in lines:
        g_line = []
        for p in line:
            g = dict(p)
            g["x"] = int(g["x"] + offset_x)
            g["y"] = int(g["y"] + offset_y)
            g["cx"] = float(g["cx"] + offset_x)
            g["cy"] = float(g["cy"] + offset_y)
            g_line.append(g)
        global_lines.append(g_line)
    return global_lines


def estimate_spacing_mm(lines, mm_per_px):
    gaps_mm = []
    for line in lines:
        if len(line) < 2:
            continue
        sorted_line = sorted(line, key=lambda p: p["x"])
        for i in range(1, len(sorted_line)):
            prev = sorted_line[i - 1]
            curr = sorted_line[i]
            gap_px = curr["x"] - (prev["x"] + prev["w"])
            if 0 < gap_px < 200:
                gaps_mm.append(gap_px * mm_per_px)
    return float(mean(gaps_mm)) if gaps_mm else None


def estimate_height_mm(lines, mm_per_px):
    heights = [p["h"] * mm_per_px for line in lines for p in line]
    return float(mean(heights)) if heights else None


def estimate_line_spacing_mm(lines, mm_per_px):
    if len(lines) < 2:
        return None

    line_baselines = []
    line_heights = []
    for line in lines:
        base_y = mean([p["y"] + p["h"] for p in line])
        h = mean([p["h"] for p in line])
        line_baselines.append(base_y)
        line_heights.append(h)

    gaps_mm = []
    for i in range(1, len(line_baselines)):
        raw_gap_px = line_baselines[i] - line_baselines[i - 1]
        ref_h_px = (line_heights[i] + line_heights[i - 1]) / 2.0
        clear_gap_px = raw_gap_px - ref_h_px
        if -50 < clear_gap_px < 300:
            gaps_mm.append(clear_gap_px * mm_per_px)

    return float(mean(gaps_mm)) if gaps_mm else None


def estimate_line_direction_angle_deg(lines):
    angles = []
    for line in lines:
        if len(line) < 8:
            continue
        xs = np.array([p["cx"] for p in line], dtype=np.float32)
        ys = np.array([p["y"] + p["h"] for p in line], dtype=np.float32)
        if np.std(xs) < 1e-3:
            continue
        slope = np.polyfit(xs, ys, 1)[0]
        angle = math.degrees(math.atan(float(slope)))
        angles.append(angle)

    return float(mean(angles)) if angles else None


def estimate_stroke_inclination_angle_deg(lines):
    angles = [p.get("angle_deg") for line in lines for p in line if p.get("angle_deg") is not None]
    if not angles:
        return None
    return float(mean(angles))


def estimate_margins_mm(global_lines, aligned_shape, mm_per_px):
    if not global_lines:
        return None, None, None

    h, w = aligned_shape[:2]
    xs_left = [p["x"] for line in global_lines for p in line]
    xs_right = [p["x"] + p["w"] for line in global_lines for p in line]
    ys_top = [p["y"] for line in global_lines for p in line]

    if not xs_left or not xs_right or not ys_top:
        return None, None, None

    margin_left_px = max(0.0, float(min(xs_left)))
    margin_right_px = max(0.0, float(w - max(xs_right)))
    margin_top_px = max(0.0, float(min(ys_top)))
    return margin_left_px * mm_per_px, margin_right_px * mm_per_px, margin_top_px * mm_per_px


def estimate_pressure_level(roi_img, binary, local_lines):
    if roi_img is None or binary is None:
        return ""
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    mask = binary > 0
    if int(mask.sum()) == 0:
        return ""

    darkness = float(255.0 - np.mean(gray[mask]))
    widths = [p["w"] for line in local_lines for p in line]
    mean_w = float(mean(widths)) if widths else 0.0
    if darkness > 150 or mean_w >= 3.2:
        return "forte"
    if darkness < 85 and mean_w <= 2.0:
        return "leve"
    return "media"


def estimate_stroke_quality_level(local_lines):
    palos = [p for line in local_lines for p in line]
    if not palos:
        return ""
    fill_ratios = []
    deviations = []
    for p in palos:
        wh = max(1.0, float(p["w"] * p["h"]))
        fill_ratios.append(float(p["area"]) / wh)
        if p.get("angle_deg") is not None:
            deviations.append(abs(float(p["angle_deg"]) - 90.0))
    mean_fill = float(mean(fill_ratios)) if fill_ratios else 0.0
    mean_dev = float(mean(deviations)) if deviations else 0.0

    if mean_fill < 0.33:
        return "descontinua"
    if mean_dev > 8.0:
        return "curva"
    return "reta"


def estimate_organization_level(local_lines, line_counts):
    if not local_lines:
        return ""
    counts_cv = 0.0
    if line_counts:
        avg = float(mean(line_counts))
        if avg > 0:
            counts_cv = float(np.std(line_counts) / avg)

    y_centers = [float(mean([p["cy"] for p in line])) for line in local_lines if line]
    y_gaps = [abs(y_centers[i] - y_centers[i - 1]) for i in range(1, len(y_centers))]
    gaps_cv = 0.0
    if y_gaps:
        gavg = float(mean(y_gaps))
        if gavg > 0:
            gaps_cv = float(np.std(y_gaps) / gavg)

    score = (counts_cv * 0.6) + (gaps_cv * 0.4)
    if score <= 0.06:
        return "muito boa"
    if score <= 0.12:
        return "boa"
    if score <= 0.20:
        return "regular"
    if score <= 0.30:
        return "ruim"
    return "muito ruim"


def estimate_order_pattern(local_lines):
    if not local_lines:
        return "nao_informado"
    dispersions = []
    for line in local_lines:
        if len(line) < 3:
            continue
        xs = [p["x"] for p in line]
        gaps = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
        if not gaps:
            continue
        gavg = float(mean(gaps))
        if gavg <= 0:
            continue
        dispersions.append(float(np.std(gaps) / gavg))
    if not dispersions:
        return "nao_informado"
    d = float(mean(dispersions))
    return "ordenados" if d <= 0.45 else "desordenados"


def estimate_auto_quality(aligned, roi_img, binary, local_lines, line_counts):
    flags = []
    score = 1.0

    if aligned is None or roi_img is None or binary is None:
        return {"score": 0.0, "requires_manual_review": True, "flags": ["imagem_invalida"]}

    roi_h, roi_w = roi_img.shape[:2]
    roi_area = max(1, roi_h * roi_w)
    ink_ratio = float(np.count_nonzero(binary)) / float(roi_area)

    if ink_ratio < 0.005:
        score -= 0.35
        flags.append("tracos_muito_fracos")
    elif ink_ratio > 0.18:
        score -= 0.25
        flags.append("ruido_ou_sombra_alta")

    if len(local_lines) < 3:
        score -= 0.25
        flags.append("poucas_linhas_detectadas")

    if line_counts:
        avg = float(mean(line_counts))
        if avg > 0:
            cv = float(np.std(line_counts) / avg)
            if cv > 0.55:
                score -= 0.15
                flags.append("alta_variacao_contagem_linhas")

    if any(len(line) < 8 for line in local_lines):
        score -= 0.1
        flags.append("linhas_curtas_detectadas")

    score = max(0.0, min(1.0, score))
    requires_manual_review = score < 0.7
    return {
        "score": round(score, 4),
        "requires_manual_review": requires_manual_review,
        "flags": flags,
        "ink_ratio": round(ink_ratio, 6),
    }


def draw_detection_overlay(base_img, lines, roi_rect=None):
    out = base_img.copy()
    palette = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 255, 255)]

    if roi_rect is not None:
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)

    for i, line in enumerate(lines):
        color = palette[i % len(palette)]
        for p in line:
            cv2.rectangle(out, (p["x"], p["y"]), (p["x"] + p["w"], p["y"] + p["h"]), color, 1)

        y_label = int(line[0]["y"] - 5) if line else 10
        cv2.putText(
            out,
            f"L{i + 1}: {len(line)}",
            (10, max(15, y_label)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def save_line_counts_csv(csv_path, line_counts):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["linha", "contagem"])
        for i, count in enumerate(line_counts, start=1):
            writer.writerow([i, count])


def save_outputs(output_dir, aligned, roi_img, binary, overlay, line_counts, metrics):
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(str(Path(output_dir) / "aligned.jpg"), aligned)
    cv2.imwrite(str(Path(output_dir) / "roi.jpg"), roi_img)
    cv2.imwrite(str(Path(output_dir) / "binary.jpg"), binary)
    cv2.imwrite(str(Path(output_dir) / "overlay.jpg"), overlay)

    save_line_counts_csv(Path(output_dir) / "contagem_por_linha.csv", line_counts)

    with open(Path(output_dir) / "resultado.json", "w", encoding="utf-8") as f:
        json.dump({"line_counts": line_counts, "metrics": metrics}, f, ensure_ascii=False, indent=2)


def process_image(
    image_path: str,
    errors: int = 0,
    roi_frac: Optional[Tuple[float, float, float, float]] = None,
    output_dir: Optional[str] = None,
    save_artifacts: bool = True,
    swap_lr_margins: bool = False,
) -> PipelineResult:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise RuntimeError("Nao foi possivel abrir a imagem com OpenCV")

    aligner = DocumentAligner()
    detector = PaloDetector()

    aligned = aligner.get_aligned_image(original_img)
    roi_img, roi_rect = aligner.crop_roi(aligned, roi_frac=roi_frac)
    binary = aligner.binarize(roi_img)

    detector.find_palos(binary)
    local_lines = detector.group_lines()
    line_counts = detector.get_line_counts()

    x1, y1, x2, y2 = roi_rect
    global_lines = to_global_lines(local_lines, x1, y1)
    overlay = draw_detection_overlay(aligned, global_lines, roi_rect=roi_rect)

    mm_per_px = 210.0 / float(aligned.shape[1])
    spacing_mm = estimate_spacing_mm(local_lines, mm_per_px=mm_per_px)
    height_mm = estimate_height_mm(local_lines, mm_per_px=mm_per_px)
    line_spacing_mm = estimate_line_spacing_mm(local_lines, mm_per_px=mm_per_px)
    line_direction_angle_deg = estimate_line_direction_angle_deg(local_lines)
    stroke_inclination_angle_deg = estimate_stroke_inclination_angle_deg(local_lines)
    margin_left_mm, margin_right_mm, margin_top_mm = estimate_margins_mm(global_lines, aligned.shape, mm_per_px)
    if swap_lr_margins:
        margin_left_mm, margin_right_mm = margin_right_mm, margin_left_mm
    pressure_level = estimate_pressure_level(roi_img, binary, local_lines)
    stroke_quality_level = estimate_stroke_quality_level(local_lines)
    organization_level = estimate_organization_level(local_lines, line_counts)
    order_pattern = estimate_order_pattern(local_lines)
    auto_quality = estimate_auto_quality(aligned, roi_img, binary, local_lines, line_counts)

    metrics = compute_metrics(
        line_counts=line_counts,
        error_count=errors,
        avg_spacing_mm=spacing_mm,
        avg_height_mm=height_mm,
        line_spacing_mm=line_spacing_mm,
        line_direction_angle_deg=line_direction_angle_deg,
        stroke_inclination_angle_deg=stroke_inclination_angle_deg,
        margin_left_mm=margin_left_mm,
        margin_right_mm=margin_right_mm,
        margin_top_mm=margin_top_mm,
        pressure_level=pressure_level,
        stroke_quality_level=stroke_quality_level,
        organization_level=organization_level,
        order_pattern=order_pattern,
        reasoning_level="nao_informado",
    )
    metrics["roi_rect"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    metrics["swap_lr_margins"] = bool(swap_lr_margins)
    metrics["detection_stats"] = detector.get_detection_stats()
    metrics["auto_quality"] = auto_quality

    if save_artifacts and output_dir:
        save_outputs(output_dir, aligned, roi_img, binary, overlay, line_counts, metrics)

    return PipelineResult(
        metrics=metrics,
        line_counts=line_counts,
        local_lines=local_lines,
        global_lines=global_lines,
        roi_rect=roi_rect,
        aligned=aligned,
        roi_img=roi_img,
        binary=binary,
        overlay=overlay,
    )
