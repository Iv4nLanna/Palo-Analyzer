import cv2
import math
import numpy as np

from config import (
    HORIZONTAL_LINE_KERNEL_WIDTH,
    LINE_TOLERANCE_Y,
    MAX_AREA,
    MAX_WIDTH,
    MIN_AREA,
    MIN_ASPECT_RATIO,
    MIN_HEIGHT,
    MIN_PALOS_PER_LINE,
    VERTICAL_KERNEL_HEIGHT,
)


class PaloDetector:
    def __init__(self):
        self.palos = []
        self.lines = []

    def _filter_vertical_strokes(self, binary_img):
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, VERTICAL_KERNEL_HEIGHT))
        vertical = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vert_kernel)

        # Remove linhas horizontais de formulario para reduzir falso positivo.
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZONTAL_LINE_KERNEL_WIDTH, 1))
        horizontal = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, horiz_kernel)
        cleaned = cv2.subtract(vertical, horizontal)

        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        return cleaned

    def find_palos(self, binary_img):
        processed = self._filter_vertical_strokes(binary_img)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
        detected = []

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            ratio = (h / float(w)) if w > 0 else 0.0

            if area < MIN_AREA or area > MAX_AREA:
                continue
            if h < MIN_HEIGHT:
                continue
            if w > MAX_WIDTH:
                continue
            if ratio < MIN_ASPECT_RATIO:
                continue

            # Orientacao em graus no eixo X (0..180), util para inclinacao dos palos.
            comp_mask = (labels == i).astype(np.uint8)
            m = cv2.moments(comp_mask, binaryImage=True)
            angle_deg = 90.0
            den = (m["mu20"] - m["mu02"])
            num = (2.0 * m["mu11"])
            if abs(den) > 1e-9 or abs(num) > 1e-9:
                theta = 0.5 * math.atan2(num, den)
                angle_deg = math.degrees(theta)
                if angle_deg < 0:
                    angle_deg += 180.0

            detected.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": area,
                    "cx": x + (w / 2.0),
                    "cy": y + (h / 2.0),
                    "angle_deg": float(angle_deg),
                }
            )

        detected.sort(key=lambda p: (p["cy"], p["cx"]))
        self.palos = detected
        return detected

    def group_lines(self):
        if not self.palos:
            self.lines = []
            return []

        sorted_palos = sorted(self.palos, key=lambda p: p["cy"])
        median_h = float(np.median([p["h"] for p in sorted_palos])) if sorted_palos else 0.0
        threshold = max(float(LINE_TOLERANCE_Y), median_h * 0.75)

        bands = []
        for palo in sorted_palos:
            placed = False
            for band in bands:
                if abs(palo["cy"] - band["center_y"]) <= threshold:
                    band["items"].append(palo)
                    band["center_y"] = float(np.mean([p["cy"] for p in band["items"]]))
                    placed = True
                    break

            if not placed:
                bands.append({"center_y": palo["cy"], "items": [palo]})

        # Ordena bandas por posicao vertical e filtra ruido.
        bands.sort(key=lambda b: b["center_y"])

        filtered_lines = []
        for band in bands:
            line = sorted(band["items"], key=lambda p: p["x"])
            if len(line) >= MIN_PALOS_PER_LINE:
                filtered_lines.append(line)

        self.lines = filtered_lines
        return filtered_lines

    def get_line_counts(self):
        return [len(line) for line in self.lines]

    def get_detection_stats(self):
        return {
            "palos_detectados": len(self.palos),
            "linhas_detectadas": len(self.lines),
            "media_altura": round(float(np.mean([p["h"] for p in self.palos])), 4) if self.palos else 0.0,
            "media_largura": round(float(np.mean([p["w"] for p in self.palos])), 4) if self.palos else 0.0,
            "media_angulo_palos": round(float(np.mean([p["angle_deg"] for p in self.palos])), 4) if self.palos else None,
        }
