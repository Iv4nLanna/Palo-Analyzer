import cv2
import numpy as np

from config import (
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    ROI_X1,
    ROI_X2,
    ROI_Y1,
    ROI_Y2,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)


class DocumentAligner:
    def __init__(self, debug=False):
        self.debug = debug

    @staticmethod
    def _order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _clamp_frac(v):
        return max(0.0, min(1.0, float(v)))

    def _find_document_contour(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        img_area = image.shape[0] * image.shape[1]
        candidates = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

        for c in candidates:
            area = cv2.contourArea(c)
            if area < img_area * 0.25:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)

        return None

    def _warp_to_target(self, image, points):
        rect = self._order_points(points)
        dst = np.array(
            [[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (TARGET_WIDTH, TARGET_HEIGHT))

    def get_aligned_image(self, image):
        contour = self._find_document_contour(image)
        if contour is None:
            return cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
        return self._warp_to_target(image, contour)

    def get_roi_rect(self, image_shape, roi_frac=None):
        h, w = image_shape[:2]
        if roi_frac is None:
            x1f, y1f, x2f, y2f = ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
        else:
            x1f, y1f, x2f, y2f = roi_frac

        x1f = self._clamp_frac(x1f)
        y1f = self._clamp_frac(y1f)
        x2f = self._clamp_frac(x2f)
        y2f = self._clamp_frac(y2f)

        if x2f <= x1f:
            x1f, x2f = 0.0, 1.0
        if y2f <= y1f:
            y1f, y2f = 0.0, 1.0

        x1 = int(round(w * x1f))
        y1 = int(round(h * y1f))
        x2 = int(round(w * x2f))
        y2 = int(round(h * y2f))

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return (x1, y1, x2, y2)

    def crop_roi(self, image, roi_frac=None):
        x1, y1, x2, y2 = self.get_roi_rect(image.shape, roi_frac=roi_frac)
        return image[y1:y2, x1:x2], (x1, y1, x2, y2)

    def to_grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        return clahe.apply(gray)

    def binarize(self, aligned_image):
        gray = self.to_grayscale(aligned_image)

        # Pequena remocao de ruido mantendo bordas.
        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35)

        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C,
        )
        return binary
