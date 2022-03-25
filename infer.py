# encoding: utf-8

"""Home of the code for inference of Sentinel-2 MS images with U-Net"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from data import preprocess_ms_image
from helpers import SentinelHelper


class Infererer:
    def __init__(self, path):
        self._unet = load_model(path)

    def infer(self, img):
        input_ = np.array([preprocess_ms_image(img.astype("float32"))])
        return self._unet.predict(input_)[0]


def create_mask(classes) -> np.array:
    """Create color mask in RGB based on mask classes."""
    codes = {
        0: (147, 221, 187),  # ---- Annual Crop
        1: (62, 157, 87),  # ------ Forest
        2: (44, 93, 51),  # ------- Herbaceous Vegetation
        3: (148, 150, 151),  # ---- Highway
        4: (156, 159, 235),  # ---- Industrial
        5: (124, 192, 241),  # ---- Pasture
        6: (79, 235, 247),  # ----- Permanent Crop
        7: (44, 52, 208),  # ------ Residential
        8: (175, 120, 60),  # ----- River
        9: (224, 205, 173),  # ---- SeaLake
    }
    mask = np.full(classes.shape + (3,), (0, 0, 0), np.uint8)
    for class_ in codes:
        mask[classes == class_] = codes[class_]
    return mask


def infer_lulc(image):
    """Infer LULC and save image as 'fused.png' in the local directory."""
    inferer = Infererer("unet-ms-sentinel-0.0.h5")

    h, w, _ = image.shape
    m = h // 64
    n = w // 64

    row_patches = []
    for i in range(m):
        col_patches = []
        for j in range(n):
            img_patch = image[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, :]
            pred = inferer.infer(img_patch)
            classes = np.argmax(pred, axis=2)
            col_patches.append(classes)
        row_patch = np.hstack(col_patches)
        row_patches.append(row_patch)

    mask = create_mask(np.vstack(row_patches))
    img = image[0 : m * 64, 0 : n * 64, :]
    tc_image = np.array(img[:, :, 1:4] * 3.5 / 1e4 * 255, dtype="uint8")
    fused = cv2.addWeighted(tc_image, 0.6, mask, 0.4, 0)
    cv2.imwrite("fused.png", fused)


if __name__ == "__main__":
    bbox = [19.820823, 45.268260, 19.847773, 45.284206]
    sh = SentinelHelper(bbox, 3, 2021)
    infer_lulc(sh.image())
