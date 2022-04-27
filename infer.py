# encoding: utf-8

"""Home of the code for inference of Sentinel-2 MS images with U-Net"""

import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from data import preprocess_ms_image
from helpers.sentinel import SentinelHelper
from utils import lazyproperty

try:
    from helpers.npu import NpuHelperForTF as NH

    npu_config = {
        "device_id": "0",
        "rank_id": "0",
        "rank_size": "0",
        "job_id": "10385",
        "rank_table_file": "",
    }

    print("************************ INIT NPU SESSION *********************************")
    sess = NH(**npu_config).sess() if NH else None
except ModuleNotFoundError:
    sess = None


class Infererer:
    def __init__(self, path):
        self._unet = load_model(path)

    @lazyproperty
    def n(self):
        return self._unet.input_shape[-2]

    def infer(self, img):
        input_ = np.array([preprocess_ms_image(img.astype("float32"))])
        return self._unet.predict(input_)[0]


def create_mask(classes) -> np.array:
    """Create color mask in RGB based on mask classes."""
    codes = {
        0: (255, 255, 255),  # ------- Nothing
        1: (0, 255, 255),  # --------- Cultivated
        2: (7, 73, 5),  # ------------ Forest
        3: (0, 165, 255),  # --------- Grassland
        4: (0, 96, 128),  # ---------- Shrubland
        5: (243, 154, 6),  # --------- Water
        6: (252, 208, 149),  # ------- Wetlands
        7: (182, 123, 150),  # ------- Tundra
        8: (60, 20, 220),  # --------- Artificial Surface
        9: (166, 166, 166),  # ------- Bareland
    }

    mask = np.full(classes.shape + (3,), (0, 0, 0), np.uint8)
    for class_ in codes:
        mask[classes == class_] = codes[class_]
    return mask


def infer_lulc(image, pad):
    """Infer LULC and save image as 'fused.png' in the local directory."""
    inferer = Infererer("unet-ms-sentinel-0.0.h5")

    h, w, _ = image.shape
    dim = inferer.n
    span = dim - 2 * pad
    nrows = (h - 2 * pad) // span
    ncols = (w - 2 * pad) // span

    row_patches = []
    for i in range(nrows):
        col_patches = []
        for j in range(ncols):
            img_patch = image[i * span : i * span + dim, j * span : j * span + dim, :]
            pred = inferer.infer(img_patch)
            classes = np.argmax(pred, axis=2)[pad:(-pad), pad:(-pad)]
            col_patches.append(classes)
        row_patch = np.hstack(col_patches)
        row_patches.append(row_patch)

    mask = create_mask(np.vstack(row_patches))
    img = image[pad : pad + nrows * span, pad : pad + ncols * span, :]
    tc_image = np.array(img[:, :, 1:4] * 3.5 / 1e4 * 255, dtype="uint8")
    fused = cv2.addWeighted(tc_image, 0.6, mask, 0.4, 0)
    cv2.imwrite("fused.png", fused)


if __name__ == "__main__":
    pad = int(sys.argv[-1])

    bbox = [19.820823, 45.268260, 19.847773, 45.284206]
    # bbox = [19.5, 45.09, 20.31, 45.5]
    sh = SentinelHelper(bbox, 10, 2021)
    infer_lulc(sh.image, pad)

    if sess is not None:
        print("************************ CLOSE NPU SESSION ****************************")
        sess.close()
