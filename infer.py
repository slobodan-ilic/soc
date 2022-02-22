# encoding: utf-8

"""Home of the code for inference of Sentinel-2 MS images with U-Net"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from data import preprocess_ms_image


class Infererer:
    def __init__(self, path):
        self._unet = load_model(path)

    def infer(self, img):
        input_ = np.array([preprocess_ms_image(img.astype("float32"))])
        return self._unet.predict(input_)[0]

    def save(self, img, classes):
        # ---Save True-Color image
        tci = np.array(img[:, :, 1:4] * 3.5 / 1e4 * 255, dtype="uint8")
        cv2.imwrite("tci.png", tci)

        # ---Save the mask
        categories = np.argmax(classes, axis=2)
        msk = self._create_mask(categories)
        cv2.imwrite("msk.png", msk)

    def _create_mask(self, labels) -> np.array:
        codes = {
            0: (255, 255, 255),  # ------------ "No Data", "#ffffff"
            1: (255, 255, 0),  # -------------- "Cultivated Land", "#ffff00"
            2: (5, 73, 7),  # ----------------- "Forest", "#054907"
            3: (255, 165, 0),  # -------------- "Grassland", "#ffa500"
            4: (128, 96, 0),  # --------------- "Shrubland", "#806000"
            5: (6, 154, 243),  # -------------- "Water", "#069af3"
            6: (149, 208, 252),  # ------------ "Wetlands", "#95d0fc"
            7: (150, 123, 182),  # ------------ "Tundra", "#967bb6"
            8: (220, 20, 60),  # -------------- "Artificial Surface", "#dc143c"
            9: (166, 166, 166),  # ------------ "Bareland", "#a6a6a6"
            10: (0, 0, 0),  # ----------------- "Snow and Ice", "#000000"
        }
        mask = np.full((64, 64, 3), (0, 0, 0), np.uint8)
        for lbl in codes:
            mask[labels == lbl] = codes[lbl][::-1]
        return mask


if __name__ == "__main__":
    img_path = "./ns.npy"  # Saved image that you get from Sentinel
    master_img = np.load(img_path)

    # ---Position of the patch in the (larger) image from Sentinel---
    i = 0
    j = 0
    img = master_img[i : i + 64, j : j + 64, :]

    # ---Perform prediction for the given patch---
    inferer = Infererer("unet-ms-sentinel-0.0.h5")
    classes = inferer.infer(img)
    inferer.save(img, classes)
