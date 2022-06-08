# encoding: utf-8
"""Extract land composition features from UNet."""

from data import preprocess_ms_image
import csv
import os

import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model, load_model

from sentinel_downloader import SentinelDownloader
from utils import lazyproperty


def custom_loss(y_true, y_pred):
    return categorical_crossentropy(y_true[:, :, :, 1:], y_pred[:, :, :, 1:])


DIR = "./data/features"


class FeatureExtractor:
    """Implementation of UNet feature extractor for land composition."""

    def __init__(self, filename, start="2020-06-01", end="2020-06-30"):
        self._filename = filename
        self._start = start
        self._end = end

        # --- Create models UNet and Feature Extractor ---
        path = "unet-ms-sentinel-0.0.h5"
        unet = load_model(path, custom_objects={"custom_loss": custom_loss})
        self._unet = unet
        self._feat_ext = Model(inputs=unet.inputs, outputs=unet.layers[-2].output)

    @lazyproperty
    def _coords(self) -> list:
        with open(self._filename, newline="") as csvfile:
            return [
                (lon, lat)
                for i, (lon, lat, *_) in enumerate(csv.reader(csvfile))
                if i != 0
            ]

    def _images(self):
        for lon, lat in self._coords:
            path = f"{DIR}/{lon}-{lat}-ms-img.py"
            if not os.path.exists(path):
                sd = SentinelDownloader(
                    float(lon), float(lat), "2020-06-12", "2020-07-01"
                )
                img = sd.download()
                np.save(path, img)
            else:
                img = np.load(path)
            yield lon, lat, img

    def _features(self):
        for lon, lat, img in self._images():
            input_ = np.array([preprocess_ms_image(img.astype("float32"))])
            features = self._feat_ext.predict(input_)
            yield lon, lat, features[0, 31, 31]


if __name__ == "__main__":
    filename = "/Users/slobodanilic/Downloads/barilaTexture_v2/fvgTexture.csv"
    fe = FeatureExtractor(filename)
    features_fn = filename.split(".")[0] + "Feats.csv"
    with open(features_fn, "w", newline="") as csvfile:
        featwriter = csv.writer(csvfile)
        featwriter.writerow(["lon", "lat"] + [f"feat_{i}" for i in range(10)])
        for lon, lat, feat in fe._features():
            featwriter.writerow([lon, lat] + feat.tolist())
