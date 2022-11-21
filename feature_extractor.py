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


# def custom_loss(y_true, y_pred):
    # return categorical_crossentropy(y_true[:, :, :, 1:], y_pred[:, :, :, 1:])

def custom_loss(y_true, y_pred):
    pad = 1
    return categorical_crossentropy(
        y_true[:, pad:-pad, pad:-pad, 1:], y_pred[:, pad:-pad, pad:-pad, 1:]
    )

DIR = "./features"


class FeatureExtractor:
    """Implementation of UNet feature extractor for land composition."""

    pixel = 32 // 2 - 1
    layers_back = -16

    def __init__(self, filename, start="2020-06-01", end="2020-06-30"):
        self._filename = filename
        self._start = start
        self._end = end

        # --- Create models UNet and Feature Extractor ---
        path = "unet-ms-sentinel-0.0.h5"
        unet = load_model(path, custom_objects={"custom_loss": custom_loss})
        self._unet = unet
        self._feat_ext = Model(inputs=unet.inputs, outputs=unet.layers[self.layers_back].output)
        print(self._unet.summary())

    @lazyproperty
    def _coords(self) -> list:
        with open(self._filename, newline="") as csvfile:
            return [
                (lon, lat)
                for i, (lon, lat, *_) in enumerate(csv.reader(csvfile))
                if i != 0
            ]

    def _images(self):
        for i, (lon, lat) in enumerate(self._coords):
            print(f"image: {i}")
            path = f"{DIR}/{lon}-{lat}-ms-img"
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
        for i, (lon, lat, img) in enumerate(self._images()):
            print(f"featue: {i}")
            input_ = np.array([preprocess_ms_image(img.astype("float32"))])
            features = self._feat_ext.predict(input_)
            yield lon, lat, features[0, self.pixel, self.pixel]

    @lazyproperty
    def _n_feats(self):
        n_feats = self._feat_ext.layers[-1].output_shape[-1]
        print(f"NFEATS: {n_feats}")
        return n_feats


if __name__ == "__main__":
    filename = f"{DIR}/lucas2018_EDIT.csv"
    fe = FeatureExtractor(filename)
    features_fn = ".".join(filename.split(".")[:-1]) + "Feats.csv"
    with open(features_fn, "w", newline="") as csvfile:
        featwriter = csv.writer(csvfile)
        featwriter.writerow(["lon", "lat"] + [f"feat_{i}" for i in range(fe._n_feats)])
        for lon, lat, feat in fe._features():
            featwriter.writerow([lon, lat] + feat.tolist())
