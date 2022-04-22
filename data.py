# eoncoding: utf-8

"""Home of the module for loading Sentinel-2 images and ground-truth masks."""

import itertools
import os
from random import sample

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def preprocess_ms_image(x):
    # define mean and std values
    mean = [
        1353.036,
        1116.468,
        1041.475,
        945.344,
        1198.498,
        2004.878,
        2376.699,
        2303.738,  # 8
        732.957,
        12.092,
        1818.820,
        1116.271,
        2602.579,  # 8A
    ]
    std = [
        65.479,
        154.008,
        187.997,
        278.508,
        228.122,
        356.598,
        456.035,
        531.570,  # 8
        98.947,
        1.188,
        378.993,
        303.851,
        503.181,  # 8A
    ]
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x


class SentinelUnetLoader:
    """Implementation for the U-Net related functionality for Sentinel segmentation."""

    master_size = 500  # Size of master image and mask in pixels per side
    patch_size = 64  # Size of patch used for network training and prediction in pps
    skip = 64  # Determines overlap between patches (1 - max overlap, 64 - no overlap)

    def __init__(self, path):
        self._path = path

    # ------------------------- API --------------------------------------------------

    def img_gen(self, inds, bch_size=8):
        """return generator of training images, based on initial indexes."""
        while True:
            batch_patches = []
            batch_masks = []
            shuffled = sample(inds, bch_size)
            for index in shuffled:
                img, msk = self._preprocess_index(index)
                # print(index)
                batch_patches += [img]
                batch_masks += [msk]
            X = np.array(batch_patches)
            Y = np.array(batch_masks)
            yield X, Y

    @property
    def _patches(self) -> list:
        return [psh for psh in os.listdir(self._path) if not psh.startswith(".")]

    @property
    def split_indices(self) -> tuple:
        """generate training, validation and test splits based on patch indices.

        All possible patch indices go from the upper left corner of the master image, to
        the lower right one. The total number of possible patches along a single
        dimension of the master image is 1500 - 64 (the dimension of the master image
        minus the dimension of a patch). The total number of possible patches is the
        product of the number of possible patches along each dimension.
        """
        n_patches = len(self._patches)
        n_pixels_per_dim, skip = self.master_size - self.patch_size, self.skip
        patch_inds = list(range(n_patches))
        one_dim_inds = list(range(0, n_pixels_per_dim, skip))
        n_rotate = [
            0,
            1,
            2,
            3,
        ]  # N of 90 degree rotations
        flip_codes = [
            0,  # don't flip
            1,  # flip 1st axis
            2,  # flip 2nd axis
            3,  # flip both axes
        ]  # Codes for flip axes
        all_patch_inds = list(
            itertools.product(
                patch_inds, one_dim_inds, one_dim_inds, n_rotate, flip_codes
            )
        )

        trn, tst = train_test_split(all_patch_inds, test_size=0.3, random_state=42)
        trn, vld = train_test_split(trn, test_size=0.2, random_state=42)

        return trn, vld, tst

    # ------------------------- IMPLEMENTATION ---------------------------------------

    def _preprocess_index(self, ind):
        n = self.patch_size
        pch, i, j, rotate, flip = ind
        patch = self._path + self._patches[pch]
        img_filename = patch + "/img.npy"
        msk_filename = patch + "/LULC.npy"
        img_pch = preprocess_ms_image(np.load(img_filename).astype("float64"))
        lulc = np.load(msk_filename)
        msk_pch = to_categorical(lulc, num_classes=10)
        img = img_pch[i : i + n, j : j + n, :]
        msk = msk_pch[i : i + n, j : j + n, :]
        if flip:
            axes = {1: 0, 2: 1, 3: (0, 1)}[flip]  # flip code -> axes
            img = np.flip(img, axes)
            msk = np.flip(msk, axes)
        img = np.rot90(img)
        msk = np.rot90(msk)
        return img, msk


if __name__ == "__main__":
    path = "./data/"
    loader = SentinelUnetLoader(path)
    trn, vld, tst = loader.split_indices
    print(len(trn))
    for i, el in enumerate(loader.img_gen(trn)):
        print(i)
