# eoncoding: utf-8

"""Home of the module for loading Sentinel-2 images and ground-truth masks."""

import itertools
import os
from random import sample

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from utils import lazyproperty, preprocess_ms_image


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

    @lazyproperty
    def _patches(self) -> list:
        """list of all the sentinel patches."""
        return [psh for psh in os.listdir(self._path) if not psh.startswith(".")]

    @lazyproperty
    def _images(self) -> np.ndarray:
        images = []
        for patch in self._patches:
            filename = f"{self._path}{patch}/img.npy"
            img = preprocess_ms_image(np.load(filename).astype("float64"))
            images.append(img)
        return np.array(images)

    @lazyproperty
    def _masks(self) -> np.ndarray:
        masks = []
        for patch in self._patches:
            filename = f"{self._path}{patch}/LULC.npy"
            msk = to_categorical(np.load(filename), num_classes=10)
            masks.append(msk)
        return np.array(masks)

    @lazyproperty
    def _n_rotate(self) -> list:
        """N of 90 degree rotations."""
        return [0, 1, 2, 3]

    @lazyproperty
    def _flip_codes(self) -> list:
        """List of flip codes."""
        return [
            0,  # don't flip
            1,  # flip 1st axis
            2,  # flip 2nd axis
            3,  # flip both axes
        ]

    @lazyproperty
    def split_patch_indices(self) -> tuple:
        """generate training, validation and test splits based on patch indices.

        All possible patch indices go from the upper left corner of the master image, to
        the lower right one. The total number of possible patches along a single
        dimension of the master image is 1500 - 64 (the dimension of the master image
        minus the dimension of a patch). The total number of possible patches is the
        product of the number of possible patches along each dimension.
        """
        n_pixels_per_dim = self.master_size - self.patch_size
        all_patch_inds = list(
            itertools.product(
                range(len(self._patches)),
                range(0, n_pixels_per_dim, self.skip),
                range(0, n_pixels_per_dim, self.skip),
                self._n_rotate,
                self._flip_codes,
            )
        )

        trn, tst = train_test_split(all_patch_inds, test_size=0.3, random_state=42)
        trn, vld = train_test_split(trn, test_size=0.2, random_state=42)

        return trn, vld, tst

    # ------------------------- IMPLEMENTATION ---------------------------------------

    def _preprocess_index(self, ind):
        n = self.patch_size
        pch, i, j, rotate, flip = ind
        img = self._images[pch][i : i + n, j : j + n, :]
        msk = self._masks[pch][i : i + n, j : j + n, :]
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
    trn, vld, tst = loader.split_patch_indices
    print(len(trn))
    for i, el in enumerate(loader.img_gen(trn)):
        print(i)
