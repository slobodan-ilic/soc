# eoncoding: utf-8

"""Home of the module for loading Sentinel-2 images and ground-truth masks."""

import itertools
import os
import sys
from random import sample

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from utils import lazyproperty, preprocess_ms_image


class Loader:
    """Implementation for the U-Net related functionality for Sentinel segmentation."""

    def __init__(self, path, patch_size, skip):
        self._path = path
        self._patch_size = patch_size
        self._skip = skip

    # ------------------------- API --------------------------------------------------

    def img_gen(self, inds, bch_size=8):
        """return generator of training images, based on initial indexes."""
        while True:
            batch_patches = []
            batch_masks = []
            shuffled = sample(inds, bch_size)
            for index in shuffled:
                img, msk = self._preprocess_patch_index(index)
                # print(index)
                batch_patches += [img]
                batch_masks += [msk]
            X = np.array(batch_patches)
            Y = np.array(batch_masks)
            yield X, Y

    @lazyproperty
    def _master_size(self):
        """int representing dim of input."""
        # filename = f"{self._path}{self._patches[0]}/{1}-img.npy"
        filename = f"{self._path}{self._patches[0]}/img.npy"
        img = preprocess_ms_image(np.load(filename).astype("float64"))
        return img.shape[0]

    @lazyproperty
    def _patches(self) -> list:
        """list of all the sentinel patches."""
        return [psh for psh in os.listdir(self._path) if not psh.startswith(".")]

    @lazyproperty
    def _images(self) -> np.ndarray:
        filename = f"{self._path}{self._patches[0]}/img.npy"
        img = preprocess_ms_image(np.load(filename).astype("float64"))
        return img

    # # def _images(self, pch_ind, mth_ind, i, j) -> np.ndarray:
    # def _images(self, pch_ind, i, j) -> np.ndarray:
    #     # patches = []
    #     # for patch in self._patches:
    #     #     images = []
    #     #     for i in range(12):
    #     n = self._patch_size
    #     # month = mth_ind + 1
    #     patch = self._patches[pch_ind]
    #     # filename = f"{self._path}{patch}/{month}-img.npy"
    #     filename = f"{self._path}{patch}/img.npy"
    #     img = preprocess_ms_image(np.load(filename).astype("float64"))
    #     return img[i : i + n, j : j + n, :]
    #     # images.append(img)
    #     # patches.append(images)
    #     # return np.array(patches)

    @lazyproperty
    def _masks(self) -> np.ndarray:
        masks = []
        for patch in self._patches:
            filename = f"{self._path}{patch}/LULC.npy"
            logits = np.load(filename)
            one_hot = to_categorical(logits, num_classes=10)
            masks.append(one_hot)
        return np.array(masks)

    # def _masks(self, pch_ind, i, j) -> np.ndarray:
    #     # masks = []
    #     # for patch in self._patches:
    #     patch = self._patches[pch_ind]
    #     filename = f"{self._path}{patch}/LULC.npy"
    #     logits = np.load(filename)
    #     n = self._patch_size
    #     logits = logits[i : i + n, j : j + n, :]
    #     # images.append(img)
    #     one_hot = to_categorical(logits, num_classes=10)
    #     return one_hot
    #     # masks.append(one_hot)
    #     # return np.array(masks)

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
    def split_patch_indices(self) -> list:
        """generate training, validation and test splits based on patch indices.

        All possible patch indices go from the upper left corner of the master image, to
        the lower right one. The total number of possible patches along a single
        dimension of the master image is 1500 - 64 (the dimension of the master image
        minus the dimension of a patch). The total number of possible patches is the
        product of the number of possible patches along each dimension.
        """
        n_pixels_per_dim = self._master_size - self._patch_size
        all_patch_inds = list(
            itertools.product(
                range(len(self._patches)),
                # range(12),
                range(0, n_pixels_per_dim, self._skip),
                range(0, n_pixels_per_dim, self._skip),
                self._n_rotate,
                self._flip_codes,
            )
        )

        # trn, tst = train_test_split(all_patch_inds, test_size=0.3, random_state=42)
        # trn, vld = train_test_split(trn, test_size=0.2, random_state=42)
        trn, vld = train_test_split(all_patch_inds, test_size=0.2, random_state=42)

        # return trn, vld, tst
        return trn, vld

    # ------------------------- IMPLEMENTATION ---------------------------------------

    def _preprocess_patch_index(self, ind):
        n = self._patch_size
        # pch, month_ind, i, j, n_rotate, flip_code = ind
        pch, i, j, n_rotate, flip_code = ind
        # img = self._images(pch, month_ind, i, j)  # ][i : i + n, j : j + n, :]
        img = self._images[i : i + n, j : j + n, :]
        # msk = self._masks(pch, i, j)  # [pch][i : i + n, j : j + n, :]
        msk = self._masks[pch][i : i + n, j : j + n, :]
        if flip_code:
            axes = {1: 0, 2: 1, 3: (0, 1)}[flip_code]  # flip code -> axes
            img = np.flip(img, axes)
            msk = np.flip(msk, axes)
        img = np.rot90(img, n_rotate)
        msk = np.rot90(msk, n_rotate)
        return img, msk


if __name__ == "__main__":
    path = sys.argv[-3]
    patch_size = int(sys.argv[-2])
    skip = int(sys.argv[-1])
    print(f"path: {path}, ps: {patch_size}, skip: {skip}")
    loader = Loader(path, patch_size, skip)
    trn, vld = loader.split_patch_indices
    for i, el in enumerate(loader.img_gen(trn)):
        print(el)
