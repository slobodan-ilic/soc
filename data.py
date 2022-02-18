# eoncoding: utf-8

"""Home of the module for loading Sentinel-2 images and ground-truth masks."""

import itertools

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class SentinelUnetLoader:
    """Implementation for the U-Net related functionality for Sentinel segmentation."""

    master_size = 1500  # Size of master image and mask in pixels per side
    limit_size = 500  # Set to master size for training on all data
    patch_size = 64  # Size of patch used for network training and prediction in pps
    step = 30  # Determines overlap between patches (1 - max overlap, 64 - no overlap)

    def __init__(self, path):
        self._path = path
        n = self.limit_size
        self._master_image = self._preprocess_ms_image(
            np.load(path + "image.npy").astype("float32")[:n, :n, :]
        )
        self._master_mask = to_categorical(
            np.load(path + "mask.npy")[:n, :n, :], num_classes=10
        )

    # ------------------------- API --------------------------------------------------

    def dataset(self, inds, batch=8):
        """return dataset for passed indices.

        The dataset will fetch each smaller patch (64 x 64)
        from the master patch (1500 x 1500).
        """
        ds = tf.data.Dataset.from_tensor_slices(inds)
        ds = ds.shuffle(buffer_size=5000)
        ds = ds.map(self._preprocess_sentinel)
        ds = ds.batch(batch)
        ds = ds.repeat()
        ds = ds.prefetch(2)
        return ds

    @property
    def split_patch_indices(self) -> tuple:
        """generate training, validation and test splits based on patch indices.

        All possible patch indices go from the upper left corner of the master image, to
        the lower right one. The total number of possible patches along a single
        dimension of the master image is 1500 - 64 (the dimension of the master image
        minus the dimension of a patch). The total number of possible patches is the
        product of the number of possible patches along each dimension.
        """
        n, step = self.limit_size - self.patch_size, self.step
        one_dim_inds = list(range(0, n, step))
        unrav = list(zip(*[el for el in itertools.product(one_dim_inds, one_dim_inds)]))
        all_patch_indices = np.ravel_multi_index(unrav, (n, n))

        trn, tst = train_test_split(all_patch_indices, test_size=0.3, random_state=42)
        trn, vld = train_test_split(trn, test_size=0.2, random_state=42)

        return trn, vld, tst

    # ------------------------- IMPLEMENTATION ---------------------------------------

    def _preprocess_sentinel(self, ind) -> tuple:
        """return tuple with data for the HS image and the respective mask."""

        def f(ind):
            n = self.patch_size
            stride = self.limit_size - n  # How many patches can be create from stripe
            i, j = np.unravel_index(ind, (stride, stride))
            img = self._master_image[i : i + n, j : j + n, :]
            msk = self._master_mask[i : i + n, j : j + n, :]
            return img, msk

        img, msk = tf.numpy_function(f, [ind], [tf.float32, tf.float32])
        return img, msk

    @staticmethod
    def _preprocess_ms_image(x):
        # define mean and std values
        mean = [
            1353.036,
            1116.468,
            1041.475,
            945.344,
            1198.498,
            2004.878,
            2376.699,
            2303.738,
            732.957,
            12.092,
            1818.820,
            1116.271,
            2602.579,
        ]
        std = [
            65.479,
            154.008,
            187.997,
            278.508,
            228.122,
            356.598,
            456.035,
            531.570,
            98.947,
            1.188,
            378.993,
            303.851,
            503.181,
        ]
        # loop over image channels
        for idx, mean_value in enumerate(mean):
            x[..., idx] -= mean_value
            x[..., idx] /= std[idx]
        return x


if __name__ == "__main__":
    path = "./sentinel-data/"
    loader = SentinelUnetLoader(path)
    trn, vld, tst = loader.split_patch_indices
    print(f"DS: Train: {len(trn)} - Valid: {len(vld)} - Test: {len(tst)}")
    for (img, msk), i in zip(loader.dataset(trn), range(3)):
        print(img.shape, msk.shape)
