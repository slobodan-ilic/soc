# eoncoding: utf-8

"""Home of the module for loading Sentinel-2 images and ground-truth masks."""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class SentinelUnetLoader:
    """Implementation for the U-Net related functionality for Sentinel segmentation."""

    master_size = 1500
    limit_size = 100  # Set to master size for training in production
    patch_size = 64

    def __init__(self, path):
        self._path = path
        n = self.limit_size
        self._master_image = np.load(path + "image.npy")[:n, :n, :]
        self._master_mask = np.load(path + "mask.npy")[:n, :n, :]

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
        all_patch_indices = [i for i in range((self.limit_size - self.patch_size) ** 2)]

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

        img, msk = tf.numpy_function(f, [ind], [tf.uint16, tf.uint8])
        return img, msk


if __name__ == "__main__":
    path = "./sentinel-data/"
    loader = SentinelUnetLoader(path)
    trn, vld, tst = loader.split_patch_indices
    print(f"DS: Train: {len(trn)} - Valid: {len(vld)} - Test: {len(tst)}")
    for (img, msk), i in zip(loader.dataset(trn), range(3)):
        print(img.shape, msk.shape)
