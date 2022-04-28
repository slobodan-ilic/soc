# encoding: utf-8

"""Home of the U-Net model preparation and training."""

import sys

import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

from data import Loader

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

sm.set_framework("tf.keras")


def custom_loss(y_true, y_pred):
    return categorical_crossentropy(y_true[:, :, :, 1:], y_pred[:, :, :, 1:])


def create_and_train_unet_model(path, input_shape, skip, n_classes, batch_size, epochs):
    """Create U-Net keras model in tensorflow based on the input parameters."""
    # ---Create base U-Net model without weight initialization---
    pretrained_shape = input_shape[:2] + (3,)
    unet_rgb = sm.Unet(
        "vgg16",
        classes=n_classes,
        input_shape=pretrained_shape,
        activation="softmax",
        encoder_weights="imagenet",
    )
    unet_ms = sm.Unet(
        "vgg16",
        classes=n_classes,
        input_shape=input_shape,
        activation="softmax",
        encoder_weights=None,
    )

    # ---Load Sentinel-2 data with masks, to training and validation datasets---
    loader = Loader(path, input_shape[0], skip)
    training_indices, validation_indices = loader.split_patch_indices
    train_gen = loader.img_gen(training_indices)
    validation_gen = loader.img_gen(validation_indices)

    # ---Replace first layer of the pretrained network to match MS with 13 channels---
    for i in range(2, 24):
        unet_ms.layers[i].set_weights(unet_rgb.layers[i].get_weights())
    for i in range(2, 24):
        unet_ms.layers[i].trainable = False

    # ---Prepare various callbacks---
    callbacks = [
        ModelCheckpoint("unet-ms-sentinel-0.0.h5", verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    ]

    # ---Train the model
    train_steps = len(training_indices) // batch_size
    valid_steps = len(validation_indices) // batch_size
    unet_ms.compile(
        # loss="categorical_crossentropy",
        loss=custom_loss,
        optimizer="adam",
        metrics=["categorical_accuracy"],
    )
    unet_ms.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=validation_gen,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks,
    )

    # ---Prepare for "tightening"---
    for i in range(len(unet_rgb.layers)):
        unet_ms.layers[i].trainable = True
    for i in range(2, 8):
        unet_ms.layers[i].trainable = False
    unet_ms.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    unet_ms.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=validation_gen,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    path = sys.argv[-4]
    patch_size = int(sys.argv[-3])
    skip = int(sys.argv[-2])
    batch = int(sys.argv[-1])

    print("************************ CREAATE AND TRAIN ********************************")
    unet = create_and_train_unet_model(
        path,
        input_shape=(patch_size, patch_size, 13),
        skip=skip,
        n_classes=10,
        batch_size=batch,
        epochs=100,
    )

    if sess is not None:
        print("************************ CLOSE NPU SESSION ****************************")
        sess.close()
