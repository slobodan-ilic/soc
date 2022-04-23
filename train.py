# encoding: utf-8

"""Home of the U-Net model preparation and training."""

import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data import SentinelUnetLoader
from helpers import NpuHelperForTF

sm.set_framework("tf.keras")


def create_and_train_unet_model(path, input_shape, n_classes, batch_size, epochs):
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
    # ---Replace first layer of the pretrained network to match MS with 13 channels---
    # for i in range(len(unet_rgb.layers)):
    for i in range(3, 20):
        unet_ms.layers[i].set_weights(unet_rgb.layers[i].get_weights())
    unet_ms.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_accuracy"],
    )

    # ---Load Sentinel-2 data with masks, to training and validation datasets---
    loader = SentinelUnetLoader(path)
    training_indices, validation_indices, test_indices = loader.split_indices
    train_gen = loader.img_gen(training_indices)
    validation_gen = loader.img_gen(validation_indices)

    # ---Prepare various callbacks---
    callbacks = [
        ModelCheckpoint("unet-ms-sentinel-0.0.h5", verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    ]

    # ---Train the model
    train_steps = len(training_indices) // batch_size
    valid_steps = len(validation_indices) // batch_size
    for i in range(3, 20):
        unet_ms.layers[i].trainable = False

    unet_ms.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=validation_gen,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks,
    )
    for i in range(len(unet_rgb.layers)):
        unet_ms.layers[i].trainable = True
    unet_ms.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=validation_gen,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks,
    )

    for i in range(3, 30):
        unet_ms.layers[i].trainable = True
    unet_ms.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_accuracy"],
    )
    for i in range(len(unet_rgb.layers)):
        unet_ms.layers[i].trainable = True
    unet_ms.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=validation_gen,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks,
    )
    sess.close()


if __name__ == "__main__":
    path = "./data/"
    size = 64

    npu_config = {
        "device_id": "0",
        "rank_id": "0",
        "rank_size": "0",
        "job_id": "10385",
        "rank_table_file": "",
    }

    print("________________________ INIT NPU SESSION ________________________")
    sess = NpuHelperForTF(**npu_config).sess()
    unet = create_and_train_unet_model(
        path, input_shape=(size, size, 13), n_classes=10, batch_size=256, epochs=100
    )
    sess.close()
