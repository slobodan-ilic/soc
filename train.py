# encoding: utf-8

"""Home of the U-Net model preparation and training."""

import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data import SentinelUnetLoader

sm.set_framework("tf.keras")


def create_and_train_unet_model(path, input_shape, n_classes, batch_size, epochs):
    """Create U-Net keras model in tensorflow based on the input parameters."""
    # ---Create base U-Net model without weight initialization---
    unet = sm.Unet(
        "vgg16",
        classes=n_classes,
        input_shape=input_shape,
        activation="softmax",
        encoder_weights=None,
    )
    unet.compile(loss="categorical_crossentropy", optimizer="adam")

    # ---Load Sentinel-2 data with masks, to training and validation datasets---
    loader = SentinelUnetLoader(path)
    training_indices, validation_indices, test_indices = loader.split_patch_indices
    train_dataset = loader.dataset(training_indices)
    validation_dataset = loader.dataset(validation_indices)

    # ---Prepare various callbacks---
    callbacks = [
        ModelCheckpoint("unet-sentinel-0.0.h5", verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    ]

    # ---Train the model
    train_steps = len(training_indices) // batch_size
    valid_steps = len(validation_indices) // batch_size
    unet.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=validation_dataset,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks,
    )
    return unet


if __name__ == "__main__":
    path = "./sentinel-data/"
    unet = create_and_train_unet_model(
        path, input_shape=(64, 64, 13), n_classes=10, batch_size=8, epochs=2
    )
    print(unet.summary())
