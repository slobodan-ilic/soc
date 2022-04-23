# encoding: utf-8

"""Home of the U-Net model preparation and training."""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data import SentinelUnetLoader
from model_01 import build_vgg16_unet

try:
    from helpers import NpuHelperForTF as NH

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


def create_and_train_unet_model(path, input_shape, n_classes, batch_size, epochs):
    """Create U-Net keras model in tensorflow based on the input parameters."""
    # ---Create base U-Net model without weight initialization---

    print("------------------ BUILDING MODEL ------------------------")

    unet_APU2 = build_vgg16_unet(input_shape, n_classes)
    print(unet_APU2.summary())

    print("--------------------- compiling")
    unet_APU2.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_accuracy"],
    )

    # ---Load Sentinel-2 data with masks, to training and validation datasets---
    print("--------------------- loading")
    loader = SentinelUnetLoader(path)
    training_indices, validation_indices, test_indices = loader.split_patch_indices
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

    print("--------------------- fitting")
    unet_APU2.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=validation_gen,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    path = "./data/"
    unet = create_and_train_unet_model(
        path, input_shape=(64, 64, 13), n_classes=10, batch_size=8, epochs=20
    )

    if sess is not None:
        print("************************ CLOSE NPU SESSION ****************************")
        sess.close()
