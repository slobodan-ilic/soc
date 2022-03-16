# encoding: utf-8

"""Home of the U-Net model preparation and training."""
import tensorflow as tf

from tensorboard import summary

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data import SentinelUnetLoader

from model_01 import build_vgg16_unet

from helpers import NpuHelperForTF


def create_and_train_unet_model(path, input_shape, n_classes, batch_size, epochs):
    """Create U-Net keras model in tensorflow based on the input parameters."""
    # ---Create base U-Net model without weight initialization---

    print("------------------ BUILDING MODEL ------------------------")

    #unet_ms=build_vgg16_unet(input_shape, n_classes)

    ###unet_APU=build_unet(input_shape, n_classes)

    unet_APU2=build_vgg16_unet(input_shape, n_classes)
    #unet_ms_sloba = sm.Unet(
     #   "vgg16",
      #  classes=n_classes,
       # input_shape=input_shape,
        #activation="softmax",
        #encoder_weights=None,
    #)
    print(unet_APU2.summary())

    #print("-------------------!----------------------")

    #unet_ms.summary()

    #print("----------------------------------3----------------------")


    # ---Replace first layer of the pretrained network to match MS with 13 channels---
    # for i in range(len(unet_rgb.layers)):
    #for i in range(3, 20):
        #unet_ms.layers[i].set_weights(unet_rgb.layers[i].get_weights())

    npu_config = {
    "device_id": "0",
    "rank_id": "0",
    "rank_size": "0",
    "job_id": "10385",
    "rank_table_file": "",
    }

    print("________________________ INIT NPU SESSION ________________________")
    
    sess = NpuHelperForTF(**npu_config).sess()

    print("--------------------- compiling")

    unet_APU2.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_accuracy"],
    )
    print("--------------------- loading")

    # ---Load Sentinel-2 data with masks, to training and validation datasets---
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


    #for i in range(3, 20):
      #  unet_ms.layers[i].trainable = False
    #for i in range(3):
     #   unet_ms.layers[i].trainable = True
    print("--------------------- fitting")

    unet_APU2.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=validation_gen,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks,
    )
    print("--------------------------------done")
    # for i in range(len(unet_rgb.layers)):
     #   unet_ms.layers[i].trainable = True
    #unet_ms.fit(
     #   train_gen,
      #  steps_per_epoch=train_steps,
       # validation_data=validation_gen,
       # validation_steps=valid_steps,
       # epochs=epochs,
       # callbacks=callbacks,
    #) """

    print("________________________ CLOSE NPU SESSION _______________________")
    sess.close()


if __name__ == "__main__":
    path = "./sentinel-data/"
    unet = create_and_train_unet_model(path, input_shape=(64, 64, 13), n_classes=10, batch_size=8, epochs=20)
    # print(unet.summary())
