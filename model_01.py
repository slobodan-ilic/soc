from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_vgg16_unet(input_shape, num_classes):
    """Input"""
    inputs = Input(input_shape)

    """ Pre-trained VGG16 Model """
    # vgg16_img_net = VGG16(
    #      include_top=False, weights="imagenet", input_shape=(64, 64, 3)
    # )
    # model_path = "../tutorials/sentinel/data/models/vgg_ms_transfer_final.44-0.969.hdf5"
    # vgg16_img_net = load_model(model_path)
    vgg16 = VGG16(include_top=False, weights=None, input_tensor=inputs)

    """ Encoder """
    # names = [
    #     "block1_conv2",
    #     "block1_pool",
    #     "block2_conv1",
    #     "block2_conv2",
    #     "block2_pool",
    #     "block3_conv1",
    #     "block3_conv2",
    #     "block3_conv3",
    #     "block3_pool",
    #     "block4_conv1",
    #     "block4_conv2",
    #     "block4_conv3",
    #     "block4_pool",
    #     "block5_conv1",
    #     "block5_conv2",
    #     "block5_conv3",
    # ]
    # for name in names:
    #     source_layer = vgg16_img_net.get_layer(name)
    #     target_layer = vgg16.get_layer(name)
    #     target_layer.set_weights(source_layer.get_weights())

    s1 = vgg16.get_layer("block1_conv2").output  # (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output  # (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output  # (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output  # (64 x 64)

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output  # (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  # (64 x 64)
    d2 = decoder_block(d1, s3, 256)  # (128 x 128)
    d3 = decoder_block(d2, s2, 128)  # (256 x 256)
    d4 = decoder_block(d3, s1, 64)  # (512 x 512)

    """ Output """
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model


if __name__ == "__main__":
    input_shape = (512, 512, 10)
    num_classes = 10
    model = build_vgg16_unet(input_shape, num_classes)
    model.summary()
