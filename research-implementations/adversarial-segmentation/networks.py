from keras.layers import Input, Conv2D, MaxPooling2D,\
    Conv2DTranspose, Dropout, concatenate, BatchNormalization,\
    Lambda, UpSampling2D, Activation, Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
import numpy as np


def set_trainable(net, val):
    for l in net.layers:
        l.trainable = val


def get_trainable_count(net):
    """https://stackoverflow.com/questions/45046525"""
    return sum([K.count_params(p) for p in set(net.trainable_weights)])


def get_flat_weights(net):
    return np.concatenate([w.flatten() for w in net.get_weights()])


def UNet(io_shape, output_name='seg'):

    def conv_layer(nb_filters, x):
        x = Conv2D(nb_filters, (3, 3), strides=(1, 1),
                   padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1)(x)
        return LeakyReLU(0.2)(x)

    nfb = 32

    x = inputs = Input(io_shape)
    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    dc_0_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    dc_1_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    dc_2_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    dc_3_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 16, x)
    x = conv_layer(nfb * 16, x)
    x = UpSampling2D()(x)

    x = concatenate([x, dc_3_out], axis=-1)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    x = UpSampling2D()(x)

    x = concatenate([x, dc_2_out], axis=-1)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    x = UpSampling2D()(x)

    x = concatenate([x, dc_1_out], axis=-1)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    x = UpSampling2D()(x)

    x = concatenate([x, dc_0_out], axis=-1)
    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    x = Conv2D(1, 1)(x)
    x = Activation('sigmoid', name=output_name)(x)

    return Model(inputs=inputs, outputs=x)


def ConvNetClassifier(input_shape):
    """Binary real/fake classifier. Basically just the downward pass of UNet
    with a logistic regression classifier replacing the upward pass."""

    def conv_layer(nb_filters, x):
        x = Conv2D(nb_filters, (3, 3), strides=(1, 1),
                   padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        return LeakyReLU(0.2)(x)

    nfb = 32

    x = inputs = Input(input_shape)
    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 16, x)
    x = conv_layer(nfb * 16, x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=inputs, outputs=x)
