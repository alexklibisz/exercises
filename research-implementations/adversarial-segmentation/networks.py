from keras.layers import Input, Conv2D, MaxPooling2D,\
    Conv2DTranspose, Dropout, concatenate, BatchNormalization,\
    Lambda, UpSampling2D, Activation, Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K


def UNet(io_shape, nb_classes, output_name='seg'):

    def conv_layer(nb_filters, x):
        x = Conv2D(nb_filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1)(x)
        return Activation('relu')(x)

    nfb = 32
    drp = 0.1

    x = inputs = Input(io_shape)
    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    dc_0_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    x = Dropout(drp)(x)
    dc_1_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    x = Dropout(drp * 2)(x)
    dc_2_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    x = Dropout(drp * 2)(x)
    dc_3_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 16, x)
    x = conv_layer(nfb * 16, x)
    x = UpSampling2D()(x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_3_out], axis=-1)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    x = UpSampling2D()(x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_2_out], axis=-1)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    x = UpSampling2D()(x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_1_out], axis=-1)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    x = UpSampling2D()(x)
    x = Dropout(drp)(x)

    x = concatenate([x, dc_0_out], axis=-1)
    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    x = Conv2D(nb_classes, 1)(x)
    x = Activation('softmax', name=output_name)(x)

    return Model(inputs=inputs, outputs=x)


def ConvNetClassifier(input_shape, output_name='adv'):
    """GAN-specific tricks https://github.com/soumith/ganhacks"""

    def conv_layer(nb_filters, x):
        x = Conv2D(nb_filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        return Activation('relu')(x)

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
    x = Dense(2)(x)
    x = Activation('softmax', name=output_name)(x)
    return Model(inputs=inputs, outputs=x)


# def ConvNetClassifier(input_shape, output_name='adv'):
#     """GAN-specific tricks https://github.com/soumith/ganhacks"""
#
#     kreg = l2(0.01)
#     x = inputs = Input(input_shape)
#
#     x = Conv2D(64, 3, strides=1, padding='same', kernel_regularizer=kreg)(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D()(x)
#
#     x = Conv2D(128, 3, strides=1, padding='same', kernel_regularizer=kreg)(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D()(x)
#
#     x = Conv2D(256, 3, strides=1, padding='same', kernel_regularizer=kreg)(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D()(x)
#
#     x = Conv2D(512, 3, strides=1, padding='same', kernel_regularizer=kreg)(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D()(x)
#
#     x = Flatten()(x)
#     x = Dense(2)(x)
#     x = Activation('softmax', name=output_name)(x)
#
#     return Model(inputs=inputs, outputs=x)
