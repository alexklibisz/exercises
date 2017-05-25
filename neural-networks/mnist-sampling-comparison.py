'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.backend.tensorflow_backend import set_session
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
import random
import os
from itertools import cycle
from scipy.stats import linregress

# Don't use all the GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

batch_size = 100
nb_classes = 10
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# model = Sequential()
#
# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                         border_mode='valid',
#                         input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

def gen_random(X, Y, batch_size, seed = 865):

    # draw random samples from the full dataset at each iteration.

    random.seed(seed)
    combined = [(x,y) for x,y in zip(X,Y)]
    while True:
        sample = random.sample(combined, batch_size)
        _X = np.array([x for x,_ in sample])
        _Y = np.array([y for _,y in sample])
        yield (_X, _Y)

def gen_cycle(X, Y, batch_size, seed=865):

    # shuffle the dataset before beginning; draw batches sequentially over the
    # shuffled dataset.

    random.seed(seed)
    combined = [(x,y) for x,y in zip(X,Y)]
    random.shuffle(combined)
    combined = cycle(combined)
    while True:
        sample = [next(combined) for _ in range(batch_size)]
        _X = np.array([x for x, _ in sample])
        _Y = np.array([y for _, y in sample])
        yield (_X, _Y)

def gen_shuffle(X, Y, batch_size, seed=865):

    # Shuffle the dataset before each epoch, and draw batches without replacement.

    random.seed(seed)
    nb_batches_per_epoch = len(X) / batch_size
    combined = [(x, y) for x, y in zip(X, Y)]

    while True:
        random.shuffle(combined)
        combined_cycle = cycle(combined)
        for _ in range(nb_batches_per_epoch):
            sample = [next(combined_cycle) for _ in range(batch_size)]
            _X = np.array([x for x, _ in sample])
            _Y = np.array([y for _, y in sample])
            yield (_X, _Y)


tests = [
    ('Random full.', gen_random, len(X_train), nb_epoch),
    ('Cycle full.', gen_cycle, len(X_train), nb_epoch),
    ('Shuffle full.', gen_shuffle, len(X_train), nb_epoch),
    ('Random subset.', gen_random, len(X_train) / 2 , nb_epoch * 2)
]

for name, gen, samples_per_epoch, nb_epoch in tests:
    print(name, 'Training')
    result = model.fit_generator(gen(X_train,Y_train,batch_size),
        nb_epoch=nb_epoch, samples_per_epoch=samples_per_epoch, verbose=0)
    history = result.history
    Y = np.array(result.history['acc'])
    X = np.arange(len(Y))
    print(name, 'Accuracy slope: %.5lf' % linregress(X,Y).slope)
    Y = np.array(result.history['loss'])
    X = np.arange(len(Y))
    print(name, 'Loss slope: %.5lf' % linregress(X, Y).slope)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print(name, 'Test score:', score[0])
    print(name,'Test accuracy:', score[1])

# Results:
# Randomly sample batch_size samples from the full dataset at each batch.
# Random full. Accuracy slope: 0.00334
# Random full. Loss slope: -0.01104
# Random full. Test score: 0.0289998629413
# Random full. Test accuracy: 0.9903
# ...
# Shuffle the samples once at the beginning, then cycle through them.
# Cycle full. Accuracy slope: 0.00015
# Cycle full. Loss slope: -0.00049
# Cycle full. Test score: 0.0284191274357
# Cycle full. Test accuracy: 0.9905
# ...
# Shuffle the samples at the beginning of each epoch, cycle through them.
# Shuffle full. Accuracy slope: 0.00003
# Shuffle full. Loss slope: -0.00012
# Shuffle full. Test score: 0.0266968571738
# Shuffle full. Test accuracy: 0.9925
# ...
# Sample batch_size / 2 samples from the full dataset at each batch.
# Train for twice as many epochs.
# This way each epoch see's half the training set, but trains twice as long.
# Random subset. Accuracy slope: 0.00002
# Random subset. Loss slope: -0.00002
# Random subset. Test score: 0.0253595153973
# Random subset. Test accuracy: 0.9928