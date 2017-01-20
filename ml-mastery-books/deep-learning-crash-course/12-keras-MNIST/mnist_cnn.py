# Basically just copying the mnist_cnn.py from the examples in the Keras repo.
# This took about 3.5 hours to run on my i7-5500U laptop. I imagine it would
# be much faster on a GPU or if I knew how to parallelize it.

from __future__ import print_function
import numpy as np
np.random.seed(1337)
import argparse
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=False, help='file with weights')
args = vars(ap.parse_args())

batch_size = 128 # 128 digits at a time
nb_classes = 10  # (10 different digits)
nb_epoch = 12    # 12 epochs

img_rows, img_cols = 28, 28    # input image dimensions
nb_filters = 32                # number of convolutional filters to use
nb_pool = 2                    # size of pooling area for max pooling
nb_conv = 3                    # cnvolution kernel size

# load data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Print some info.
print()
print('X_train shape:', X_train.shape)
print('Y_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
# to_categorical maps all values in y_train to 10 classes
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)
print(Y_train.shape[0], 'train labels')
print(Y_test.shape[0], 'test labels')

# Define the model
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Compile the model
print('compiling...')
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Fit the data
if args["weights"] is not None:
  print("loading weights...")
  model.load_weights(args["weights"])
else:
  print('training...')
  checkpt = ModelCheckpoint('weights.best.h5', monitor='acc', save_best_only=True, mode='max')
  model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpt])

# Score the model and print results
print('evaluating...')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

