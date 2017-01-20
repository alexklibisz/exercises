from keras.datasets import mnist
import numpy as np
import os

def get_mnist_data():
	# Use keras dataset instead.
	# conda install keras
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# Reshape data once for training, again for test.
	# X_train 50Kx28x28 matrix -> 50Kx784x1 matrix
	# y_train 10Kx1 array -> 10Kx10x1 matrix (hot-encoded vector)
	images_train = [x.reshape(784,1) / 255.0 for x in X_train]
	labels_encoded_train = [np.zeros((10,1)) for y in y_train]
	for i, l in enumerate(y_train): labels_encoded_train[i][l] = 1

	# Reshape for test, same as training
	images_test = [x.reshape(784,1) / 255.0 for x in X_test]
	labels_encoded_test = [np.zeros((10,1)) for y in y_test]
	for i, l in enumerate(y_test): labels_encoded_test[i][l] = 1

	# training is a list of (image, hot-encoded label vector) tuples.
	# test is a list of (image, label) tuples.
	train = [(x,y) for x,y in zip(images_train, labels_encoded_train)]
	test =[(x,y) for x,y in zip(images_test, labels_encoded_test)]
	# test = [(x,y) for x,y in zip(images_test, y_test)]
	validate = test[8000:]
	test = test[:8000]

	return train, test, validate