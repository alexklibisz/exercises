from keras.datasets import mnist
import numpy as np
import os
import Network

# # Fetch from sklearn (mldata.org)
# # For some reason, this performed drastically worse than
# # the example from neuralnetworksanddeeplearning.com.
# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')
# mnist.target = [int(t) for t in mnist.target]
# images = [x.reshape(784,1) / 255.0 for x in mnist.data]
# labels = [np.zeros((10,1)) for x in mnist.target]
# for i, t in enumerate(mnist.target): labels[i][t] = 1
# pairs = [(x,y) for x,y in zip(images,labels)]
# training = pairs[0:50000]
# test = [(x,y) for x,y in zip(images[50000:], mnist.target[50000:])]
# validate = pairs[60000:70000]

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
training = [(x,y) for x,y in zip(images_train, labels_encoded_train)]
test =[(x,y) for x,y in zip(images_test, labels_encoded_test)]
# test = [(x,y) for x,y in zip(images_test, y_test)]
validate = test[8000:]
test = test[:8000]

# Train, save, delete the network.
net = Network.Network([784, 30, 10])
net.SGD(training, 30, 10, 3.0, test_data=test)
net.save(fname='mnist.out')
del net

# Load the network back in and run validation.
net = Network.Network()
net.load(fname='mnist.out')
num_matches, cost = net.evaluate(validate)
print('Validation: %d / %d, %.4lf, cost = %.4lf' %
    (num_matches, len(validate), num_matches / len(validate), cost))
