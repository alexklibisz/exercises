import pickle
import random
import numpy as np

# Static Functions
def sigmoid(z):
    '''Sigmoid activation function.'''
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    '''Derivative of the sigmoid activation function.
    Used in back prop.'''
    return sigmoid(z) * (1 - sigmoid(z))

class Network(object):

    def __init__(self, sizes=[10,10]):
        '''Initializes member variables and randomly
        initializes biases and weights'''

        self.num_layers = len(sizes)
        self.sizes = sizes

        # Initialize biases and weights randomly  with mean 0, stddev 1.
        # By convention, no weights, biases for the first (input) layer.

        # Biases are a (y by 1) column vector.
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # Weights are a (y by x) matrix, where x is the
        # size of the layer i and y is the size of the layer i+1.
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feed_forward(self, a):
        '''Return network output given input a.
        Uses biases and weights already initialized as member variables.'''
        # Each layer's activation requires multiplying the weight
        # matrix by the input vector, adding the layer's weight,
        # and taking the sigmoid of that sum.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''Trains the NN using mini-batch stochastic gradient
        descent. Expects training_data to be a list of tuples
        (x,y) representing training inputs and correct outputs.'''

        n = len(training_data)
        for e in range(epochs):

            # Shuffle the training data and split into
            # non-overlapping mini-batches.
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            # Use each of the mini_batches to update
            # biases and weights via gradient descent.
            for mb in mini_batches:
                self.update_by_mini_batch(mb, eta)

            # Evaluate on test_data if given.
            if test_data:
                print("SGD Epoch %d: %d / %d" %
                (e, self.evaluate(test_data), len(test_data)))
            else:
                print("SGD Epoch %d complete" % (e))


    def update_by_mini_batch(self, mini_batch, eta):
        '''Use back prop to compute the gradients for each
        (x,y) in mini_batch. Use the gradients to update
        self.weights and self.biases at each iteration.'''

        # Gradients start as all zeros.
        # Note: Nielsen uses "nabla_b" and "nabla_w" because
        # those nabla is the name for the uspide-down triangle.
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.biases]

        # Loop over x,y pairs, compute bias and weight gradients.
        # Add gradients to the existing gradients.
        for x,y in mini_batch:
            delta_grad_b, delta_grad_w = self.back_prop(x,y)
            grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]

        # Update weights and biases according to equation 20 and 21,
        # chapter 1.
        self.weights = [w - (eta / len(mini_batch)) * gw
                        for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - (eta / len(mini_batch)) * gb
                        for b, gb in zip(self.biases, grad_b)]

    def evaluate(self, test_data):
        '''Evaluate using the member variable biases, weights,
        and the given test_data. test_data should be a list of
        tuples (x,y). Returns the number of accurate classifications.'''

        # Results will be tuples indicating the index of the
        # maximum activation from the feed_forward and the index
        # of the 1 in the known labels list y.
        results = [(np.argmax(self.feed_forward(x)),y)
                    for (x,y) in test_data]

        # The number of correct matches is the number of tuples
        # containing the same index (i.e. feed_forward assigned
        # the highest activation at the index corresponding to
        # the correct output.
        return sum(int(x == y) for (x,y) in results)

    def back_prop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def save(self, fname):
        serialized = dict()
        serialized['biases'] = self.biases
        serialized['num_layers'] = self.num_layers
        serialized['sizes'] = self.sizes
        serialized['weights'] = self.weights
        pickle.dump(serialized, open(fname, "wb"))

    def load(self, fname):
        serialized = pickle.load(open(fname, "rb"))
        self.biases = serialized['biases']
        self.num_layers = serialized['num_layers']
        self.sizes = serialized['sizes']
        self.weights = serialized['weights']
