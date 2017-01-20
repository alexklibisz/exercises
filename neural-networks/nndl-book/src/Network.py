import pickle
import random
import numpy as np
import math

# Static Functions
def sigmoid(z):
    '''Sigmoid activation function.'''
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    '''Derivative of the sigmoid activation function.
    Used in back prop to compute the output and layer errors.'''
    return sigmoid(z) * (1 - sigmoid(z))

def quadratic_cost(examples, outputs):
    '''Computes quadratic cost (eq. 6) for the outputs relative
    the example (x,y) pairs. Assumes that outputs is a list of
    column vectors representing the output activations for
    the given examples.'''
    n = len(examples)
    error_sum = 0
    for (o, (x,y)) in zip(outputs, examples):
        error_sum += math.pow(np.sum(y - o), 2)
    return (1 / (2*n)) * error_sum

def cost_prime(output_activations, y):
    '''Derivative of the quadratic cost function.
    Used in back prop to compute the output error.
    Assumes output_activations and y are equal sized
    column vectors.'''
    return (output_activations - y)

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
                correct, cost = self.evaluate(test_data)
                print("SGD Epoch %2d: %5d/%5d, cost = %.4lf" %
                (e, correct, len(test_data), cost))
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
        tuples (x,y). Returns the number of accurate classifications.
        and the overall cost.'''

        # Compute the outputs as a forward pass once.
        outputs = [self.feed_forward(x) for (x,y) in test_data]

        # It's a match if the index of the greatest activation
        # in the output matches the index of the greatest
        # value (the digit) in the example label.
        num_matches = 0
        for (o,(x,y)) in zip(outputs, test_data):
            num_matches += int(np.argmax(o) == np.argmax(y))

        cost = quadratic_cost(test_data, outputs)

        return num_matches, cost

    def back_prop(self, x, y):
        '''Execute back propogation for a single example input x
        and its corresponding output y. Used in the update_by_mini_batch
        function for applying changes to weights and biases.
        Return a tuple of the gradients (grad_b, grad_w) where grad_b and
        grad_w are layer-by-layer lists of numpy arrays.
        grad_w[l][j][k] is the gradient for the weight from neuron k in
        layer l-1 to neuron j in layer l.
        grad_b[l][j] is the bias for neuron j in layer l.'''

        # Both gradients initialized as zeros.
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        # Duplicate the feed_forward logic.
        # Keep track of a and z for later computing deltas.
        a = x
        A = [a]
        Z = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b       # The inner term of the activation, z.
            Z.append(z)
            a = sigmoid(z)
            A.append(a)

        # Backward pass.
        # Compute the output error first; see eq. BP1 and BP1a
        delta = cost_prime(A[-1], y) * sigmoid_prime(Z[-1])
        grad_b[-1] = delta                            # eq. BP3
        grad_w[-1] = np.dot(delta, A[-2].transpose()) # eq. BP4

        # Use python negative indices to loop backward through layers.
        # TODO: find a clean way to do this without negative indexing.
        for l in range(2, self.num_layers):
            sp = sigmoid_prime(Z[-l])
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp # eq. BP2
            grad_b[-l] = delta                                           # eq. BP3
            grad_w[-l] = np.dot(delta, A[-l-1].transpose())              # eq. BP4

        return (grad_b,grad_w)

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
