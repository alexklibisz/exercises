from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################

        def relu(z):
            return np.maximum(0, z)  # Element-wise max against 0.

        def softmax(z):
            z_ = z - np.max(z)
            expz = np.exp(z_)
            return expz / np.sum(expz, axis=1, keepdims=True)

        a0 = X
        z1 = np.dot(a0, W1) + b1
        a1 = relu(z1)               # hidden_layer in the notes.
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)
        scores = z2

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################

        def entropy_loss(p, y):
            '''Entropy loss - returns the mean loss for the correct class' probability across all examples.'''
            label_probs = p[range(y.shape[0]), y]
            return np.mean(-np.log(label_probs))

        def l2reg(reg, W):
            return 0.5 * reg * np.sum([np.sum(w**2) for w in W])

        loss = entropy_loss(a2, y) + l2reg(reg, [W1, W2])

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################

        def relu_prime(z):
            '''Relu is max(0, z), so derivative wrt z is 1 if z > 0, otherwise 0.'''
            return z > 0.

        # Equation BP1 from Nielsen's Chapter 2.
        # Compute the delta at the output layer. The delta's express the partial
        # derivative of the cost function wrt the intermediate z outputs, which
        # are dot products of the inputs and weights. So they tell us how much to
        # change the weights independent of the activations.
        # Full form is delta_L = partial C/partial a_L * softmax'(z_2).
        # **This is the cost function's instantaneous ROC wrt each activation,
        # scaled by the activation's ROC wrt each input from the previous layer.**

        # The full form simplifies to the output error for a softmax classifier with entropy loss.
        # (If we were to use a different cost or activation on the last layer,
        # the full form would be necessary.) For the full form, the following would be useful:
        # Cross-entropy cost derivative: https://stats.stackexchange.com/questions/154879
        # Softmax derivative:
        # http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        d2 = a2
        d2[range(N), y] -= 1

        # Equation BP2 from Nielsen's Chapter 2.
        # Compute error at the hidden layer. Similar to how the output layer's
        # error was expressed in terms of the cost function, this layer's error is
        # expressed in terms of the subsequent layer's cost. This layer's output
        # affects the subsequent layer's weights, so we scale subsequent layer's
        # error by its weights. Finally, we scale the error/weight product by ROC
        # for this layer's relu activation.
        d1 = np.dot(d2, W2.T) * relu_prime(z1)

        # Average the errors over all training examples.
        d1 /= N
        d2 /= N

        # Use the averaged layer-wise errors to compute the gradient for weights
        # and biases. Add l2 reg. derivative (reg * W_i) to each weight gradient.
        grads['b1'] = np.sum(d1, axis=0)
        grads['b2'] = np.sum(d2, axis=0)
        grads['W1'] = np.dot(a0.T, d1) + (reg * W1)
        grads['W2'] = np.dot(a1.T, d2) + (reg * W2)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 1000 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        def relu(z):
            return np.maximum(0, z)  # Element-wise max against 0.

        def softmax(z):
            z_ = z - np.max(z)
            expz = np.exp(z_)
            return expz / np.sum(expz, axis=1, keepdims=True)

        a0 = X
        z1 = np.dot(a0, W1) + b1
        a1 = relu(z1)               # hidden_layer in the notes.
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)
        y_pred = np.argmax(a2, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred
