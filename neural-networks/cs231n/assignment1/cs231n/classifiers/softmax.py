import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    N, D = X.shape
    _, C = W.shape

    for i in xrange(N):

        # Activations for this example.
        f_i = np.dot(X[i], W)
        f_i -= np.max(f_i)

        # Softmax probability of the correct class.
        p_y_i = np.exp(f_i[y[i]]) / np.sum(np.exp(f_i))

        # Loss contributed by this sample.
        loss_i = -np.log(p_y_i)

        # Loss is the mean of all losses.
        loss += loss_i / N

        # Update the gradient of each class based on this example.
        for k in xrange(C):

            # Softmax probability of class k.
            p_k = np.exp(f_i[k]) / np.sum(np.exp(f_i))

            # Partial derivative of the loss function wrt W is X_i * (p_k - (1 if y_i = k))
            # Gradient is the mean of these partial derivatives.
            dW[:, k] += X[i] * (p_k - (k == y[i])) / N

    dW += reg * W
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, D = X.shape
    _, C = W.shape

    # Loss.
    f = np.dot(X, W)
    exp_f = np.exp(f)
    loss = -f[np.arange(N), y]  # Activations of correct class per example.
    loss += np.log(np.sum(exp_f, axis=1))  # Probability sums for all classes per example.
    loss /= N  # Mean loss per example.
    loss = np.sum(loss)  # Sum of the means.

    # Gradient.
    # Partial derivative of the loss function wrt W is X_i * (p_k - (1 if y_i = k))
    # Gradient is the mean of these partial derivatives.
    # Class activations normalized by sums of activations -> "probabilities".
    dW = exp_f / np.sum(exp_f, axis=1, keepdims=True)

    # Subtract one from the true class probabilities.
    dW[range(N), y] -= 1

    # Multiply by X to reflect partial derivative wrt X.
    dW = np.dot(X.T, dW)

    # Mean probabilities per example.
    dW /= N

    # Add regularization
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
