import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg, delta=1):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                loss += margin

                # Partial derivative of L_i wrt w_y_i = -x_i.
                # Partial derivative of L_i wrt w_j = x_i.
                # In both cases, take the mean slope of all training examples.
                dW[:, y[i]] -= (X[i, :] / num_train)
                dW[:, j] += (X[i, :] / num_train)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg, delta=1):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    pass

    XW = np.matrix(X) * np.matrix(W)
    XWy = XW[np.arange(X.shape[0]), y].T

    # Vectorized sum(max(x_i*w_j - x_i*w_y_i + delta))
    # Margins for the true class activations get set to zero.
    margins = np.maximum(0, XW - XWy + delta)
    margins[np.arange(X.shape[0]), y] = 0

    # Sum along the rows to get the combined margin for each example.
    # Then take the mean of these sums and add regularization.
    loss = np.mean(np.sum(margins, axis=1))
    loss += reg * np.sum(W ** 2)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # Define a mask s.t. mask[i][j] = 1 iff the margin for example i, class j > 0
    mask = margins
    mask[mask > 0] = 1

    # Sum the mask along the class axis s.t. row_sum[i] is the number of
    # classes where example i had a positive margin.
    row_sum = np.sum(mask, axis=1)

    # Update the mask to have -1 at each example's true class if the margin
    # for that class is > 0. This will account for the fact that d L_i / d
    # W_y_i = -x_i if margin > 0.
    mask[np.arange(X.shape[0]), y] = -row_sum.T

    # At this point the mask has the following values for example i, class j:
    # - mask[i][j] = 0 if margin for example i, class j < 0 (no update needed).
    # - mask[i][j] = 1 if margin for example i, class j > 0 AND class j is not the true class (update needed).
    # - mask[i][j] < 0 if margin for example i, class j > 0 AND class j is the true class (negative update needed).

    # Multiply X' by the mask to get the sums of features according to the
    # update rules. Then take their mean to get the gradient.
    dW = np.dot(X.T, mask)
    dW /= X.shape[0]

    # Regularize
    dW += reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
