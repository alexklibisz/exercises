import pdb
import numpy as np
np.random.seed(122)

"""
Exercise 4.2: you measure the outcome of coin flips using an instrument that
is not always correct. There is a probability y of the recorded measurment
being correct.

Write a class that estimates the bias of a coin given a series of outcomes
and the value of y.

The solution below attempts to recover the MAP estimate of an unknown bias
using samples generated with binomial(p=bias) for various biases and various
values of y. We use y to introduce uncertainty in the likelihood, which is
also used to compute the joint distribution and the normalizing constant.
For each value of y we track the mean-absolute-error between the actual
bias and the MAP estimate. We notice that as y increases, MAE decreases:

y = 0.10, MAE = 0.225
y = 0.15, MAE = 0.225
y = 0.20, MAE = 0.225
y = 0.25, MAE = 0.225
y = 0.30, MAE = 0.250
y = 0.35, MAE = 0.250
y = 0.40, MAE = 0.250
y = 0.45, MAE = 0.225
y = 0.50, MAE = 0.100
y = 0.55, MAE = 0.025
y = 0.60, MAE = 0.050
y = 0.65, MAE = 0.050
y = 0.70, MAE = 0.050
y = 0.75, MAE = 0.025
y = 0.80, MAE = 0.025
y = 0.85, MAE = 0.025
y = 0.90, MAE = 0.025
y = 0.95, MAE = 0.025
y = 1.00, MAE = 0.000

"""


def estimate_biases(y, obs, biases, prior):
    """
    parameters:
    y: float probability of a measurement being correct.
    obs: array of 1s and 0s, where 1 represents Heads, 0 Tails.
    biases: array of possible bias values.
    prior: array of probabilities corresponding to each bias.

    return:
    post: posterior distribution over biases given the obs and y
    MAP: max a-posteriori estimate from dist

    p(bias|obs) = p(obs|bias) * p(bias) / p(obs)

    """

    # Adjust samples to reflect measurement error y.
    obs = (y * obs) + ((1 - y) * (1 - obs))

    # likelihood = p(obs|bias) =
    # [ p(obs 1|bias 1), p(obs 1|bias 2), ..., p(obs 1|bias m),
    #   ...
    #   p(obs n|bias 1), p(obs n|bias 2), ..., p(obs n|bias m) ]
    lH = (obs[:, np.newaxis]).dot(biases[np.newaxis, :])
    lT = (1. - obs[:, np.newaxis]).dot((1. - biases[np.newaxis, :]))
    like = lH + lT

    # joint = p(obs|bias) * p(bias)
    # p(obs|bias j) = p(obs 1|bias j) * p(obs 2|bias j) ... * p(bias j)
    joint = like.prod(0) * prior

    # normalizing constant = p(obs 1) * ... * p(obs n) = sum(joint)
    ncnst = joint.sum()

    # posterior = joint / normalizing constant.
    post = joint / ncnst

    return post, biases[np.argmax(post)]


if __name__ == "__main__":

    flips = 200
    biases = np.arange(3, 8) / 10.
    prior = np.minimum(1 - biases, biases)
    prior /= prior.sum()
    h_probs = np.arange(40, 71, 10) / 100.
    y_probs = np.arange(10, 101, 5) / 100.
    y_mae = np.zeros(len(y_probs))

    for h in h_probs:
        obs = np.random.binomial(1, h, size=flips)
        for i in range(len(y_probs)):
            post, MAP = estimate_biases(y_probs[i], obs, biases, prior)
            y_mae[i] += np.abs(MAP - h) / len(h_probs)

    for i in range(len(y_probs)):
        print 'y = %.2lf, MAE = %.3lf' % (y_probs[i], y_mae[i])
