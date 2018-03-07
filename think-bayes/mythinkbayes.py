import numpy as np
import pandas as pd
import pdb

# def Pmf(values=[]):
#     return pd.Series(values).value_counts(normalize=True)


class DiscreteVariable(pd.Series):
    """A thin wrapper on Pandas Series class used to represent a discrete
    random variable for Bayesian modeling."""

    def __init__(self, hypos, priors=None, **kwargs):
        """
        # Arguments:

        hypos: iterable of keys representing each possible hypothesis.
        priors: initial probability of each hypothesis. If not given, 
        it defaults to a uniform probability.

        """
        if priors is None:
            priors = np.ones(len(hypos)) / len(hypos)
        super().__init__(priors, index=hypos, **kwargs)

    def normalize(self):
        self /= self.sum()
        return self

    def update(self, D):
        for H in self.index.values:
            self[H] *= self.likelihood(D, H)
        return self.normalize()

    def copy(self):
        return DiscreteVariable(self.index, self.values)

    def likelihood(self, data, hypo):
        raise NotImplementedError
