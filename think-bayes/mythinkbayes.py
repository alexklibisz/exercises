import numpy as np
import pandas as pd
import pdb


class PMF(pd.Series):
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
        self.normalize()

    @property
    def hypos(self):
        return self.index

    def normalize(self):
        self /= self.sum()
        return self

    def update(self, D):
        for H in self.index.values:
            self[H] *= self.likelihood(D, H)
        return self.normalize()

    def copy(self):
        return PMF(self.index, self.values)

    def expectation(self):
        return sum(self.index * self.values)

    def cdf(self):
        return

    def likelihood(self, data, hypo):
        raise NotImplementedError


class CDF(pd.Series):

    @staticmethod
    def from_pmf(pmf):
        cdf = pmf.sort_index().cumsum()
        return CDF(cdf.index, cdf / cdf.max())

    def __init__(self, hypos, probs, **kwargs):
        super().__init__(probs, index=hypos, **kwargs)

    def percentile(self, q):
        return max(self.index.values * (self.values < q / 100))
