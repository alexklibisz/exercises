from collections import Counter
from math import ceil
import numpy as np
import pandas as pd
import pdb


class PMF(pd.Series):
    """A thin wrapper on Pandas Series class used to represent a discrete
    random variable for Bayesian modeling."""

    def __init__(self, hypos=[], priors=None, **kwargs):
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

    @staticmethod
    def from_cdf(cdf):
        hypos = cdf.hypos
        probs = cdf.probs - cdf.shift(1).fillna(0).values
        return PMF(hypos, probs)

    def to_cdf(self):
        return CDF.from_pmf(self)

    @staticmethod
    def from_observations(observations):
        """Instantiate the PMF using a list of observed values."""
        counts = Counter(observations)
        return PMF(hypos=list(counts.keys()), priors=list(counts.values()))

    @staticmethod
    def from_mixture(pmfs):
        """Instantiate a PMF using a list of (pmf, weight) tuples."""
        probs = Counter()
        for pmf, weight in pmfs:
            for hypo, prob in pmf.items():
                probs[hypo] += weight * prob

        return PMF(list(probs.keys()), list(probs.values()))

    @property
    def hypos(self):
        return self.index.values

    @property
    def probs(self):
        return self.values

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

    def MAP(self):
        return i, self[self.idxmax()]

    def likelihood(self, data, hypo):
        """Application-specific likelihood function that must be implemented
        by sub-classes of PMF."""
        raise NotImplementedError

    def sample(self, size=1):
        return np.random.choice(self.hypos, size, p=self.probs)

    def __add__(self, other):
        """Computes the PMF of the sum of values in self and other.

        other: another PMF instance.
        returns: new PMF instance.
        """
        c = Counter()
        for h1, p1 in self.items():
            for h2, p2 in other.items():
                c[h1 + h2] += p1 * p2
        return PMF(hypos=list(c.keys()), priors=list(c.values()))

    def __sub__(self, other):
        """Computes the PMF of the difference of values in self and other.

        other: another PMF instance.
        returns: new PMF instance.
        """
        c = Counter()
        for hi, p1 in self.items():
            for h2, p2 in other.items():
                c[hi - h2] += p1 * p2
        return PMF(hypos=list(c.keys()), priors=list(c.values()))

    def __pow__(self, other):
        return PMF(self.hypos, self.probs ** other)


class CDF(pd.Series):

    def __init__(self, hypos, probs, **kwargs):
        super().__init__(probs, index=hypos, **kwargs)

    @property
    def hypos(self):
        return self.index.values

    @property
    def probs(self):
        return self.values

    @staticmethod
    def from_pmf(pmf):
        cdf = pmf.sort_index().cumsum()
        return CDF(cdf.index, cdf / cdf.max())

    def to_pmf(self):
        return PMF.from_cdf(self)

    def percentile(self, q):
        mapgtq = self.values > q / 100
        return min(self.index.values[mapgtq])

    def interval(self, Q):
        return [self.percentile(q) for q in Q]

    def __pow__(self, other):
        return CDF(self.hypos, self.probs ** other)
