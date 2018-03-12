from collections import Counter
from math import ceil
import csv
import numpy as np
import pandas as pd
import pdb
import scipy.stats


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
        self.sort_index(inplace=True)
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
    def from_kde(observations, X):
        kde = scipy.stats.gaussian_kde(observations)
        return PMF(hypos=X, priors=kde.evaluate(X))

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
        return PMF(self.hypos, self.probs)

    def expectation(self):
        return sum(self.index * self.values)

    def MAP(self):
        i = self.idxmax()
        return i, self[i]

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

        if isinstance(other, (pd.Series, PMF)):
            c = Counter()
            for hi, p1 in self.items():
                for h2, p2 in other.items():
                    c[hi - h2] += p1 * p2
            return PMF(hypos=list(c.keys()), priors=list(c.values()))

        else:
            raise NotImplementedError("Not sure how to do this subtraction.")

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

    def to_ccdf(self):
        return CDF(self.hypos, 1 - self.probs)

    def percentile(self, q):
        mapgtq = self.values > q / 100
        return min(self.index.values[mapgtq])

    def interval(self, Q):
        return [self.percentile(q) for q in Q]

    def lt(self, value):
        return max(self.probs * (self.hypos < value))

    def lteq(self, value):
        return max(self.probs * (self.hypos <= value))

    def gt(self, value):
        return 1 - self.lteq(value)

    def __pow__(self, other):
        return CDF(self.hypos, self.probs ** other)

    def __sub__(self, other):

        if isinstance(other, (pd.Series, CDF)):
            c = Counter()
            for hi, p1 in self.items():
                for h2, p2 in other.items():
                    c[hi - h2] += p1 * p2
            return CDF(list(c.keys()), list(c.values()))

        elif isinstance(other, (float, int)):
            return CDF(self.hypos, self.probs - other)

        else:
            raise NotImplementedError("Not sure how to do this subtraction.")


class DataSets:

    @staticmethod
    def get_price_is_right(filepath):

        cols = ['Showcase 1', 'Showcase 2', 'Bid 1', 'Bid 2',
                'Difference 1', 'Difference 2']
        col2data = {}
        with open(filepath) as fp:
            for t in csv.reader(fp):
                col, data = t[0], t[1:]
                if col not in cols:
                    continue
                col2data[col] = [int(x) for x in data]

        return pd.DataFrame(col2data).rename(
            index=str,
            columns={
                'Showcase 1': 'showcase1',
                'Showcase 2': 'showcase2',
                'Bid 1': 'bid1',
                'Bid 2': 'bid2',
                'Difference 1': 'diff1',
                'Difference 2': 'diff2'})

        return
