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
        c = Counter(observations)
        return PMF.from_dict(c)

    @staticmethod
    def from_kde(observations, X):
        kde = scipy.stats.gaussian_kde(observations)
        return PMF(hypos=X, priors=kde.evaluate(X))

    @staticmethod
    def from_mixture(pmfs):
        """Instantiate a PMF using a list of (pmf, weight) tuples."""
        c = Counter()
        for pmf, weight in pmfs:
            for hypo, prob in pmf.items():
                c[hypo] += weight * prob
        return PMF.from_dict(c)

    @staticmethod
    def from_dict(d):
        return PMF(hypos=list(d.keys()), priors=list(d.values()))

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
        for H in self.hypos:
            self[H] *= self.likelihood(D, H)
        return self.normalize()

    def copy(self):
        return PMF(self.hypos, self.probs)

    def expectation(self):
        return sum(self.index * self.values)

    def var(self):
        mu = self.expectation()
        return sum((self.index - mu)**2 * self.values)

    def std(self):
        return np.sqrt(self.var())

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

    def __lt__(self, other):
        """Equivalent to Downey's Pmf.ProbLess function."""

        if isinstance(other, PMF):
            total = 0
            for v1, p1 in self.items():
                for v2, p2 in other.items():
                    if v1 < v2:
                        total += p1 * p2
            return total

        else:
            return sum(self.probs * (self.hypos < other))

    def __gt__(self, other):
        """Equivalent to Downey's Pmf.ProbGreater function."""

        if isinstance(other, PMF):
            total = 0
            for v1, p1 in self.items():
                for v2, p2 in other.items():
                    if v1 > v2:
                        total += p1 * p2
            return total

        else:
            return sum(self.probs * (self.hypos > other))

    def __eq__(self, other):

        if isinstance(other, PMF):
            total = 0
            for v1, p1 in self.items():
                for v2, p2 in other.items():
                    if v1 == v2:
                        total += p1 * p2
            return total

        else:
            return sum(self.probs * (self.hypos == other))


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

    def __eq__(self, x):
        return NotImplementedError

    def __lt__(self, x):
        i = self.hypos[self.hypos < x].max()
        return self[i]

    def __gt__(self, x):
        """P(X > x)"""
        ccdf = self.to_ccdf()
        i = ccdf.hypos[ccdf.index <= x].max()
        return ccdf[i]

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


class Joint:

    def marginal(self, i):
        c = Counter()
        for ndkey, prob in self.items():
            c[ndkey[i]] += prob
        return PMF(list(c.keys()), list(c.values()))

    def conditional(self, i, j, val):
        """Gets the conditional distribution of the indicated variable.

        Distribution of ndkey[i], conditioned on ndkey[j] = val.

        i: index of the variable we want
        j: which variable is conditioned on
        val: the value the jth variable has to have

        Returns: PMF
        """
        c = Counter()
        for ndkey, prob in self.items():
            if ndkey[j] == val:
                c[ndkey[i]] += prob
        return PMF.from_dict(c)

    def mlintervals(self, P=[0.25, 0.50, 0.75]):
        """Maximum-likelihood intervals.

        Uses a greedy method to identify the hypos with likelihoods that 
        sum to satisfying the specified intervals.

        # Arguments:
            P: list of interval sizes, each in [0, 1].

        # Returns:
            PMF: PMF object with probs weighted by the number of credible
            regions in which the hypo occurred.
        """

        # Pre-sort by probability in descending order.
        greedy = PMF(self.hypos, self.probs).sort_values(ascending=False)
        greedy = greedy.sort_values(ascending=False).cumsum()

        c = {hypo: 0 for hypo in greedy.index}

        for p in P:
            assert 0 <= p <= 1
            lim = sum(greedy < p) + 1
            hypos = greedy.iloc[:lim].index
            for h in hypos:
                c[h] += 1

        return PMF.from_dict(c)

    @staticmethod
    def contour_args(pmf):
        """Return the arguments needed for plt.contour and similar 
        color-map style visualizations."""
        xs, ys = zip(*pmf.hypos)
        xs = sorted(set(xs))
        ys = sorted(set(ys))
        X, Y = np.meshgrid(xs, ys)
        func = np.vectorize(lambda x, y: pmf[(x, y)] if (x, y) in pmf else 0)
        return X, Y, func(X, Y)


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


class RedLineCalculator:
    """Single class encapsulating model from Chapter 8 Redline problem.

    # Arguments:
        Z_historic: either a list of train arrival intervals in minutes, or
            a PMF representing the train arrival intervals.
        passengers_historic: a list in format (k1, y, k2) where y is the 
            number of minutes spent waiting and k2 is the number of passengers
            that arrived during the wait.

    """

    def __init__(self, Z_historic, passengers_historic):

        # PMF for Z: arrival intervals.
        if isinstance(Z_historic, PMF):
            self.Z = Z_historic
        else:
            lo, hi = 0, 2 * max(Z_historic)
            self.Z = PMF.from_kde(Z_historic, np.arange(lo, hi))

        # PMF for Zb: observer-biased Z.
        self.Zb = PMF(self.Z.hypos, self.Z.hypos * self.Z.probs)

        # PMF for X and Y: elapsed time and passenger waiting time.
        # This is computed as a weighted mixture of uniform
        # distributions for the possible waiting times.
        pmfs = []
        for interval, prob in self.Zb.items():
            pmfs.append((PMF(range(int(interval) + 1)), prob))
            pmfs[-1][0][0] *= 0

        self.X = PMF.from_mixture(pmfs)

        # PMF for lambda: arrival rate.
        # Immediately compute a posterior distribution based on the
        # wait times and numbers of passengers observed.
        hypos = np.linspace(1e-7, 5, 51)
        self.lam = PMF(hypos)
        for lam_hypo in self.lam.hypos:
            for _, y, k2 in passengers_historic:
                # y is time (minutes) spent waiting.
                # k2 is the number of passengers who arrived in that time.
                # like is P(k2 | passenger rate per minute * minutes spent
                # waiting)
                like = scipy.stats.poisson.pmf(k2, y * lam_hypo)
                self.lam[lam_hypo] *= like
        self.lam.normalize()

    def estimate_Y(self, n_passengers=15):
        """Estimates Y (waiting time) distribution given number of passengers 
        observed.

        # Arguments:
            n_passengers: number of passengers observed on the platform.

        # Returns:
            Y: mixed PMF over waiting times.
        """

        # Construct mixture of Y PMFs for values of lambda.
        pmfs = []

        for lam, lam_prob in self.lam.items():

            # Update X to reflect this arrival rate and number of passengers.
            X_post = self.X.copy()

            for elapsed, elapsed_prob in X_post.items():
                # P(n_passengers | lambda, minutes elapsed hypothesis)
                like = scipy.stats.poisson.pmf(n_passengers, lam * elapsed)
                X_post[elapsed] *= like

            X_post.normalize()

            # Estimate Y from Zb and X posterior.
            # Remove negative values and zero-out the 0 entry.
            Y = self.Zb - X_post
            Y[0] *= 0
            for h in Y.hypos:
                if h < 0:
                    del Y[h]
            Y.normalize()

            pmfs.append((Y, lam_prob))

        return PMF.from_mixture(pmfs)
