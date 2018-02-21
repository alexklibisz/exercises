from math import ceil, pi, sqrt, log
import numpy as np
import pandas as pd
import scipy.stats
import re


class FixedWidthVariables(object):
    """Represents a set of variables in a fixed width file."""

    def __init__(self, variables, index_base=0):
        """Initializes.

        variables: DataFrame
        index_base: are the indices 0 or 1 based?

        Attributes:
        colspecs: list of (start, end) index tuples
        names: list of string variable names
        """
        self.variables = variables

        # note: by default, subtract 1 from colspecs
        self.colspecs = variables[['start', 'end']] - index_base

        # convert colspecs to a list of pair of int
        self.colspecs = self.colspecs.astype(np.int).values.tolist()
        self.names = variables['name']

    def ReadFixedWidth(self, filename, **options):
        """Reads a fixed width ASCII file.

        filename: string filename

        returns: DataFrame
        """
        df = pd.read_fwf(filename,
                         colspecs=self.colspecs,
                         names=self.names,
                         **options)
        return df


def ReadStataDct(dct_file, **options):
    """Reads a Stata dictionary file.

    dct_file: string filename
    options: dict of options passed to open()

    returns: FixedWidthVariables object
    """
    type_map = dict(byte=int, int=int, long=int, float=float, double=float)

    var_info = []
    for line in open(dct_file, **options):
        match = re.search(r'_column\(([^)]*)\)', line)
        if match:
            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startswith('str'):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = ' '.join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))

    columns = ['start', 'type', 'name', 'fstring', 'desc']
    variables = pd.DataFrame(var_info, columns=columns)

    # fill in the end column by shifting the start column
    variables['end'] = variables.start.shift(-1)
    variables.loc[len(variables) - 1, 'end'] = 0

    dct = FixedWidthVariables(variables, index_base=1)
    return dct


def CleanFemPreg(df):
    """Recodes variables from the pregnancy frame.

    df: DataFrame
    """
    # mother's age is encoded in centiyears; convert to years
    df.agepreg /= 100.0

    # birthwgt_lb contains at least one bogus value (51 lbs)
    # replace with NaN
    df.loc[df.birthwgt_lb > 20, 'birthwgt_lb'] = np.nan

    # replace 'not ascertained', 'refused', 'don't know' with NaN
    na_vals = [97, 98, 99]
    df.birthwgt_lb.replace(na_vals, np.nan, inplace=True)
    df.birthwgt_oz.replace(na_vals, np.nan, inplace=True)
    df.hpagelb.replace(na_vals, np.nan, inplace=True)

    df.babysex.replace([7, 9], np.nan, inplace=True)
    df.nbrnaliv.replace([9], np.nan, inplace=True)

    # birthweight is stored in two columns, lbs and oz.
    # convert to a single column in lb
    # NOTE: creating a new column requires dictionary syntax,
    # not attribute assignment (like df.totalwgt_lb)
    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0

    # due to a bug in ReadStataDct, the last variable gets clipped;
    # so for now set it to NaN
    df.cmintvw = np.nan


def CleanFemResp(df):
    """Recodes variables from the respondent frame.

    df: DataFrame
    """
    pass


def read_data(dct_file, dat_file, nrows=None):
    return nsfg_read_data(dct_file, dat_file, nrows)


def nsfg_read_data(dct_file, dat_file, nrows=None):

    dct = ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip', nrows=nrows)

    if 'FemPreg' in dct_file and 'FemPreg' in dat_file:
        CleanFemPreg(df)

    if 'FemResp' in dct_file and 'FemResp' in dat_file:
        CleanFemResp(df)

    return df


def babyboom_read_data(filename):
    """Reads the babyboom data.

    filename: string

    returns: DataFrame
    """
    var_info = [
        ('time', 1, 8, int),
        ('sex', 9, 16, int),
        ('weight_g', 17, 24, int),
        ('minutes', 25, 32, int),
    ]
    columns = ['name', 'start', 'end', 'type']
    variables = pd.DataFrame(var_info, columns=columns)
    variables.end += 1
    dct = FixedWidthVariables(variables, index_base=1)
    return dct.ReadFixedWidth(filename, skiprows=59)


def CleanBrfssFrame(df):
    """Recodes BRFSS variables.

    df: DataFrame
    """
    # clean age
    df.age.replace([7, 9], float('NaN'), inplace=True)

    # clean height
    df.htm3.replace([999], float('NaN'), inplace=True)

    # clean weight
    df.wtkg2.replace([99999], float('NaN'), inplace=True)
    df.wtkg2 /= 100.0

    # clean weight a year ago
    df.wtyrago.replace([7777, 9999], float('NaN'), inplace=True)
    df['wtyrago'] = df.wtyrago.apply(
        lambda x: x / 2.2 if x < 9000 else x - 9000)


def brfss_read_data(filename='CDBRFS08.ASC.gz', compression='gzip', nrows=None):
    """Reads the BRFSS data.

    filename: string
    compression: string
    nrows: int number of rows to read, or None for all

    returns: DataFrame
    """
    var_info = [
        ('age', 101, 102, int),
        ('sex', 143, 143, int),
        ('wtyrago', 127, 130, int),
        ('finalwt', 799, 808, int),
        ('wtkg2', 1254, 1258, int),
        ('htm3', 1251, 1253, int),
    ]
    columns = ['name', 'start', 'end', 'type']
    variables = pd.DataFrame(var_info, columns=columns)
    variables.end += 1
    dct = FixedWidthVariables(variables, index_base=1)

    df = dct.ReadFixedWidth(filename, compression=compression, nrows=nrows)
    CleanBrfssFrame(df)
    return df


def hinc_clean(s):
    """Converts dollar amounts to integers."""
    try:
        return int(s.lstrip('$').replace(',', ''))
    except ValueError:
        if s == 'Under':
            return 0
        elif s == 'over':
            return np.inf
        return None


def hinc_read_data(filename='hinc06.csv'):
    """Reads filename and returns populations in thousands

    filename: string

    returns: pandas Series of populations in thousands
    """
    data = pd.read_csv(filename, header=None, skiprows=9)
    cols = data[[0, 1]]

    res = []
    for _, row in cols.iterrows():
        label, freq = row.values
        freq = int(freq.replace(',', ''))

        t = label.split()
        low, high = hinc_clean(t[0]), hinc_clean(t[-1])

        res.append((high, freq))

    df = pd.DataFrame(res)
    # correct the first range
    df.loc[0, 0] -= 1
    # compute the cumulative sum of the freqs
    df[2] = df[1].cumsum()
    # normalize the cumulative freqs
    total = df[2][41]
    df[3] = df[2] / total
    # add column names
    df.columns = ['income',  'freq', 'cumsum', 'ps']
    return df


def populations_read_data(filename='PEP_2012_PEPANNRES_with_ann.csv'):
    """Reads filename and returns populations in thousands

    filename: string

    returns: pandas Series of populations in thousands
    """
    df = pd.read_csv(filename, header=None, skiprows=2, encoding='iso-8859-1')
    populations = df[7]
    populations.replace(0, np.nan, inplace=True)
    return populations.dropna()


def cohen_effect_size(samplea, sampleb):
    xbara = np.mean(samplea)
    xbarb = np.mean(sampleb)
    na = len(samplea)
    nb = len(sampleb)
    s = (na * np.var(samplea) + nb * np.var(sampleb)) / (na + nb)
    s = np.sqrt(s)
    return (xbara - xbarb) / s


def trim_outliers(a, p=0.01):
    n = int(p * len(a))
    if type(a) == pd.Series:
        a.sort_values(inplace=True)
    else:
        a.sort()
    return a[n:-n]


def pmf_observer_bias(pmf):
    """Introduces observer bias in a PMF that maps count -> proportion"""
    biased = pmf.copy()
    biased *= pmf.index
    return biased / biased.sum()


def pmf_observer_unbias(pmf):
    """Removes observer bias from a PMF that maps count -> proportion"""
    unbiased = pmf.copy()
    unbiased /= pmf.index
    return unbiased / unbiased.sum()


def pmf_expectation(pmf):
    return sum(pmf.index * pmf.values)


def pmf_variance(pmf):
    exp = pmf_expectation(pmf)
    return sum(pmf.values * (pmf.index - exp) ** 2)


def percentile_rank(a, q):
    """Compute the proportion of values in a less than or equal to q"""
    if type(a) not in {pd.Series, np.array}:
        a = np.array(a)
    m = a <= q
    return 100 * sum(m) / len(m)


def percentile(a, q):
    """Compute the value in a that falls at percentile q."""
    if type(a) not in {pd.Series, np.array}:
        a = np.array(a)
    n = int(ceil(q / 100 * len(a)))
    return sorted(a)[n - 1]


def pmf_to_cdf(pmf, precision=5):
    cdf = pmf.copy()
    cdf.index = cdf.index.values.round(precision)
    cdf = cdf.sort_index().cumsum()
    return cdf / cdf.max()


def cdf_percentile(cdf, q):
    """percentile = proportion of values in cdf less than q.
    Return the value representing percentile q, e.g. in a list of integers
    1 to 100, the percentile q is equal to the number q."""
    return max(cdf.index.values * (cdf.values <= q / 100))


def cdf_percentile_rank(cdf, q):
    m = sum(cdf.index <= q) - 1
    return cdf.values[m] * 100


def cdf_random_sample(cdf, n):
    """Generate a random sample from a CDF by computing the values for
    randomly-chosen percentiles."""
    Q = np.random.uniform(0, 100, n)
    return pd.Series(Q).apply(lambda q: cdf_percentile(cdf, q))


def cdf_pvalue(cdf, x):
    """Return the P(x > X) computed from a CDF."""
    return 1 - max(cdf.values * (cdf.index < x))


def pmf_exponential(lam, X=np.linspace(0, 5, 1001)):
    return pd.Series(lam * np.exp(-lam * X), index=X)


def rnd_exponential(lam, n):
    """Random exponential sample.
    exponential CDF: p(x) = 1 - exp(-lam * x)
    solving for x: x = -log(1 - p) / lam"""
    p = np.random.uniform(0, 1, n)
    x = -1 * np.log(1 - p) / lam
    return pd.Series(x)


def pmf_normal(mu, sig, n=1000, Z=4):
    """Generate a PMF for a Normal distribution over a range of values.
    If you want a *sample* from a normal distribution, use np.random.normal."""
    X = np.random.uniform(mu - Z * sig, mu + Z * sig, n)
    P = np.exp((-1 * (X - mu) ** 2) / (2 * sig ** 2)) / sqrt(2 * pi * sig**2)
    return pd.Series(P, index=X)


def pmf_pareto(xm, alpha, X=np.linspace(0, 5, 1001)):
    P = (alpha * (xm ** alpha)) / (X ** (alpha + 1) + 1e-7) * (X > xm)
    return pd.Series(P, index=X)


def rnd_pareto(xm, alpha, n):
    """Random pareto sample.
    exponential CDF: p(x) = 1 - (xm/x)^alpha
    solving for x: x = xm / (1-p)^(1/alpha)
    http://www.wolframalpha.com/input/?i=p+%3D+1+-+(b+%2F+x)%5Ea,+solve+for+x"""
    p = np.random.uniform(0, 1, n)
    x = xm / (1 - p)**(1 / alpha)
    return pd.Series(x)


def pmf_weibull(lam, k, X=np.linspace(0.01, 5, 1001)):
    P = (k / lam) * ((X + 1e-7) / lam) ** (k - 1) * np.exp(-1 * (X / lam)**k)
    return pd.Series(P, index=X)


def rnd_weibull(lam, k, n):
    """Random weibull sample using inverse transform sampling.
    x = lam(-log(1 - p))^(1/k)
    """
    P = np.random.uniform(0, 1, n)
    X = lam * (-1 * np.log(1 - P))**(1 / k)
    return pd.Series(X)


def covariance(X, Y):
    return np.dot(
        X - X.mean(),
        Y - Y.mean()) / len(X)


def pearson_correlation(X, Y):
    """Covariance expressed in terms of (standardized) Z-scores."""
    return covariance(X, Y) / X.std() / Y.std()


def spearman_correlation(X, Y):
    """Pearson correlation evaluated on the ranks of (X_i, Y_i) pairs."""
    X_rank = pd.Series(X).rank().values
    Y_rank = pd.Series(Y).rank().values
    return pearson_correlation(X_rank, Y_rank)


def est_mse(true, est):
    return np.mean((true - est)**2)


def est_rmse(true, est):
    return np.sqrt(est_mse(true, est))


def est_mean_error(true, est):
    return np.mean(true - est)


def est_rsquared(X, Y, m, b):
    res = Y - (m * X + b)
    return 1 - np.var(res) / np.var(Y)


def est_residual(X, Y, m, b):
    return Y - (m * X + b)


def fit_least_squares(X, Y):
    """Y ~= mx + b"""
    m = covariance(X, Y) / np.var(X)
    b = np.mean(Y) - m * np.mean(X)
    return m, b


def serial_correlation(X, lag=1, corrfunc=pearson_correlation):
    return corrfunc(X[:-lag], X[lag:])


def autocorrelation(X, lags=np.arange(0, 365), corrfunc=pearson_correlation):
    """A very lazy autocorrelation implementation"""
    return [(lag, corrfunc(X[:-lag], X[lag:])) for lag in lags]
