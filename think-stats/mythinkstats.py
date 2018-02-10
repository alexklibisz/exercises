from math import ceil
import numpy as np
import pandas as pd
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

    dct = ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip', nrows=nrows)

    if 'FemPreg' in dct_file and 'FemPreg' in dat_file:
        CleanFemPreg(df)

    if 'FemResp' in dct_file and 'FemResp' in dat_file:
        CleanFemResp(df)

    return df


def cohen_effect_size(samplea, sampleb):
    xbara = np.mean(samplea)
    xbarb = np.mean(sampleb)
    na = len(samplea)
    nb = len(sampleb)
    s = (na * np.var(samplea) + nb * np.var(sampleb)) / (na + nb)
    s = np.sqrt(s)
    return (xbara - xbarb) / s


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
    return 100 * sum(b) / len(m)


def percentile(a, q):
    """Compute the value in a that falls at percentile q."""
    if type(a) not in {pd.Series, np.array}:
        a = np.array(a)
    n = int(ceil(q / 100 * len(a)))
    return sorted(a)[n - 1]


def pmf_to_cdf(pmf):
    return pmf.sort_index().cumsum()


def cdf_percentile(cdf, q):
    q = q if 0 <= q <= 1 else q / 100.
    return cdf.index[np.argmin(abs(cdf.values - q))]


def cdf_percentile_rank(cdf, q):
    m = sum(cdf.index <= q) - 1
    return cdf.values[m] * 100


def cdf_random_sample(cdf, n):
    """Generate a random sample from a CDF by computing the values for 
    randomly-chosen percentiles."""
    Q = np.random.uniform(0, 100, n)
    return pd.Series(Q).apply(lambda q: cdf_percentile(cdf, q))
