"""
Implementing Exercise 3.1, page 31.

Given:
1. There are N companies.
2. Each company has at most Tmax trains.
3. Each company has T trains, sampled from a power law distribution, alpha=1.
4. Each train is equally likely to be observed.
5. Train Tobs is observed.

Approximate:
1. The most probable number of trains? ... Not sure what the question
is asking to approximate. Perhaps an interesting one would be the most
probable company? But I'm not sure how this can be compared to the results
from train3.py?

"""

import thinkbayes
import thinkplot

from thinkbayes import Pmf, Percentile
from dice import Dice


class Company(Pmf):

    def __init__(self, Tmax_, alpha=1.0):
        print(Tmax_, alpha)
        return


def main():

    Nmax_vals = [500, 1000, 2000, 4000]  # Max. no. companies.
    Tmax_vals = [500, 1000, 2000, 4000]  # Max. no. trains for any company.
    Tobs_vals = [30, 60, 90]             # Number on the train observed.

    for Tmax_ in Tmax:
        for Nmax_ in Nmax:
            C = [Company(Tmax_) for _ in range(Nmax_)]

#     for high in Lmax:
#         suite = MakePosterior(high, L, Train2)

#     for high in [500, 1000, 2000]:
#         suite = MakePosterior(high, dataset, TrainPowerLaw)
#         print high, suite.Mean()

#     thinkplot.Save(root='train3',
#                    xlabel='Number of trains',
#                    ylabel='Probability')

#     interval = Percentile(suite, 5), Percentile(suite, 95)
#     print interval

#     cdf = thinkbayes.MakeCdfFromPmf(suite)
#     interval = cdf.Percentile(5), cdf.Percentile(95)
#     print interval


if __name__ == '__main__':
    main()
