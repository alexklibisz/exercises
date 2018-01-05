
from thinkbayes import Pmf


class Cookie(Pmf):

    mixes = dict(Bowl1=dict(vanilla=0.75, chocolate=0.25),
                 Bowl2=dict(vanilla=0.5, chocolate=0.5))

    def __init__(self, hypos):

        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1. / len(hypos))

    def Update(self, data):
        # Multiply p(hypo) * p(data | hypo)
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    def Likelihood(self, data, hypo):
        # Return likelihood = p(data | hypothesis).
        return self.mixes[hypo][data]

if __name__ == "__main__":

    pmf = Cookie(['Bowl1', 'Bowl2'])

    # Tell it we pulled a vanilla cookie.
    pmf.Update('vanilla')

    # Now print the posterior probabilities after pulling a cookie.
    for h, p in pmf.Items():
        print('p(%s | vanilla) = %.3lf' % (h, p))

    # Try it again, this time sampling multiple cookies with replacement.
    pmf = Cookie(['Bowl1', 'Bowl2'])
    samples = ['vanilla', 'vanilla', 'chocolate']
    [pmf.Update(s) for s in samples]

    for h, p in pmf.Items():
        print('p(%s | %s) = %.3lf' % (h, ','.join(samples), p))
