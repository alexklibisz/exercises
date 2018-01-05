
# Implements exercise 2.1, Page 19
import pdb
from thinkbayes import Pmf


class Bowl(Pmf):

    def __init__(self, hypos):
        Pmf.__init__(self)
        self.n = float(sum(hypos.values()))
        for hypo, cnt in hypos.iteritems():
            self.Set(hypo, cnt / self.n)

    def RemoveSingle(self, data):
        for hypo in self.Values():
            p = self.Prob(hypo)
            if hypo == data:
                p -= 1 / self.n
            self.Set(hypo, p)
        self.Normalize()


class Cookie(Pmf):

    mixes = dict(Bowl1=Bowl(dict(vanilla=75, chocolate=25)),
                 Bowl2=Bowl(dict(vanilla=50, chocolate=50)))

    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1. / len(hypos))

    def Update(self, data, replace=True):
        # Multiply p(hypo) * p(data | hypo)
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

        # Reduce the hypothetical probability if not replacing cookie.
        if not replace:
            for name, bowl in self.mixes.iteritems():
                bowl.RemoveSingle(data)

    def Likelihood(self, data, hypo):
        # Return likelihood = p(data | hypothesis).
        return self.mixes[hypo].Prob(data)

if __name__ == "__main__":

    pmf = Cookie(['Bowl1', 'Bowl2'])

    # Tell it we pulled a vanilla cookie.
    pmf.Update('vanilla')

    # Now print the posterior probabilities after pulling a cookie.
    for h, p in pmf.Items():
        print('p(%s | vanilla) = %.3lf' % (h, p))

    print '*' * 10

    # Try it again, this time sampling multiple cookies with replacement.
    pmf = Cookie(['Bowl1', 'Bowl2'])
    samples = ['vanilla', 'vanilla']
    [pmf.Update(s) for s in samples]

    for h, p in pmf.Items():
        print('p(%s | %s) = %.5lf' % (h, ','.join(samples), p))

    print '*' * 10

    # Try it again, this time sampling multiple cookies with replacement.
    pmf = Cookie(['Bowl1', 'Bowl2'])
    samples = ['vanilla', 'vanilla', 'chocolate']
    [pmf.Update(s) for s in samples]

    for h, p in pmf.Items():
        print('p(%s | %s) = %.5lf' % (h, ','.join(samples), p))

    print '*' * 10

    # Now without replacement.
    pmf = Cookie(['Bowl1', 'Bowl2'])
    samples = ['vanilla', 'vanilla', 'chocolate']
    [pmf.Update(s, replace=False) for s in samples]

    for h, p in pmf.Items():
        print('p(%s | %s) = %.5lf' % (h, ','.join(samples), p))
