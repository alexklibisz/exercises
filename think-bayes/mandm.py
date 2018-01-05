
from thinkbayes import Suite


class MandM(Suite):
    """Map from hypothesis (A or B) to probability."""

    mix94 = dict(brown=30,
                 yellow=20,
                 red=20,
                 green=10,
                 orange=10,
                 tan=10,
                 blue=0)

    mix96 = dict(blue=24,
                 green=20,
                 orange=16,
                 yellow=14,
                 red=13,
                 brown=13,
                 tan=0)

    hypoA = dict(bag1=mix94, bag2=mix96)
    hypoB = dict(bag1=mix96, bag2=mix94)

    hypotheses = dict(A=hypoA, B=hypoB)

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.
        hypo: string hypothesis (A or B)
        data: tuple of string bag, string color
        """
        bag, color = data
        mix = self.hypotheses[hypo][bag]
        like = mix[color]
        return like

if __name__ == "__main__":

    suite = MandM('AB')  # A and B are the two possible hypotheses.

    # Update loops through each hypothesis in the suite and multiplies its
    # probability by the likelihood of the data under the hypothesis, which
    # is computed by Likelihood. Likelihood uses mixes, which is a dictionary
    # mapping from the name of a bowl to the mix of cookies in the bowl. Each
    # problem implements its own Likelihood function.

    # How do the probabilities change when you: sample a yellow from bag1.
    suite.Update(('bag1', 'yellow'))

    # Then sample a green from bag2.
    suite.Update(('bag2', 'green'))

    # Print the probabilities of each bag given the sampling sequence.
    # A 0.740740740741
    # B 0.259259259259
    print(suite.Print())
