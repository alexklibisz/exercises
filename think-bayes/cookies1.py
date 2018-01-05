
from thinkbayes import Pmf

pmf = Pmf()

# Equal priors for both bowls.
pmf.Set('Bowl 1', 0.5)
pmf.Set('Bowl 2', 0.5)

# Update distribution based on the probability of drawing a vanilla cookie.
pmf.Mult('Bowl 1', 0.75)
pmf.Mult('Bowl 2', 0.5)
pmf.Normalize()

print pmf.Prob('Bowl 1')
print pmf.Prob('Bowl 2')
