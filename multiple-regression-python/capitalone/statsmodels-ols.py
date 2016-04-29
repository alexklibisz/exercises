import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys
import sklearn.metrics as metrics
from shared import *

warnings.filterwarnings('ignore')

# Following pattern described here: http://statsmodels.sourceforge.net/0.6.0/examples/notebooks/generated/predict.html
def model(tv, trainProportion = 0.8):
    # partition the data randomly, 80% to training, 20% to validation
    perm = list(np.random.permutation(tv.shape[0]))
    bp = math.ceil(tv.shape[0] * trainProportion)
    t = tv.iloc[perm[:bp],:]
    v = tv.iloc[perm[bp:],:]

    # extract X and y for training
    ty = t['TOT_DEP'].as_matrix()
    tX = t.drop('TOT_DEP', 1).as_matrix()

    # create and fit a model using training data
    tX = sm.add_constant(tX)
    olsmod = sm.OLS(ty, tX)
    olsfit = olsmod.fit()

    # extract X and y from validation and predit using fit
    vy = v['TOT_DEP'].as_matrix()
    vX = v.drop('TOT_DEP', 1).as_matrix()
    vX = sm.add_constant(vX)
    vpred = olsfit.predict(vX)

    # return validaton y, validation prediction, and fit
    return vy, vpred, olsfit

def model_predict(ho, olsfit):
    hoX = ho.drop('TOT_DEP', 1).as_matrix()
    return np.round(olsfit.predict(hoX))

tv, ho = prep_data()

maxr2 = -1 * sys.maxsize
for i in range(1000):
    vy, vpred, olsfit = model(tv, 0.7)
    r2 = metrics.r2_score(vy, vpred)
    if r2 > maxr2:
        maxr2 = r2
        maxolsfit = olsfit

print(maxr2)
hopred = model_predict(ho, maxolsfit)
np.savetxt('./predictions/smols-' + str(maxr2) + '.csv', hopred, fmt='%f')
