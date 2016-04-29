import pandas as pd
import statsmodels.api as sm
import numpy as np
from shared import *

warnings.filterwarnings('ignore')
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

def model(tv, trainProportion = 0.8):
    # partition the data randomly, 80% to training, 20% to validation
    perm = list(np.random.permutation(tv.shape[0]))
    bp = math.ceil(tv.shape[0] * trainProportion)
    t = tv.iloc[perm[:bp],:]
    v = tv.iloc[perm[bp:],:]

    # extract X and y for training
    ty = t['TOT_DEP'].as_matrix()
    tX = t.drop('TOT_DEP', 1).as_matrix()


    return vy, vpred, theta

def model_predict(ho, theta):
    hoX = ho.drop('TOT_DEP', 1).as_matrix()
    return np.round(np.dot(hoX, theta))

# Run the model 200 times and keep track of the max r2, vy, vpred, and theta
tv, ho = prep_data()
model(tv, 0.75)
quit()

allr2 = []
maxr2 = 0
for _ in range(250):
    vy, vpred, theta = model(tv, 0.75)
    r2 = calcr2(vy, vpred)
    allr2.append(r2)
    if r2 > maxr2:
        maxvy = vy
        maxvpred = vpred
        maxr2 = r2
        maxtheta = theta

# Create and save plot
visualize(maxvy, maxvpred, 'normal-equation', maxr2, (len(sys.argv) > 1 and sys.argv[1] == '-s'))

# Make and save predictions based on maxtheta
hopred = model_predict(ho, maxtheta)
np.savetxt('./predictions/normal-equation-' + str(maxr2) + '.csv', hopred, fmt='%f')

# Print information
print(' maxr2', maxr2)
print('meanr2', np.mean(np.array(allr2)))
print(hopred)
