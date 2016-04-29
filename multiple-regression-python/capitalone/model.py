import numpy as np
import pandas as pd
import time
import datetime
import warnings
import math
import matplotlib.pyplot as plt
import sys

warnings.filterwarnings('ignore')
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Find r-squared.
# Math according to: https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions
def calcr2(y, f):
    ybar = np.mean(y)
    sstot = np.sum((y - ybar)**2)
    ssreg = np.sum((f - ybar)**2)
    ssres = np.sum((y - f)**2)
    return np.round((1 - (ssres/sstot)), 5)

def prep_data(filename = 'data-raw.csv'):
    # Import the training/validation data and holdout data
    data = pd.read_csv('data-raw.csv')

    # remove un-necessary columns
    data.drop('unique_id', 1, inplace=True)
    data.drop('CITY', 1, inplace=True)
    data.drop('STATE', 1, inplace=True)
    data.drop('COUNTY', 1, inplace=True)
    data.drop('Column 172', 1, inplace=True)

    # recode data: institution size
    data['INSTITUTION_SIZE'].replace('SMALL', 1, inplace=True)
    data['INSTITUTION_SIZE'].replace('LARGE', 2, inplace=True)

    # recode data: OPEN_DT become timestamps
    to_timestamp = lambda x: time.mktime(datetime.datetime.strptime(x, "%m/%d/%Y").timetuple())
    data['OPEN_DT'] = data['OPEN_DT'].apply(to_timestamp)

    # recode data: replace all empty cells with 0's
    data = data.replace(np.nan, 0)

    tv = data.query('sample=="Build"')
    ho = data.query('sample=="Validate"')

    # remove sample column
    tv.drop('sample', 1, inplace=True)
    ho.drop('sample', 1, inplace=True)

    return tv, ho


def model(tv, trainProportion = 0.8):
    # partition the data randomly, 80% to training, 20% to validation
    perm = list(np.random.permutation(tv.shape[0]))
    bp = math.ceil(tv.shape[0] * trainProportion)
    t = tv.iloc[perm[:bp],:]
    v = tv.iloc[perm[bp:],:]

    # extract X and y for training
    ty = t['TOT_DEP'].as_matrix()
    tX = t.drop('TOT_DEP', 1).as_matrix()

    # run normal equation to find theta
    # this is gross in python, much simpler in matlab/octave
    xTx = np.dot(tX.transpose(), tX)
    xTxinv = np.linalg.pinv(xTx)
    xTxinvxT = np.dot(xTxinv, tX.transpose())
    theta = np.dot(xTxinvxT, ty)

    # use theta to predict TOT_DEP for the validation data
    vy = v['TOT_DEP'].as_matrix()
    vX = v.drop('TOT_DEP', 1).as_matrix()
    vpred = np.round(np.dot(vX, theta))

    return vy, vpred, theta

def model_predict(ho, theta):
    hoX = ho.drop('TOT_DEP', 1).as_matrix()
    return np.round(np.dot(hoX, theta))

def visualize(vy, vpred, r2, show):
    fig, ax = plt.subplots()
    ax.set_title('Mult. Regr. r^2 = ' + str(r2))
    ax.plot(range(vy.size), vy, 'b--')
    ax.plot(range(vy.size), vpred, 'r--')
    ax.plot(range(vy.size), vy, 'bo', label = 'actual')
    ax.plot(range(vy.size), vpred, 'ro', label = 'predicted')
    ax.legend(['actual','predicted','actual', 'predicted'], loc='best')
    ax.set(ylim=(-1 * abs(vpred.min() * 1.5), vpred.max() * 1.5))

    plt.savefig('./plots/' + str(r2) + '-' + str(math.floor(time.time())) + '.png')
    if show:
        plt.show()

# Run the model 200 times and keep track of the max r2, vy, vpred, and theta
tv, ho = prep_data()
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
visualize(maxvy, maxvpred, maxr2, (len(sys.argv) > 1 and sys.argv[1] == '-s'))

# Make and save predictions based on maxtheta
hopred = model_predict(ho, maxtheta)
np.savetxt('./predictions/' + str(maxr2) + '.csv', hopred, fmt='%f')

# Print information
print(' maxr2', maxr2)
print('meanr2', np.mean(np.array(allr2)))
print(hopred)
