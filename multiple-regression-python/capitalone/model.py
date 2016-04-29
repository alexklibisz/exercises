import numpy as np
import pandas as pd
import time
import datetime
import warnings
import math
import matplotlib.pyplot as plt
import sys

warnings.filterwarnings('ignore')

# Import the training/validation data and holdout data
data = pd.read_csv('data-raw.csv')
tv = data.query('sample=="Build"')
hd = data.query('sample=="Validate"')

# remove un-necessary columns
tv.drop('unique_id', 1, inplace=True)
tv.drop('CITY', 1, inplace=True)
tv.drop('STATE', 1, inplace=True)
tv.drop('COUNTY', 1, inplace=True)
tv.drop('sample', 1, inplace=True)
tv.drop('Column 172', 1, inplace=True)

# recode data: institution size
tv['INSTITUTION_SIZE'].replace('SMALL', 1, inplace=True)
tv['INSTITUTION_SIZE'].replace('LARGE', 2, inplace=True)

# recode data: OPEN_DT become timestamps
to_timestamp = lambda x: time.mktime(datetime.datetime.strptime(x, "%m/%d/%Y").timetuple())
tv['OPEN_DT'] = tv['OPEN_DT'].apply(to_timestamp)

# recode data: replace all empty cells with 0's
tv = tv.replace(np.nan, 0)

# partition the data randomly, 80% to training, 20% to validation
perm = list(np.random.permutation(tv.shape[0]))
bp = math.ceil(tv.shape[0] * 0.8)
t = tv.iloc[perm[:bp],:]
v = tv.iloc[perm[bp:],:]

# extract X and y for training
ty = t['TOT_DEP'].as_matrix()
tX = t.drop('TOT_DEP', 1).as_matrix()

# run normal equation to find theta
# this is gross, much simpler in matlab/octave
xTx = np.dot(tX.transpose(), tX)
xTxinv = np.linalg.pinv(xTx)
xTxinvxT = np.dot(xTxinv, tX.transpose())
theta = np.dot(xTxinvxT, ty)

# use theta to predict TOT_DEP for the validation data
vy = v['TOT_DEP'].as_matrix()
vX = v.drop('TOT_DEP', 1).as_matrix()
vpred = np.round(np.dot(vX, theta))

# plot stuff
fig, ax = plt.subplots()
ax.set_title('Multiple regression w/ Normal equation')
ax.plot(range(vy.size), vy, 'b--')
ax.plot(range(vy.size), vpred, 'r--')
ax.plot(range(vy.size), vy, 'bo', label = 'actual')
ax.plot(range(vy.size), vpred, 'ro', label = 'predicted')
ax.legend(['actual','predicted','actual', 'predicted'], loc='best')
ax.set(ylim=(vpred.min() * 1.5, vpred.max() * 1.5))

plt.savefig('./plots/' + str(math.floor(time.time())) + '.png')
if len(sys.argv) > 1 and sys.argv[1] == '-p':
    plt.show()
