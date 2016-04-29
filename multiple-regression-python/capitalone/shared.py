import numpy as np
import pandas as pd
import time
import datetime
import warnings
import math
import matplotlib.pyplot as plt
import sys

def calcmse(y, f):
    return np.mean(((y - f) ** 2))

# Find residual sum of squares.
# Math according to: https://en.wikipedia.org/wiki/Residual_sum_of_squares#One_explanatory_variable
def calcrss(y, f):
    return np.sum((y - f)**2)

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
    data = pd.read_csv(filename)

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

def visualize(vy, vpred, type, r2, show):
    fig, ax = plt.subplots()
    ax.set_title('r^2 = ' + str(r2))
    ax.plot(range(vy.size), vy, 'b--')
    ax.plot(range(vy.size), vpred, 'r--')
    ax.plot(range(vy.size), vy, 'bo', label = 'actual')
    ax.plot(range(vy.size), vpred, 'ro', label = 'predicted')
    ax.legend(['actual','predicted','actual', 'predicted'], loc='best')
    ax.set(ylim=(-1 * abs(vpred.min() * 1.5), vpred.max() * 1.5))

    plt.savefig('./plots/' + type + '-' + str(r2) + '-' + str(math.floor(time.time())) + '.png')
    if show:
        plt.show()
