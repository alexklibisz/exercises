
import pandas as pd
import numpy as np
import argparse
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

# Block parse script args
ap = argparse.ArgumentParser()
ap.add_argument('--seed', help='random seed', default=11, type=int)
ap.add_argument('--summary', help='show data summaries', action='store_true')
ap.add_argument('--tuning', help='do algorithm tuning', action='store_true')
args = vars(ap.parse_args())
seed = args['seed']

# Block: seed numpy random generator
np.random.seed(seed)

# Block: read in file
fname = 'train.csv'
df = pd.read_csv(fname)

# Block: some descriptive statistics
# Correlation matrix, looks like the majority of variables are not significantly correlated.
if args['summary']:
    corr = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    plt.suptitle('Correlation matrix')
    plt.savefig('tmp-correlation-matrix.png')

    # Block: data types - all integers
    print(df.dtypes)

    # Classification count - 2160 of each
    print(df.groupby('Cover_Type').size())

# Block: collapse binary wilderness area and soil type columns into single columns
# Get all the Wilderness_Area and Soil_Type columns
cols = df.columns.values
wcols = list(filter(lambda x: x.startswith('Wilderness_Area') , cols))
scols = list(filter(lambda x: x.startswith('Soil_Type'), cols))
# Initialize two new columns as all zeros
df['Wilderness_Area'] = pd.Series(np.zeros((df.shape[0])))
df['Soil_Type'] = pd.Series(np.zeros((df.shape[0])))
# Loop through the columns, multiply each column's values by (i+1),
# add it to the collapsed column.
for i, col in enumerate(wcols):
    vals = df[col].values * (i + 1)
    df['Wilderness_Area'] += pd.Series(vals)

for i, col in enumerate(scols):
    vals = df[col] * (i + 1)
    df['Soil_Type'] += pd.Series(vals)

# Drop the binary columns
df.drop(wcols + scols, axis=1, inplace=True)
df.to_csv('test.csv', index=False)


# Block: Split data for testing and validation
Y = df['Cover_Type'].values
X = df.drop(['Cover_Type'], axis = 1).values
Xtrn, Xval, Ytrn, Yval = cross_validation.train_test_split(X, Y, test_size=0.3)

#
# # Block: Spot-checking algorithms
# scoring = 'accuracy'




# # Block: Spot-checking ensemble methods
# scoring = 'accuracy'
# names = []
# results = []
# estimators = []
# ensembles = [
#     #('Adaboost', AdaBoostClassifier()),
#     ('Gradboost', GradientBoostingClassifier()),
#     #('Randfor', RandomForestClassifier()),
#     #('Extratree', ExtraTreesClassifier())
# ]
# for name, model in ensembles:
#     kfold = cross_validation.KFold(n=len(Xtrn),n_folds=10)
#     cvres = cross_validation.cross_val_score(model,Xtrn,Ytrn,cv=kfold,scoring=scoring)
#     results.append(cvres)
#     names.append(name)
#     estimators.append(model)
#     print("%s: %f, %f" % (name, cvres.mean(), cvres.std()))
