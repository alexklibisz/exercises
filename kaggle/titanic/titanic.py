import sys
import pandas
import argparse
import matplotlib.pyplot as plt
import numpy as np
import helpers
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Block: setting up arguments
ap = argparse.ArgumentParser()
ap.add_argument('--seed', help='random seed', default=11, type=int)
ap.add_argument('--summary', help='show data summaries', action='store_true')
ap.add_argument('--tuning', help='do algorithm tuning', action='store_true')
args = vars(ap.parse_args())
seed = args['seed']

# Block: seed the np random number generator
np.random.seed(seed)

# Block: get the data
df, Xtrn, Xval, Ytrn, Yval = helpers.get_train_data()

# Block: High level summaries of the data
if args['summary']:
    print(df.head(10))
    print(df.describe())
    print(df.dtypes)
    print(df.groupby('survived').size())
    print(df.shape)
    # Histogram to show distribution
    df.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
    plt.savefig('tmp-histograms.png')
    # Density plot to show distribution
    df.plot(kind='density', subplots=True, layout=(3,4), sharex=False, legend=True, fontsize=1)
    plt.savefig('tmp-density.png')
    # Box-and-whisker plot to show distribution
    df.plot(kind='box', subplots=True,layout=(3,4), sharex=False, sharey=False, fontsize=1, legend=True)
    plt.savefig('tmp-boxplots.png')
    # Correlation matrix
    corr = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, vmin=-1, vmax=1, interpolation='none')
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    fig.colorbar(cax)
    plt.savefig('tmp-correlation-matrix.png')

# Block: spot-check algorithms
print("Spot-checking, no scaling.")
scoring = 'accuracy'
# keep track of all results, estimators, and names
results = []
estimators = []
names = []
models = [
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVC', SVC())
]
for name, model in models:
    kfold = cross_validation.KFold(n=len(Xtrn),n_folds=10)
    estimator = Pipeline([(name, model)])
    cvres = cross_validation.cross_val_score(estimator, Xtrn, Ytrn, cv=kfold, scoring=scoring)
    results.append(cvres)
    names.append(name)
    estimators.append(estimator)
    print("%s: %f, %f" % (name, cvres.mean(), cvres.std()))

# Block: spot-check algorithms with scaling by creating a series of pipelines
print("Spot-checking, with scaling.")
scaler = ('Scaler', StandardScaler())
# Create a pipeline with scaling for each previously-defined model.
pipelines = []
for name, model in models:
    pipelines.append(('Scaled' + name, Pipeline([scaler, (name, model)])))

for name, pipeline in pipelines:
    kfold = cross_validation.KFold(n=len(Xtrn),n_folds=10)
    cvres = cross_validation.cross_val_score(pipeline,Xtrn,Ytrn,cv=kfold,scoring=scoring)
    names.append(name)
    results.append(cvres)
    estimators.append(pipeline)
    print("%s: %f, %f" % (name, cvres.mean(), cvres.std()))

# Block: Spot-check ensemble methods
print("Spot-checking ensemble methods, no scaling")
ensembles = [
    ('Adaboost', AdaBoostClassifier()),
    ('Gradboost', GradientBoostingClassifier()),
    ('Randfor', RandomForestClassifier()),
    ('Extratree', ExtraTreesClassifier())
]
for name, model in ensembles:
    kfold = cross_validation.KFold(n=len(Xtrn),n_folds=10)
    cvres = cross_validation.cross_val_score(model,Xtrn,Ytrn,cv=kfold,scoring=scoring)
    results.append(cvres)
    names.append(name)
    estimators.append(model)
    print("%s: %f, %f" % (name, cvres.mean(), cvres.std()))

# Block: tune the SVC algorithm by doing a grid search over
# the C and kernel parameters
if args['tuning']:

    Xtrnsc = StandardScaler().fit_transform(Xtrn)
    kfold = cross_validation.KFold(n=len(Xtrnsc),n_folds=10)

    print("Algorithm tuning: SVM")
    Cvals = np.append(np.arange(0.1, 3.0, 0.2), 1.0)
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    params = dict(C=Cvals, kernel=kernels)
    model = SVC()
    grid = GridSearchCV(estimator=model, param_grid=params, scoring=scoring, cv=kfold, n_jobs=4)
    gridres = grid.fit(Xtrnsc, Ytrn)
    print("Best score = %f, with params = %s" % (gridres.best_score_, gridres.best_params_))
    # Re-run the model with tuned parameters
    name = 'Tuned SVM'
    model = gridres.best_estimator_
    cvres = cross_validation.cross_val_score(model,Xtrn,Ytrn,cv=kfold,scoring=scoring)
    print("%s: %f, %f" % (name, cvres.mean(), cvres.std()))
    results.append(cvres)
    names.append(name)
    estimators.append(model)

# Block: plot the spot-checking results
pltfname = 'tmp-algorithm-spot-checking.png'
print("Saving histograms to %s" % (pltfname))
fig = plt.figure()
fig.suptitle('Algorithm Spot-checking')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names, fontsize=9, rotation='vertical')
plt.subplots_adjust(bottom=0.15)
plt.savefig(pltfname)

# Block: find the best model
print("Determining best model")
best = dict(name=None, mean=0, estimator=None)
for name, result, estimator in zip(names,results,estimators):
    if(result.mean() > best['mean']):
        best = dict(name=name, mean=result.mean(), estimator=estimator)

print("Best model: %s, %f" % (best['name'],best['mean']))

# Block: use the best estimator on validation data
print("Evaluating valdiation data with %s" % (best['name']))
model = best['estimator']
model.fit(Xtrn,Ytrn)
predictions = model.predict(Xval)
print("accuracy = %f" % (accuracy_score(Yval, predictions)))
print("classification report")
print(classification_report(Yval, predictions))

# Block: use the estimator on holdout data
print("Making predictions for test (holdout) data with: %s" % (best['name']))
_, Xtest = helpers.get_test_data()
predictions = model.predict(Xtest)
submissionfname = 'tmp-submission.csv'
print("Saving predictions to %s" % (submissionfname))
helpers.create_submission(predictions, submissionfname=submissionfname)
