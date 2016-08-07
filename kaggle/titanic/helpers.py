import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

def get_train_data(fname='data-train.csv', impute = True, valsize = 0.3,seed=9):

    # Block: reading in data
    names = ['passengerid','survived','pclass','name','sex',
        'age','sibsp','parch','ticket','fare','cabin','embarked']
    df = pandas.read_csv(fname, names=names, skiprows=[0])

    # Block: cleaning up and recoding categorical columns
    # remove passenger id, names, ticket
    df.drop(['passengerid','name','ticket'], axis=1, inplace=True)
    # recode categorical variables to integers using LabelEncoder
    le_sex = LabelEncoder()
    df['sex'] = le_sex.fit_transform(df['sex'].astype('str'))
    le_cabin = LabelEncoder()
    df['cabin'] = le_cabin.fit_transform(df['cabin'].astype('str'))
    le_embarked = LabelEncoder()
    df['embarked'] = le_embarked.fit_transform(df['embarked'].astype('str'))

    # Block: train/test split
    array = df.values
    X = array[:, 1:]    # "features"
    Y = array[:, 0]   # "labels"
    valsize = 0.3
    Xtrn, Xval, Ytrn, Yval = cross_validation.train_test_split(X,Y,test_size=valsize,random_state=seed)

    # Block: impute missing values seperately for train and test
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    Xtrn = imp.fit_transform(Xtrn)
    Xval = imp.fit_transform(Xval)

    return df, Xtrn, Xval, Ytrn, Yval

def get_test_data(fname='data-test.csv', impute = True):

    # Block: reading in data
    names = ['passengerid','pclass','name','sex', 'age',
        'sibsp','parch','ticket','fare','cabin','embarked']
    df = pandas.read_csv(fname, names=names, skiprows=[0])

    # Block: cleaning up and recoding categorical columns
    # remove passenger id, names, ticket
    df.drop(['passengerid','name','ticket'], axis=1, inplace=True)
    # recode categorical variables to integers using LabelEncoder
    le_sex = LabelEncoder()
    df['sex'] = le_sex.fit_transform(df['sex'].astype('str'))
    le_cabin = LabelEncoder()
    df['cabin'] = le_cabin.fit_transform(df['cabin'].astype('str'))
    le_embarked = LabelEncoder()
    df['embarked'] = le_embarked.fit_transform(df['embarked'].astype('str'))

    # Block: get test data
    Xtest = df.values

    # Block: impute missing values seperately for train and test
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    Xtest = imp.fit_transform(Xtest)

    return df, Xtest
