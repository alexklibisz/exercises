import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

def get_train_data(fname='data-train.csv', valsize = 0.3, seed=9, collapse=False):

    # Block: read in file
    df = pd.read_csv(fname)

    if collapse:
        # Block: collapse binary wilderness_area and soil_type into single columns
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

    # Block: Convert all columns to floats so that scaling doesn't show warnings.
    cols = df.columns.values
    for col in cols:
        df[col] = df[col].astype('float64')

    # Block: Split data for testing and validation
    Y = df['Cover_Type'].values
    X = df.drop(['Cover_Type'], axis = 1).values
    Xtrn, Xval, Ytrn, Yval = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=seed)

    return df, Xtrn, Xval, Ytrn, Yval

def get_test_data(fname='data-test.csv', collapse=False):

    # Block: read in file
    df = pd.read_csv(fname)

    if collapse:
        # Block: collapse binary wilderness_area and soil_type into single columns
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

    # Block: Convert all columns to floats so that scaling doesn't show warnings.
    cols = df.columns.values
    for col in cols:
        df[col] = df[col].astype('float64')

    # Block: Split data for testing and validation
    Xtest = df.values

    return df, Xtest

def create_submission(predictions, datafname='data-test.csv', submissionfname='tmp-submission.csv'):

    df = pd.read_csv(datafname)
    df['Cover_Type'] = pd.Series(predictions).astype('str')
    df.to_csv(submissionfname, columns=['Id', 'Cover_Type'], index=False)
