from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class MyLinearRegression():

    def __init__(self):

        self._theta = None # Weights kept as state b/w fit and predict functions.

    def fit(self, X, y, alpha=0.0005, num_iter=10000, stochastic=False, 
        verbose=False):
        
        m,n = X.shape

        # Prepend X with a column of ones (bias) of size m x 1.
        # Explained: http://stats.stackexchange.com/questions/181603
        X = np.column_stack((np.ones(shape=(m,1)), X))

        # Initialize theta, one for each of the n + 1 columns.
        self._theta = np.ones(n + 1)

        # Transpose once for gradient multiplication.
        XT = X.transpose()

        # TODO: implement stochastic gradient descent.

        # Update weights up to num_iter times.
        for i in range(0, num_iter):

            # Hypothesis equals features (x) times theta.
            h = np.dot(X, self._theta)

            # Update status while iterating.
            if verbose and i % 100 is 0:
                print('%04d MAE = %.3f' % (i, MSE(h,y)))

            # Compute the gradient with size m x 1.
            # This is based on the derivative of the MSE cost function.
            # Derivation worked out: http://mccormickml.com/2014/03/04/gradient-descent-derivation/

            errors = h - y                          # Hypothesis minus true results.
            slope = (2/m) * np.dot(XT, errors)      # Slope tells us which direction to go next.
            change = alpha * slope                  # Move slope iterations of alpha in that direction.
            self._theta = self._theta - change      # Make the update

        return self._theta

    def predict(self, X, make_normal=True, scale=True):

        m,n = X.shape

        # Prepend ones to X.
        ones = np.ones(shape=(m,1))
        X_ones = np.column_stack((ones, X))
        
        # Multiply to get the prediction.
        return np.dot(X_ones, self._theta)

def main():

    # Use age, tax, and number of rooms to predict median housing price.
    # http://archive.ics.uci.edu/ml/datasets/Housing

    names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
    'TAX','PTRATIO','B','LSTAT','MEDV'] 
    df = pd.read_csv('./data/housing.data', names=names, header=None, 
        delim_whitespace=True)

    X = df[['AGE','RM','TAX']].values
    y = df['MEDV'].values

    # Prepare the data: scale to mean 0 stddev 1.
    # http://stats.stackexchange.com/a/10298
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=48)

    # Scikit-learn baseline.
    print('Baseline: scikit-learn LinearRegression')
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print('  mean absolute error = %lf' % (MAE(pred, y_test)))
    print('  mean squared error  = %lf' % (MSE(pred, y_test)))

    # My implementation.
    print('Comparison: custom implementation MyLinearRegression')
    model = MyLinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print('  mean absolute error = %lf' % (MAE(pred, y_test)))
    print('  mean squared error  = %lf' % (MSE(pred, y_test)))  


if __name__ == "__main__":
    main()


