# Forest fires dataset

Practicing multi-variable linear regression

## Task

- Create a multi-variable regression model using the normal equation to predict the 'area' variable.
- Use part of the data for training and part for validation.
- Not necessary to normalize variables when using the normal equation.

## Steps

1. Clean the data. Convert month and day abbreviations to integer values.
2. Write function `[train, test] = partitionData(trainPercentage, testPercentage)` to randomly split the data into training and testing groups.
3. Write function `theta = normalEquation(trainingData)` to calculate theta for the training data.

... Still fuzzy here ...


4. Write a function `area = predictArea(features, theta)` to take a row of features and calculate its area.  
5. Apply theta on the training data and find the R^2 value for the known y values.
6. Apply the same theta on the test data and find the R^2 values.
7. Compare R^2 values.
