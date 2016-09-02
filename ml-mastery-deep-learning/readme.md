Based on exercises from the book Deep Learning with Python by Jason Brownlee: https://machinelearningmastery.com/deep-learning-with-python/

## Notes

- Chapter 6
  - Softmax activation used to output probabilities for a multi-class classification.
  - "One-hot encoding" = converting categorical variables to numerical (e.g. male/female becomes 0/1).
  - ANN required scaling (e.g. each column has a mean of zero and stdd of 1).
  - Good explanation of back propogation on page 41.
  - "Batch learning" = save the errors from each training example, update the network weights all at once after processing all of the training examples. (As opposed to updating weights for every training example.)
  - "Learning rate decay" = decrease the learning rate over time so that larger changes are made to weights at first and then smaller changes are made later on.

- Chapter 8
  - use the `validation_split` argument for `model.fit()` to create an automatic validation set.
  - use the `validation_data` argument for `model.fit()` to specify which data gets split.
  - use the sklearn `StratifiedKFold` to create folds for cross-validation, then loop over the folds to evaluate the algorithm.
