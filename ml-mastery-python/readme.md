Based on exercises from the book Machine Learning Mastery with Python by Jason Brownlee: https://machinelearningmastery.com/machine-learning-with-python/

I bought the text, including code, so I've git-ignored the code I write based on Brownlee's code. I'm not sure about how licensing works there, and most of my code is near-verbatim from his snippets.

Chapter-by-chapter information I found valuable in this text:

- Chapter 5: pandas summary(), hist(), and plot() functions
- Chapter 6: correlation matrices, scatter plot matrices
- Chapter 7: sklearn.preprocessing functions for preparing irregular data
  - standardize: transform features with a Gaussian distribution and differing means and std deviations to a standard Gaussian distribution with mean 0 and std dev 1.
  - rescale: scale values so that they fit between two fixed values (e.g. 0 and 1)
  - normalize: transform data so that the sqrt of the sum of squares for each row is 1
- Chapter 8: various methods for selecting the most important features
  - SelectKBest based on stastistical tests like chi2
  - Recursive Feature Elimination
  - Principle component analysis: this one could have been more descriptive
- Chapter 9: methods for evaluating models
  - K-fold cross validation
  - Leave-one-out validation
  - Random splits
  - When to use specific techniques
- Chapter 10: performance metrics
  - Classification: classification accuracy, logarithmic loss, area under ROC curve, confusion matrix, classification report
  - Regression: mean absolute error, mean squared error, R^2
- Chapter 11: spot-checking classification algorithms
  - Simple scikit-learn recipes for logistic regression, linear discriminant analysis, K-nearest neightbors, naive bayes, classification and regresion trees, support vector machines
- Chapter 12: spot-checking regression algorithms
  - Simple scikit-learn recipes for Linear regression, ridge regression, LASSO linear regression, elastic net regression, K-Nearest neightbors, classification and regression trees, support vector machines

Areas where I need to study more on my own:

- Extra trees and random trees classifiers - was shown very quickly in the text
- Logloss performance metric - seems intuitive but how is it actually measured?
- Area under ROC Curve - is this the same as precision/recall?
- Intuitive interpretation for F-scores, precision, and recall
- Linear discriminant analysis
- Naive Bayes and Gaussian Naive Bayes
- Support Vector Machines (review Coursera notes)
- A review of all of the algorithms in chapters 11 and 12. I should probably implement them all myself at some point.
