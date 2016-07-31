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


Areas where I need to study more on my own:

- Extra trees and random trees classifiers - was shown very quickly in the text
