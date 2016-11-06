# Neural Networks and Deep Learning

Exercises from http://neuralnetworksanddeeplearning.com/.
Some source code is directly copied (e.g. reading in mnist), other parts are modified for learning.

In general, I just adapted it to work in python 3 and used the keras mnist data set instead of the data set that Nielsen included in his repo.

## TODO

- Implement concepts from chapter 3:
    - Weights initialized with mean 0 stddev 1.
    - Cross-entropy cost function
    - Early stopping
    - L2 regularization
    - Dropout regularlization
    - Momentum-based gradient descent
    - Try other optimization methods (scipy has some alternatives built-in for quick evaluations.)
    - tanh and rectified-linear activation neurons
- Create a Jupyter notebook demonstrating the performance differences in the first-pass and second-pass networks.
- Try an ensemble method with one network trained per digit.
