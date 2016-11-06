
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.optimizers import SGD
import numpy
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=False, help='file with weights')
ap.add_argument('-s', '--seed', required=False, help='random seed')
args = vars(ap.parse_args())

# Seed numpy, which will affect how the NN weights are initialized.
if args["seed"] is not None:
  numpy.random.seed(int(args["seed"]))
else:
  numpy.random.seed(7)

# Load the Pima Indians dataset
print("Loading dataset...")
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
features = dataset[:, 0:8] # features are columns 0 through 8 
labels = dataset[:, 8]     # label value is the 9th column

# Define the model
print("Defining model...")
model = Sequential()
# Layer 1: dense layer with 12 outputs, 8 inputs, 
# uniform weight initialization, rectified linear unit activation 
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))

# Layer 2: dense layer with 8 outputs, uniform weight initialization,
# rectified linear unit activation
model.add(Dense(8, init='uniform', activation='relu'))

# Layer 3: dropout layer to prevent over-fitting
model.add(Dropout(0.25))

# Layer 4: dense layer with 1 output, uniform weight initialization,
# sigmoid activation function
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile the model
print("Compilng model...")
# Define the parameters for stochastic-gradient-descent, where
# the learning rate will decay 0.0001 at each epoch
sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=False)

# Compiled to minimize binary_crossentropy (logloss) using Adaptive Moment Estimation
# or stochastic-gradient-descent and measuring accuracy
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Load weights if defined or fit the model
if args["weights"] is not None:
  print("Loading weights...")
  model.load_weights(args["weights"])
else:
  print("Fitting and saving weights...")
  # Set checkpoint for saving the best weights
  checkpt = ModelCheckpoint('weights.best.h5', monitor='acc', save_best_only=True, mode='auto')
  # Fit the model for 150 epochs in batches of 10
  fit = model.fit(features, labels, nb_epoch=150, batch_size=10, verbose=0, callbacks=[checkpt])
  model.save_weights("weights.h5", overwrite=True)
  # Save history accuracy and loss to file
  history = numpy.zeros((len(fit.history["acc"]), len(fit.history.keys())))
  history[:,0] = fit.history["acc"]
  history[:,1] = fit.history["loss"]
  numpy.savetxt("history.csv", history, delimiter=",")

# Evaluate the model
print("Evaluating...")
scores = model.evaluate(features, labels, verbose=0)

# Print results
print('loss', str(scores[0]))
print('accuracy', str(scores[1]))
print(model.summary())
