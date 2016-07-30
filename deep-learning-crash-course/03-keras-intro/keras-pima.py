
from keras.models import Sequential
from keras.layers import Dense
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

# Layer 3: dense layer with 1 output, uniform weight initialization,
# sigmoid activation function
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile the model
print("Compilng model...")
# Compiled to minimize binary_crossentropy (logloss) using Adaptive Moment Estimation
# and measuring accuracy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load weights if defined or fit the model
if args["weights"] is not None:
  print("Loading weights...")
  model.load_weights(args["weights"])
else:
  print("Fitting and saving weights...")
  # Fit the model for 150 epochs in batches of 10
  model.fit(features, labels, nb_epoch=150, batch_size=10, verbose=0)
  model.save_weights("out.h5", overwrite=True)

# Evaluate the model
print("Evaluating...")
scores = model.evaluate(features, labels, verbose=0)

# Print results
print('loss', str(scores[0]))
print('accuracy', str(scores[1]))
print(model.summary())
