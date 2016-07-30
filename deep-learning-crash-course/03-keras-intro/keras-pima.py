
from keras.models import Sequential
from keras.layers import Dense
import numpy
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=False, help='file with weights')
ap.add_argument('-s', '--seed', required=False, help='random seed')
args = vars(ap.parse_args())

# Seed numpy
seed = 7
numpy.random.seed(seed)

# Load the dataset
print("Loading dataset...")
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
features = dataset[:, 0:8] # features are columns 0 through 8 
labels = dataset[:, 8]   # prediction value is the 9th column

# Define the model
print("Defining model...")
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile the model
print("Compilng model...")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load weights if defined or fit the model
if args["weights"] is not None:
  print("Loading weights...")
  model.load_weights(args["weights"])
else:
  print("Fitting...")
  model.fit(features, labels, nb_epoch=150, batch_size=10, verbose=0)
  model.save_weights("out.h5", overwrite=True)

# Evaluate the model
print("Evaluating...")
scores = model.evaluate(features, labels, verbose=0)

# Print results
print('loss', str(scores[0]))
print('accuracy', str(scores[1]))
