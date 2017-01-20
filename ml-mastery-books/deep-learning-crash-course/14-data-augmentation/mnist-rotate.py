
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# Load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# Convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Define data preparation
datagen = ImageDataGenerator(rotation_range=90)

# Fit parameters from data
datagen.fit(X_train)

# Configure batch size and retrieve one batch of images
shown = 0
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
  # Create a 3x3 grid of images
  for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_batch[i].reshape(28,28), cmap=pyplot.get_cmap('gray'))
  
  # Show the plot
  pyplot.show()
  shown += 1
  if shown > 3:
    break
