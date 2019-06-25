# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import skimage
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pydicom

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

X = []
y = []

dir = "Actual0"
for filename in os.listdir(dir):
    img = pydicom.read_file(os.path.join(dir, filename)).pixel_array
    im_rez = skimage.transform.resize(img, (256, 256, 1))
    X.append(im_rez)
    y.append(0)

dir = "Actual1"
for filename in os.listdir(dir):
    img = pydicom.read_file(os.path.join(dir, filename)).pixel_array
    im_rez = skimage.transform.resize(img, (256, 256, 1))
    X.append(im_rez)
    y.append(1)

X = np.array(X)
y = np.array(y)

Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.25, random_state=42)

print(type(Xtr))
classifier.summary()
classifier.fit(Xtr, ytr,
               steps_per_epoch = 8000,
               epochs = 25,
               validation_data = (Xts, yts),
               validation_steps = 2000)

classifier.evaluate(Xts, yts)
print(classifier.metrics_names)
