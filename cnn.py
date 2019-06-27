# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as K
import skimage
from sklearn.model_selection import train_test_split
import pydicom

os.environ["CUDA_VISIBLE_DEVICES"]="0"

classifier = Sequential()

classifier.add(Conv2D(64, (5, 5), input_shape = (256, 256, 1), activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(MaxPooling2D(pool_size = (4, 4)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(MaxPooling2D(pool_size = (4, 4)))

classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

X = []
y = []

dir = "Data/Processed0"
for filename in os.listdir(dir):
    img = np.load(os.path.join(dir, filename))
    im_rez = skimage.transform.resize(img, (256, 256, 1))
    X.append(im_rez)
    y.append(0)

dir = "Data/Processed1"
for filename in os.listdir(dir):
    img = np.load(os.path.join(dir, filename))
    im_rez = skimage.transform.resize(img, (256, 256, 1))
    X.append(im_rez)
    y.append(1)

X = np.array(X)
y = np.array(y)

Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.25, random_state=42)

print(type(Xtr))
classifier.summary()
es = EarlyStopping(patience=10, restore_best_weights = True) 

history = classifier.fit(Xtr, ytr,
               callbacks = [es],
               epochs = 100,
               batch_size = 8,
               validation_data = (Xts, yts))
classifier.evaluate(Xts, yts)
classifier.save("model_even_better.h5")
import pickle as pickle
pickle.dump(history, open("history_even_better.pkl", "rb"))
