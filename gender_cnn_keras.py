#!/usr/bin/env python
# encoding: utf-8

import cPickle as pickle
import numpy as np
import re
import sys
import os
from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization


if len(sys.argv) != 2:
    print "usage: python {0} csv_file".format(sys.argv[0])
    sys.exit()
csv_file = sys.argv[1]


def load_data(csv_file):
    x = []
    y = []
    with open(csv_file, 'r') as f:
        for line in f:
            img = Image.open(line[:-4])
            img.load()
            r, g, b = img.split()
            channel = []
            channel.append(np.array(r))
            channel.append(np.array(g))
            channel.append(np.array(b))
            x.append(channel)
            y.append(line[-3])
    return np.array(x), np.array(y)

nb_classes = 2

X, y = load_data(csv_file)
X = X.reshape(-1, 3*141*165)
X, y = shuffle(X, y, random_state=55)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
x_train = x_train.reshape(-1, 3, 141, 165)
x_test = x_test.reshape(-1, 3, 141, 165)

X_train = x_train.astype("float32")
X_test = x_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 12, 12, border_mode='full'))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(48, 32, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 48, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(80, 64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(80*8*10, 1000))
model.add(Activation('relu'))
model.add(BatchNormalization((1000,)))
model.add(Dropout(0.5))

model.add(Dense(1000, 1000))
model.add(Activation('relu'))
model.add(BatchNormalization((1000,)))
model.add(Dropout(0.5))

model.add(Dense(1000, nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adadelta()

fname = 'gender_best_weights.hdf5'
if os.path.isfile(fname):
    model.load_weights(fname)
    print "load weights successful!"
model.compile(loss='categorical_crossentropy', optimizer=opt)

print "X_train shape: {0}".format(X_train.shape)
print "y_train shape: {0}".format(Y_train.shape)
print "X_test shape: {0}".format(X_test.shape)
print "y_test shape: {0}".format(Y_test.shape)

checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
early_stop = EarlyStopping(patience=20, verbose=1)
model.fit(X_train, Y_train, batch_size=128, nb_epoch=100, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer, early_stop])

score = model.evaluate(X_test, Y_test, batch_size=128, show_accuracy=True, verbose=1)
print "Test score: {0}".format(score)

#model.save_weights("trained_weights", overwrite=True)
