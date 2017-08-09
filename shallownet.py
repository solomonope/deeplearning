import numpy as NP;
import os;

#os.environ['KERAS_BACKEND'] = 'theano';
import keras
from keras.datasets import mnist;
from keras.models import Sequential;
from keras.layers import Dense;
from keras.optimizers import SGD;

NP.random.seed(42);

(X_train, y_train), (X_test, y_test) = mnist.load_data();

X_train = X_train.reshape(60000, 784).astype("float32");
X_test = X_test.reshape(10000, 784).astype("float32");

X_train /= 255;
X_test /= 255;

n_classes = 10;

y_train = keras.utils.to_categorical(y_train, num_classes=n_classes);
y_test = keras.utils.to_categorical(y_test, num_classes=n_classes);

model = Sequential();

model.add(Dense((64), activation='relu', input_shape=(784,)));
model.add(Dense((10), activation="softmax"));

model.summary();

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.001), metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=200, verbose=1, validation_data=(X_test, y_test))

#print(model)

