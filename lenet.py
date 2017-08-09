import keras
import keras.datasets as DataSets;
import keras.models as Models;
import keras.layers as Layers;
import keras.optimizers as Optimizers;
import numpy as NP;

NP.random.seed(42)

NP.random.seed(42)

(X_train, y_train), (X_test, y_test) = DataSets.mnist.load_data();

X_train = X_train.reshape(60000, 28,28,1).astype("float32");
X_test = X_test.reshape(10000, 28,28,1).astype("float32");

X_train /= 255;
X_test /= 255;

n_classes = 10;

y_train = keras.utils.to_categorical(y_train, num_classes=n_classes);
y_test = keras.utils.to_categorical(y_test, num_classes=n_classes);

model = Models.Sequential();

model.add(Layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28, 1)));
model.add(Layers.Conv2D(64, kernel_size=(3,3), activation ="relu"));
model.add(Layers.MaxPool2D(pool_size=(2,2)))
model.add(Layers.Dropout(0.25));
model.add(Layers.Flatten());
model.add(Layers.Dense(128, activation="relu"))
model.add(Layers.Dropout(0.5));
model.add(Layers.Dense(n_classes, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, y_test))