import keras as Keras;
import keras.models as Models;
import keras.layers as Layer;
import keras.layers.normalization as Normalization
import tflearn.datasets.oxflower17 as oxflower17
import keras.callbacks as Callbacks;

X, Y = oxflower17.load_data(one_hot=True);

model = Models.Sequential();

model.add(Layer.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape=(224, 224, 3)));
model.add(Layer.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Normalization.BatchNormalization());

model.add(Layer.Conv2D(256, kernel_size=(5, 5), activation="relu"));
model.add(Layer.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Normalization.BatchNormalization());

model.add(Layer.Conv2D(256, kernel_size=(3, 3), activation="relu"));
model.add(Layer.Conv2D(384, kernel_size=(3, 3), activation="relu"));
model.add(Layer.Conv2D(384, kernel_size=(3, 3), activation="relu"));

model.add(Layer.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Normalization.BatchNormalization());

model.add(Layer.Flatten());
model.add(Layer.Dense(4096, activation="tanh"));
model.add(Layer.Dropout(0.5));
model.add(Layer.Dense(4096, activation="tanh"));
model.add(Layer.Dropout(0.5))

model.add(Layer.Dense(17, activation="softmax"));
model.summary();

tensorBoard =  Callbacks.TensorBoard("./logs/alexnet")
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

model.fit(X, Y, batch_size=64, epochs=1, verbose=1, validation_split=0.1, shuffle=True);
