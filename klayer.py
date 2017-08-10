import keras.layers as Layers;

denseLayer =  Layers.Dense(32);
config =  denseLayer.get_config();
config;