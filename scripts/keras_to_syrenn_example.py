"""Example of converting a sequential Keras model to SyReNN.
"""
import tensorflow.keras as keras
import numpy as np
from keras_to_syrenn import keras_to_syrenn
import pysyrenn

# https://keras.io/guides/sequential_model/
model = keras.Sequential(
    [
        keras.layers.Dense(2, activation="relu", name="layer1"),
        keras.layers.Dense(3, activation=None, name="layer2"),
        keras.layers.Dense(4, name="layer3"),
    ]
)
# We need to evaluate the model at least once before converting to SyReNN. I
# think this is because Keras doesn't actually initialize the parameters until
# this.
model(np.ones((1, 3)))

syrenn_network = keras_to_syrenn(model)

x = np.ones((3, 3))

print("Keras model output:")
print(model(x))

print("SyReNN network output:")
print(syrenn_network.compute(x))
