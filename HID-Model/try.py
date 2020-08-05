import tensorflow as tf
import numpy as np
import keras

inputs = tf.ones([2, 20])
a = keras.layers.Dense(50, activation='sigmoid')(inputs)
print(a)

