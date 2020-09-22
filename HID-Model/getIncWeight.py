from keras.models import load_model
import keras
import numpy as np

model_path = 'best_model_22_50_nopos_sub_8m6.h5'
model = load_model(model_path, custom_objects={'keras': keras})
weight_Dense_1,bias_Dense_1 = model.get_layer('dense_6').get_weights()

print(weight_Dense_1.shape)
print(bias_Dense_1.shape)
np.save("inconsistent_weight_nopos_8m6.npy",weight_Dense_1)
np.save("inconsistent_bias_nopos_8m6.npy",bias_Dense_1)

weight = np.load("inconsistent_weight_nopos_8m6.npy")
print(weight.shape)

bias = np.load("inconsistent_bias_nopos_8m6.npy")
print(bias.shape)
