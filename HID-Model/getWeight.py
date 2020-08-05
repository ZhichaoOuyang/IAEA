from keras.models import load_model
import numpy as np

model_path1 = 'best_model_1_100_nopos.h5'
model1 = load_model(model_path1)
weight_Dense_1,bias_Dense_1 = model1.get_layer('dense_3').get_weights()
print(weight_Dense_1.shape)
print(bias_Dense_1.shape)
np.save("relevant_wight_nopos.npy",weight_Dense_1)
np.save("relevant_bias_nopos.npy",bias_Dense_1)


model_path1 = 'best_model_22_100_nopos.h5'
model1 = load_model(model_path1)
weight_Dense_2,bias_Dense_2 = model1.get_layer('dense_6').get_weights()
print(weight_Dense_2.shape)
print(bias_Dense_2.shape)
np.save("inconsistent_wight_nopos.npy",weight_Dense_2)
np.save("inconsistent_bias_nopos.npy",bias_Dense_2)