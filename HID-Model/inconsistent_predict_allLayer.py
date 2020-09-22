from keras.models import load_model
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from stanfordcorenlp import StanfordCoreNLP
import string
import keras


if __name__ == "__main__":
    model_path = 'best_model_1_50_nopos_sub_8m10.h5'
    model = load_model(model_path, custom_objects={'keras': keras})
    x1 = np.load("NNx1_test_nopos.npy")
    x2 = np.load("NNx2_test_nopos.npy")
    x1 = np.array(x1)
    x2 = np.array(x2)
    print(len(x1))
    print(len(x2))
    label = model.predict({'x1_input': x1, 'x2_input': x2})
    result = []
    index = []
    for i in range(len(label)):
        if label[i] >= 0.5:
            result.append(label[i])
            index.append(i)
            print(i)
        else:
            result.append(label[i])

    layer2_x1 = []
    layer2_x2 = []
    for i in index:
        layer2_x1.append(x1[i])
        layer2_x2.append(x2[i])
    layer2_x1 = np.array(layer2_x1)
    layer2_x2 = np.array(layer2_x2)
    print(len(layer2_x1))
    model_path2 = 'best_model_22_50_nopos_sub_8m10.h5'
    model2 = load_model(model_path2, custom_objects={'keras': keras})
    label2 = model2.predict({'x1_input': layer2_x1, 'x2_input': layer2_x2})
    for i in range(len(label2)):
        xiabiao = index[i]
        print(xiabiao)
        if label2[i] >= 0.5:
            result[xiabiao] = 1
        else:
            result[xiabiao] = 0

    result = np.array(result)
    np.save("result_all_nopos_sub_8m10.npy",result)
    result = np.load("result_all_nopos_sub_8m10.npy")
    result = np.array(result)
    flag = 0
    flag2 = 0
    for r in result:
        if r == 0:
            flag += 1
        else:
            flag2 += 1
    #print(result[0])
    print(flag)
    print(flag2)
    #print(len(label))