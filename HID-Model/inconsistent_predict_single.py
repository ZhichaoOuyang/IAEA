from keras.models import load_model
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from stanfordcorenlp import StanfordCoreNLP
import string


if __name__ == "__main__":
    model_path = 'best_model_single.h5'
    model = load_model(model_path)
    x1 = np.load("NNx1_test.npy")
    x2 = np.load("NNx1_test.npy")
    x1 = np.array(x1)
    x2 = np.array(x2)
    print(len(x1))
    print(len(x2))
    label = model.predict({'x1_input': x1, 'x2_input': x2})
    result = []
    for l in label:
        if l >= 0.5:
            result.append(1)
        else:
            result.append(0)
    result = np.array(result)
    np.save("resultSingle.npy",result)
    result = np.load("resultSingle.npy")
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