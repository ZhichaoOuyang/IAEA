import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from stanfordcorenlp import StanfordCoreNLP
import string


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def loadData_Tokenizer(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH):
    # nlp = StanfordCoreNLP(r'data/tweet/stanford-corenlp-full-2018-10-05')
    # trantab = str.maketrans({key: None for key in string.punctuation})  # 删除字符串
    fname = os.path.join("data/tweet/data/NNTestData.txt")

    # content1 = []
    # content2 = []
    with open(fname) as f:   # 得到content，里面是一对推文集
        content = f.readlines()
        content = [clean_str(x) for x in content]

    content = np.array(content)

    X_1 = []
    X_2 = []

    for x in content:
        X_1.append(x.split("ouyang")[0].strip())
        X_2.append(x.split("ouyang")[1].strip())

    # 处理成文本
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    # delete "ouyang"
    # dContent = []
    # for c in content:
    #     c = c.replace('ouyang','')
    #     dContent.append(c)
    # dContent = np.array(dContent)
    # tokenizer.fit_on_texts(dContent)
    pos_x1 = []
    pos_x2 = []

    #pos_content = []
    flag1 = 0
    print("begin for1")

    for train1 in X_1:
        # train1 = str(nlp.pos_tag(train1))
        # train1 = train1.translate(trantab)
        pos_x1.append(train1)
        # pos_content.append(train1)
        flag1 += 1
        if flag1 % 10000 == 0:
            print("x1 =" , flag1)
    flag2 = 0
    print("begin for2")

    for train2 in X_2:
        # train2 = str(nlp.pos_tag(train2))
        # train2 = train2.translate(trantab)
        pos_x2.append(train2)
        # pos_content.append(train2)
        flag2 += 1
        if flag2 % 10000 == 0:
            print("x2 =" , flag2)
    print("POS end")
    pos_content = np.load("tokenizer_content_nopos.npy")
    tokenizer.fit_on_texts(pos_content)
    # np.save("tokenizer_content_layer2test.npy",pos_content)
    # 转成成向量
    sequences_train1 = tokenizer.texts_to_sequences(pos_x1)
    sequences_train2 = tokenizer.texts_to_sequences(pos_x2)

    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))
    # 填充序列

    X_1 = pad_sequences(sequences_train1, maxlen=MAX_SEQUENCE_LENGTH)
    X_2 = pad_sequences(sequences_train2, maxlen=MAX_SEQUENCE_LENGTH)
    # print(X_1[0])
    np.save("NNx1_test_nopos.npy", X_1)
    # a = np.load("x1.npy")

    np.save("NNx2_test_nopos.npy", X_2)
    # b = np.load("x2.npy")
    # np.save("word_index_layer2test.npy",word_index)
    print("save")

    # x1 = np.matrix(a, dtype=int)
    # x2 = np.matrix(b, dtype=int)
    # x = np.column_stack((x1, x2))  # 把L1和L2合成一个矩阵
    # #print(x1.shape)
    # print(x.shape)
    # x1 = np.array(x1)
    # x2 = np.array(x2)
    # x = np.array(x)
    # print(x1[0])
    # print(x2[0])
    # print(x[0,:MAX_SEQUENCE_LENGTH])
    # x1 = np.array(x1)
    # indices = np.arange(len(x1))
    # np.random.shuffle(indices)
    # x1 = x1[indices]
    # print(x1[0])
    # print('..')
    # print(len(a[999]))
    # print('...')
    # print(b[0])




if __name__ == "__main__":
    MAX_SEQUENCE_LENGTH = 50      # Maximum sequance lenght 500 words
    MAX_NB_WORDS = 55000     # Maximum number of unique words
    loadData_Tokenizer(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
