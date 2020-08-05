# coding=utf-8
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from stanfordcorenlp import StanfordCoreNLP
import string

''' Location of the dataset'''
GLOVE_DIR = 'Glove'
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def text_cleaner(text):
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()


def loadData_Tokenizer(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH):

    fnamek = os.path.join("data/tweet/data/NNModelData_SingleLabel_poslab.txt")
    # content1 = []
    # content2 = []

    with open(fnamek) as fk:   # L1 的标签
        contentk = fk.readlines()
        contentk = [x.strip() for x in contentk]
    Label = contentk
    Label = np.array(Label)
    number_of_classes_L1 = 2 #number of classes in Level 1

    np.random.seed(7)


    # 打扰数组

    x1 = np.load("NNx1_model_nopos.npy")
    x2 = np.load("NNx2_model_nopos.npy")
    x1 = np.matrix(x1, dtype=int)
    x2 = np.matrix(x2, dtype=int)
    x = np.column_stack((x1, x2))
    x = np.array(x)
    word_index = np.load('word_index_model_nopos.npy').item()
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    Label = Label[indices]
    #随机划分数据集
    X_train, X_test, y_train, y_test = train_test_split(x, Label, test_size=0.2, random_state=0)
    # 这时候X_train还没划分成向量，还是文本
    X_train1 = []
    X_train2 = []
    X_test1 = []
    X_test2 = []
    for i in range(len(X_train)):
        X_train1.append(X_train[i,:MAX_SEQUENCE_LENGTH])
        X_train2.append(X_train[i,MAX_SEQUENCE_LENGTH:])
    for i in range(len(X_test)):
        X_test1.append(X_test[i,MAX_SEQUENCE_LENGTH:])
        X_test2.append(X_test[i,MAX_SEQUENCE_LENGTH:])
    X_train1 = np.array(X_train1)
    X_train2 = np.array(X_train2)
    X_test1 = np.array(X_test1)
    X_test2 = np.array(X_test2)


    embeddings_index = {}
    '''
    For CNN and RNN, we used the text vector-space models using $100$ dimensions as described in Glove. A vector-space model is a mathematical mapping of the word space
    '''
    Glove_path = os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt')
    print(Glove_path)
    f = open(Glove_path, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            print("Warnning"+str(values)+" in" + str(line))
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return (X_train1,X_train2, y_train, X_test1,X_test2, y_test,word_index,embeddings_index,number_of_classes_L1)

