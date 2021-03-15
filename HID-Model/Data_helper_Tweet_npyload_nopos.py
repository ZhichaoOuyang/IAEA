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
GLOVE_DIR = 'data/Glove'
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

    fnamek = os.path.join("data/HID/data/NNL1_train_poslab.txt")
    fnameL2 = os.path.join("data/HID/data/NNL2_train_poslab.txt")
    # content1 = []
    # content2 = []

    with open(fnamek) as fk:   # L1 label
        contentk = fk.readlines()
        contentk = [x.strip() for x in contentk]
    with open(fnameL2) as fk:   # L2 label
        contentL2 = fk.readlines()
        contentL2 = [x.strip() for x in contentL2]
    Label = np.matrix(contentk, dtype=int)
    Label = np.transpose(Label)
    number_of_classes_L1 = np.max(Label)+1 #number of classes in Level 1

    Label_L2 = np.matrix(contentL2, dtype=int)
    Label_L2 = np.transpose(Label_L2)
    np.random.seed(7)

    Label = np.column_stack((Label, Label_L2))  # Combine L1 and L2 into a matrix
    # number of classes in Level 2 that is 1D array with size of (number of classes in level one,1)
    number_of_classes_L2 = np.zeros(number_of_classes_L1,dtype=int) #number of classes in Level 2 that is 1D array with size of (number of classes in level one,1)
    # shuffle array

    x1 = np.load("data/HID/data/NNx1_model_nopos.npy")
    x2 = np.load("data/HID/data/NNx2_model_nopos.npy")
    x1 = np.matrix(x1, dtype=int)
    x2 = np.matrix(x2, dtype=int)
    x = np.column_stack((x1, x2))
    x = np.array(x)
    word_index = np.load('data/HID/data/word_index_model_nopos.npy').item()
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    Label = Label[indices]
    # Randomly divide the data set
    X_train, X_test, y_train, y_test = train_test_split(x, Label, test_size=0.2, random_state=0)
    # At this time, X_train has not been divided into vectors, it is still text
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

    L2_Train = []
    L2_Test = []
    content1_L2_Train = []
    content2_L2_Train = []
    content1_L2_Test = []
    content2_L2_Test = []
    '''
    crewate #L1 number of train and test sample for level two of Hierarchical Deep Learning models
    '''
    for i in range(0, number_of_classes_L1):
        L2_Train.append([])
        L2_Test.append([])
        content1_L2_Train.append([])
        content2_L2_Train.append([])
        content1_L2_Test.append([])
        content2_L2_Test.append([])
        # content_L2_Train.append([])
        # content_L2_Test.append([])

        # X_train = np.array(X_train)
        # X_test= np.array(X_test)
        X_train1 = np.array(X_train1)
        X_train2 = np.array(X_train2)
        X_test1 = np.array(X_test1)
        X_test2 = np.array(X_test2)
    for i in range(0, X_train1.shape[0]):
        L2_Train[y_train[i, 0]].append(y_train[i, 1])
        number_of_classes_L2[y_train[i, 0]] = max(number_of_classes_L2[y_train[i, 0]],(y_train[i, 1]+1))
        #content_L2_Train[y_train[i, 0]].append(X_train[i])
        content1_L2_Train[y_train[i, 0]].append(X_train1[i])
        content2_L2_Train[y_train[i, 0]].append(X_train2[i])
    for i in range(0, X_test1.shape[0]):
        L2_Test[y_test[i, 0]].append(y_test[i, 1])
        content1_L2_Test[y_test[i, 0]].append(X_test1[i])
        content2_L2_Test[y_test[i, 0]].append(X_test2[i])

    for i in range(0, number_of_classes_L1):
        L2_Train[i] = np.array(L2_Train[i])
        L2_Test[i] = np.array(L2_Test[i])
        content1_L2_Train[i] = np.array(content1_L2_Train[i])
        content2_L2_Train[i] = np.array(content1_L2_Train[i])
        content1_L2_Test[i] = np.array(content1_L2_Test[i])
        content2_L2_Test[i] = np.array(content2_L2_Test[i])

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
    return (X_train1,X_train2, y_train, X_test1,X_test2, y_test, content1_L2_Train,content2_L2_Train, L2_Train, content1_L2_Test, content2_L2_Test, L2_Test, number_of_classes_L2,word_index,embeddings_index,number_of_classes_L1)
