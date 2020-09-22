import time


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import joblib
import os

if __name__ == '__main__':
    print('prepare datasets...')
    EMBEDDING_DIM = 100      # Maximum sequance lenght 500 words
    MAX_NB_WORDS = 50000     # Maximum number of unique words
    time_1 = time.time()
    X_train = []
    X_test = []

    x1 = np.loadtxt("NNModelx1_bert.txt", delimiter=" ")
    x2 = np.loadtxt("NNModelx2_bert.txt", delimiter=" ")
    print(len(x1))
    print(len(x2))
    # x1 = np.array(x1)
    # x2 = np.array(x2)

    X_train1 = np.matrix(x1, dtype=float)
    X_train2 = np.matrix(x2, dtype=float)
    X_train = np.abs(X_train1 - X_train2)
    X_train = np.array(X_train)
    fnamek = os.path.join("NNL1_train_poslab.txt")
    # content1 = []
    # content2 = []

    with open(fnamek) as fk:   # L1 的标签
        Label = fk.readlines()
        Label = [int(x.strip()) for x in Label]
    y_train = np.array(Label)


    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=None)

    time_2=time.time()
    print('read data cost %f seconds' % (time_2 - time_1))
    print('Start training ...')
    # print('Start training...')
    # # multi_class可选‘ovr’, ‘multinomial’，默认为ovr用于二类分类，multinomial用于多类分类
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100,100), random_state=1, activation='tanh')  # svm class
    clf.fit(X_train, y_train)
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))
    joblib.dump(clf, "MLP1_model.m")



    x1 = np.loadtxt("layer2Modelx1_bert.txt", delimiter=" ")
    x2 = np.loadtxt("layer2Modelx2_bert.txt", delimiter=" ")
    print(len(x1))
    print(len(x2))
    # x1 = np.array(x1)
    # x2 = np.array(x2)

    X_train1 = np.matrix(x1, dtype=float)
    X_train2 = np.matrix(x2, dtype=float)
    X_train = np.abs(X_train1 - X_train2)
    X_train = np.array(X_train)
    fnamek = os.path.join("layer2TrainLabel.txt")
    # content1 = []
    # content2 = []

    with open(fnamek) as fk:   # L1 的标签
        Label = fk.readlines()
        Label = [int(x.strip()) for x in Label]
    y_train = np.array(Label)


    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=None)

    time_2=time.time()
    print('read data cost %f seconds' % (time_2 - time_1))
    print('Start training ...')
    # print('Start training...')
    # # multi_class可选‘ovr’, ‘multinomial’，默认为ovr用于二类分类，multinomial用于多类分类
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100,100), random_state=1, activation='tanh')  # svm class
    clf.fit(X_train, y_train)
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))
    joblib.dump(clf, "MLP2_model.m")


    # clf = joblib.load("train_model.m")
    print('Start predicting...')
    test_predict = clf.predict(X_test)
    # print(test_predict.T)
    # predict = np.array(test_predict.T)
    # print(predict)
    # np.save("predicttestsub.npy", predict)
    # output = pd.DataFrame(test_predict.T,columns=['sort'])
    # output.to_csv('predictiontestsub.csv')     #3 and 5 is useful
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))


    # score = accuracy_score(y_test, test_predict)
    # print("The accruacy score of L1 is %f" % score)
    #
    # score = precision_score(y_test, test_predict)
    # print("The precision score of L1 is %f" % score)
    #
    # score = recall_score(y_test, test_predict)
    # print("The recall score of L1 is %f" % score)
    #
    # score = f1_score(y_test, test_predict)
    # print("The f1 score of L1 is %f" % score)
    #
    # score = roc_auc_score(y_test,test_predict)
    # print("The auc score of L1 is %f" % score)

