import time
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

from sklearn import svm


if __name__ == '__main__':
    print('prepare datasets...')
    EMBEDDING_DIM = 100      # Maximum sequance lenght 500 words
    MAX_NB_WORDS = 50000     # Maximum number of unique words
    time_1 = time.time()
    X_train = []
    X_test = []

    x1 = np.loadtxt("nnAllx1_bert.txt", delimiter=" ")
    x2 = np.loadtxt("nnAllx2_bert.txt", delimiter=" ")
    print(len(x1))
    print(len(x2))
    # x1 = np.array(x1)
    # x2 = np.array(x2)

    X_train1 = np.matrix(x1, dtype=float)
    X_train2 = np.matrix(x2, dtype=float)
    X_train = np.abs(X_train1 - X_train2)
    X_train = np.array(X_train)
    fnamek = os.path.join("NNAllLabel.txt")
    # content1 = []
    # content2 = []

    with open(fnamek) as fk:   # L1 的标签
        Label = fk.readlines()
        Label = [int(x.strip()) for x in Label]
    y_train = np.array(Label)


    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=None)

    time_2=time.time()
    print('Start training ...')

    clf = svm.SVC(C=0.01,kernel='rbf')  # svm class
    clf.fit(X_train, y_train)  # training the svc model
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))
    # joblib.dump(clf, "doc2vec_svm_model.m")
    # clf = joblib.load("doc2vec_svm_model.m")
    print('Start predicting...')
    test_predict=clf.predict(X_test)

    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(y_test, test_predict)
    print("The accruacy score of L1 is %f" % score)

    score = precision_score(y_test, test_predict)
    print("The precision score of L1 is %f" % score)

    score = recall_score(y_test, test_predict)
    print("The recall score of L1 is %f" % score)

    score = f1_score(y_test, test_predict)
    print("The f1 score of L1 is %f" % score)

    score = roc_auc_score(y_test,test_predict)
    print("The auc score of L1 is %f" % score)

