import time


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import joblib
import os

x1 = np.loadtxt("NNTestx1_bert.txt", delimiter=" ")
x2 = np.loadtxt("NNTestx2_bert.txt", delimiter=" ")
print(len(x1))
print(len(x2))
# x1 = np.array(x1)
# x2 = np.array(x2)

X_train1 = np.matrix(x1, dtype=float)
X_train2 = np.matrix(x2, dtype=float)
X_train = np.abs(X_train1 - X_train2)
X_test = np.array(X_train)
fnamek = os.path.join("NNL1_test_poslab_3.txt")
# content1 = []
# content2 = []

with open(fnamek) as fk:  # L1 的标签
    Label = fk.readlines()
    Label = [int(x.strip()) for x in Label]
y_test = np.array(Label)

clf = joblib.load("LR1_model.m")

print('Start predicting...')
test_predict = clf.predict(X_test)

result = []
index = []
for i in range(len(test_predict)):
    if test_predict[i] >= 0.5:
        result.append(test_predict[i])
        index.append(i)
    else:
        result.append(test_predict[i])

layer2_x1 = []
layer2_x2 = []
for i in index:
    layer2_x1.append(x1[i])
    layer2_x2.append(x1[i])


# x1 = np.array(x1)
# x2 = np.array(x2)
layer2_x1 = np.matrix(layer2_x1, dtype=float)
layer2_x1 = np.matrix(layer2_x1, dtype=float)
X_test = np.abs(layer2_x1 - layer2_x2)
X_test = np.array(X_test)
fnamek = os.path.join("NNTestData_SingleLabel_poslab.txt")
# content1 = []
# content2 = []

with open(fnamek) as fk:  # L1 的标签
    Label = fk.readlines()
    Label = [int(x.strip()) for x in Label]
y_test = np.array(Label)

clf = joblib.load("LR2_model.m")

print('Start predicting...')
label2 = clf.predict(X_test)


for i in range(len(label2)):
    xiabiao = index[i]
    print(xiabiao)
    if label2[i] >= 0.5:
        result[xiabiao] = 1
    else:
        result[xiabiao] = 0

result = np.array(result)

score = accuracy_score(y_test, result)
print("The accruacy score of L1 is %f" % score)

score = precision_score(y_test, result)
print("The precision score of L1 is %f" % score)

score = recall_score(y_test, result)
print("The recall score of L1 is %f" % score)

score = f1_score(y_test, result)
print("The f1 score of L1 is %f" % score)

score = roc_auc_score(y_test, result)
print("The auc score of L1 is %f" % score)