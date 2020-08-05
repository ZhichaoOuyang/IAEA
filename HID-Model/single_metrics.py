import numpy as np
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score,precision_score
import os
result = np.load("resultSingle.npy")
result = np.array(result)
result = result.astype(int)
print(len(result))
#print(len(result))
fnamek = os.path.join("NNTestData_SingleLabel_poslab.txt")
with open(fnamek) as fk:  # L1 的标签
    contentk = fk.readlines()
    label = [int(x.strip()) for x in contentk]
label = np.array(label)
print(len(label))
#print(len(label))
accuracy = accuracy_score(label,result)
precision = precision_score(label,result)
recall = recall_score(label,result)
f1 = f1_score(label,result)
roc = roc_auc_score(label,result)  # 直接根据真实值（必须是二值）、预测值（可以是0/1,也可以是proba值）计算出auc值，中间过程的roc计算省略。

print("accuracy: ",accuracy," ,","precision: ",precision," ,","recall: ",recall," ,","f1-score: ",f1," ,","auc: ",roc)