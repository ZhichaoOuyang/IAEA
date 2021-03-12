import numpy as np
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score,precision_score
import os
result = np.load("resultAll.npy")
result = np.array(result)
result = result.astype(int)
#print(result)
#print(len(result))
fnamek = os.path.join("NNTestData_SingleLabel_poslab.txt")
with open(fnamek) as fk:  # L1 label
    contentk = fk.readlines()
    label = [int(x.strip()) for x in contentk]
label = np.array(label)
#print(label)
#print(len(label))
accuracy = accuracy_score(label,result)
precision = precision_score(label,result)
recall = recall_score(label,result)
f1 = f1_score(label,result)
roc = roc_auc_score(label,result)

print("accuracy: ",accuracy," ,","precision: ",precision," ,","recall: ",recall," ,","f1-score: ",f1," ,","auc: ",roc)