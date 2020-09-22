import numpy as np
import os
import string
label1 = "NNL1_train_poslab.txt"
label2 = "NNL2_train_poslab.txt"

labels1 = []
labels2 = []
with open(label1) as l1:
    for line in l1:
        labels1.append(int(line.strip()))

with open(label2) as l2:
    for line in l2:
        labels2.append(int(line.strip()))
data = []
for i in range(len(labels1)):
    if labels1[i] == 1:   # 第一层是1
        data.append(str(labels2[i]) + '\n')

print(len(data))

print(len(data))
file = open('layer2TrainLabel.txt',mode='w')
file.writelines(data)
file.close()