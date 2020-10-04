import numpy as np
import os
import string
import json
from random import shuffle
fnamek = "NNModelData.txt"

with open(fnamek) as fk:  # L1 的标签
    text = fk.readlines()
    #y = [int(x.strip()) for x in y]
text = np.array(text)
data = []
for t in text:
    t = t.replace(' ouyang', '').strip()
    t = t.split(' ')
    data.append(t)

label = []
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

for i in range(len(labels1)):
    if labels1[i] == 0:
        label.append('0')

    if labels1[i] == 1:   # 第一层是1
        if labels2[i] == 0:
            label.append('1--2')
        if labels2[i] == 1:
            label.append('1--3')

print(len(label))

result = []
strs = []
for i in range(len(data)):
    obj = {
        'doc_label': [label[i]],
        'doc_token': data[i],
        'doc_keyword': [],
        'doc_topic': [],
    }
    result.append(obj)
    json_str = json.dumps(obj)
    strs.append(json_str)
shuffle(strs)

file = open('tecentTrainData.json', mode='w')
for line in strs:
    file.write(line + '\n')

file.close()