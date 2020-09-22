import numpy as np
import os
import string
fnamek2 = "../data/tweet/data/NNModelData.txt"
with open(fnamek2) as fk:  # L1 的标签
    text = fk.readlines()
    #y = [int(x.strip()) for x in y]
text = np.array(text)
data = []
for t in text:
    t = t.split(' ouyang ')
    data.append(t[0]+"\n")
    data.append(t[1])
print(len(data))
file = open('NNModel.txt',mode='w')
file.writelines(data)
file.close()