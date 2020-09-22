import os
import numpy as np
fname = os.path.join("NNTest_bert.txt")
with open(fname) as f:  # 得到content，里面是一对推文集
    content = f.readlines()
    content = [x.strip() for x in content]
content = np.array(content)
x1 = []
x2 = []
for i in range(len(content)):
    if i%2 == 0:
        x1.append(content[i])
    else:
        x2.append(content[i])

x1 = np.array(x1)
x2 = np.array(x2)
print(len(x1))
print(len(x2))
np.savetxt("NNTestx1_bert.txt",x1,fmt="%s",delimiter=",")
np.savetxt("NNTestx2_bert.txt",x2,fmt="%s",delimiter=",")
print('success')
# np.save("x1data9w_baseline.npy",x1)
# np.save("x2data9w_baseline.npy",x2)