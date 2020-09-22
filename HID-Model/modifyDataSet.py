f = open('data/tweet/data/NNTestData_SingleLabel_poslab.txt', 'a')
for i in range(0, 8000):
    f.write('0' + '\n')
f.close()
print('success')