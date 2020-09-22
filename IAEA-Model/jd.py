arr = []
s = int(input())
for i in range(s):
    op = input().strip().split(' ')
    if op[0] == '1':
        arr.insert(int(op[1])-1, int(op[2]))
    if op[0] == '2':
        del arr[int(op[1])]
    if op[0] == '3':
        print(' '.join(map(str, arr)))

