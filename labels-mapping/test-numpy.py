import numpy as np

# 创建一个numpy二维数组
arr = np.array([[-1, 2, 3], [4, -1, 6], [7, 8, 9], [-1, 99, 88]])
print(arr)

# 遍历二维数组中的每一行
for row in arr:
    row[0] += 1
print(arr)

arr = np.delete(arr, np.where(arr[:,0]==0), axis=0)

print(arr)
