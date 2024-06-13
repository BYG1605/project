import numpy as np
arr = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]],
    [[13, 14, 15], [16, 17, 18]]
]

# 交换前两个维度的顺序
# a = [[[arr[j][i] for j in range(len(arr))] for i in range(len(arr[0]))]]
a = np.transpose(arr.cpu().numpy(), (1, 0, 2))
print(a)