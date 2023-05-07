import numpy as np
import matplotlib.pyplot as plt

def fBm(H, N):
    # 生成正态分布随机数
    Z = np.random.normal(0, 1, N)
    # 初始条件
    X = np.zeros(N)
    X[0] = Z[0]
    # 递推计算
    for i in range(1, N):
        X[i] = 0.5 * (Z[i] + Z[i-1]) * pow((i+1), H)
    # 计算均值和标准差
    mean = np.mean(X)
    std = np.std(X)
    # 归一化
    X = (X - mean) / std
    return X

H = 0.6 # 赫斯特指数
N = 1000 # 时间序列长度
M = 10 # 生成路径数

for i in range(M):
    Y = fBm(H, N)
    plt.plot(Y)
plt.show()