import numpy as np
import cvxpy as cp
import math
import sys
import os


# 生成服从高斯分布的噪声矩阵
#参数:
#    shape (tuple): 噪声矩阵的形状，例如 (m, n)
#    mean (float): 高斯分布的均值，默认为 0.0
#    std (float): 高斯分布的标准差，默认为 1.0
def generate_gaussian_noise(shape, mean=0.0, std=1.0):
    E = np.random.normal(loc=mean, scale=std, size=shape)
    return E

# 生成一个 x 的高斯噪声矩阵，均值为 0，标准差为 1

# 设置随机种子以确保结果可重复
np.random.seed(100)

def generate_binary_matrix(rows, cols):
    # 生成一个服从标准正态分布的矩阵
    gaussian_matrix = np.random.randn(rows, cols)
    
    # 将矩阵的中位数作为阈值，低于中位数的设为0，高于中位数的设为1
    median = np.median(gaussian_matrix)
    binary_matrix = (gaussian_matrix > median).astype(int)
    
    return binary_matrix


#这里要用X(true) = LL^{T}替换
# 定义矩阵的维度
d = int(input("请输入L的d :"))  # d 很小
r = int(input("请输入L的r :"))  # r 远小于 d

# loc 是均值，scale 是标准差
mean = 0  # 高斯分布的均值
std_dev = 1  # 高斯分布的标准差
# 生成满足高斯分布的 d x r 矩阵
L = np.random.normal(loc=mean, scale=std_dev, size=(d, r))
np.random.seed(200)



matrix_X = L @ L.T


p = int(input("请输入A和B的p(行数), 也是error矩阵的维度 :"))
d = int(input("请输入A和B的d(列数) :"))
matrix_a = generate_binary_matrix(p, d)
matrix_A = np.array(matrix_a)
matrix_b = generate_binary_matrix(p, d)
matrix_B = np.array(matrix_b)

noise_matrix_custom = 0.1 * generate_gaussian_noise((p, p))

matrix_y = matrix_A @ matrix_X @ matrix_B.T + noise_matrix_custom
matrix_Y = np.array(matrix_y)

# --- Nuclear norm 恢复部分 ---
X_hat = cp.Variable((d, d), symmetric=True)
_lambda = 1.0

objective = cp.Minimize(0.5 * cp.norm(matrix_Y - matrix_A @ X_hat @ matrix_B.T, 'fro')**2 + _lambda * cp.normNuc(X_hat))
problem = cp.Problem(objective)
problem.solve(solver=cp.SCS)

X_estimated = X_hat.value


error = np.linalg.norm(matrix_X - X_estimated, 'fro')

Xfro = np.linalg.norm(matrix_X)
fraction = error / Xfro

# 设置输出格式
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)  # 保留 4 位小数，禁止科学计数法

# 将矩阵转换为符合 Jupyter Notebook 格式的字符串
def format_matrix(matrix):
    # 将矩阵转换为字符串，并去掉换行符
    matrix_str = np.array2string(matrix, separator=', ', prefix='    ')
    # 去掉多余的空格和换行符
    matrix_str = matrix_str.replace('\n', '').replace('  ', ' ')
    return matrix_str

# 输出到文件
with open('Nuclear_error_and_fraction', 'w') as f:
    sys.stdout = f
    print("\nError:")
    print(error)
    print("\nFraction:")
    print(fraction)
    sys.stdout = sys.__stdout__  # 恢复标准输出
