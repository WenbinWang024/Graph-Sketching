import numpy as np
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




# 生成 5x10 和 10x5 的二值矩阵
p = int(input("请输入A和B的p(行数), 也是error矩阵的维度 :"))
d = int(input("请输入A和B的d(列数) :"))
matrix_a = generate_binary_matrix(p, d)
matrix_A = np.array(matrix_a)
matrix_b = generate_binary_matrix(p, d)
matrix_B = np.array(matrix_b)

noise_matrix_custom = 0.1 * generate_gaussian_noise((p, p))

matrix_y = matrix_A @ matrix_X @ matrix_B.T + noise_matrix_custom
matrix_Y = np.array(matrix_y)

# 输入算法参数
b = float(input("请输入 b :"))
Lambda0 = float(input("请输入 Lambda0 :"))
LambdaTarget = float(input("请输入 LambdaTarget :"))
EpsilonOptional = float(input("请输入 EpsilonOptional :"))
LMin = float(input("请输入 LMin :"))
eta = float(input("请输入 eta :"))
delta = float(input("请输入 delta :"))

# 计算 K
K = math.floor(math.log(Lambda0 / LambdaTarget) / math.log(1 / eta))

# 初始化参数列表
LAMBDAGROUP = [None for _ in range(K + 1)]
XGROUP = [None for _ in range(K + 2)]
EPSILONGROUP = [None for _ in range(K + 1)]
LGROUP = [LMin for _ in range(K + 2)]  # 初始化为 LMin
NGROUP = [None for _ in range(K + 1)]

# 初始化第一个元素
LAMBDAGROUP[0] = Lambda0
LGROUP[0] = LMin
XGROUP[0] = matrix_X

def singular_value_thresholding(X, threshold):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S = np.maximum(S - threshold, 0)
    return U @ np.diag(S) @ Vt

def compute_phi_lambda(X, lambda_val):
    # 计算损失函数
    residual = matrix_Y - matrix_A @ X @ matrix_B.T
    loss = 0.5 * np.linalg.norm(residual, 'fro') ** 2
    
    # 计算核范数
    nuclear_norm = np.linalg.norm(X, 'nuc')
    
    # 计算 Q_lambda
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Q_lambda = 0
    for sigma in S:
        if abs(sigma) <= b * lambda_val:
            Q_lambda += -sigma ** 2 / (2 * b)
        else:
            Q_lambda += (b * lambda_val ** 2 / 2) - lambda_val * abs(sigma)
    
    # 计算正则化项
    reg = lambda_val * nuclear_norm + Q_lambda
    
    return float(loss + reg)

def compute_psi_L_lambda(X, M, L, lambda_val):
    global b
    # 检查 L 的值
    if not np.isfinite(L) or L <= 0:
        L = LMin  # 如果 L 无效，重置为 LMin
    
    # 计算 X - M
    diff = X - M
    
    # 检查 X - M 的值
    if not np.all(np.isfinite(diff)):
        diff = np.zeros_like(X)  # 如果 X - M 包含无效值，重置为 0
    
    # 计算损失函数的局部模型
    residual = matrix_Y - matrix_A @ M @ matrix_B.T
    quadratic = 0.5 * L * np.linalg.norm(diff, 'fro') ** 2##第三项
    
    # 计算核范数
    nuclear_norm = np.linalg.norm(X, 'nuc')##第四项

    
    # 计算 Q_lambda
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Q_lambda = 0
    for sigma in S:
        if abs(sigma) <= b * lambda_val:
            Q_lambda += -sigma ** 2 / (2 * b)
        else:
            Q_lambda += (b * lambda_val ** 2 / 2) - lambda_val * abs(sigma)
    grad = compute_grad_L_n_lambda_tilde(M, lambda_val)
    inner_product = np.sum(grad * diff)  # 保证是标量计算
    
    return float(0.5 * np.linalg.norm(residual, 'fro') ** 2 + Q_lambda + quadratic + lambda_val * nuclear_norm + inner_product)##这里要加的是L_n_lambda_tilde的梯度

def compute_omega_lambda(X, lambda_val):
    grad_L_tilde = compute_grad_L_n_lambda_tilde(X, lambda_val)
    Upsilon_prime = compute_subgradient_nuclear_norm(X)
    omega = np.linalg.norm(grad_L_tilde + lambda_val * Upsilon_prime, ord=2)
    return float(omega)

def compute_grad_L_n_lambda_tilde(X, lambda_val):
    # 计算损失函数的梯度
    global b
    residual = matrix_Y - matrix_A @ X @ matrix_B.T
    grad_L_n = -matrix_A.T @ residual @ matrix_B
    
    # 计算 Q_lambda 的梯度
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    grad_Q_lambda = np.zeros_like(X)
    for i, sigma in enumerate(S):
        if abs(sigma) <= b * lambda_val:
            grad_Q_lambda += (-sigma / b) * np.outer(U[:, i], Vt[i, :])
        else:
            grad_Q_lambda += (-lambda_val * np.sign(sigma)) * np.outer(U[:, i], Vt[i, :])
    
    # 计算 ∇L̃_n,λ(X) = ∇L_n(X) + ∇Q_λ(X)
    grad_L_tilde = grad_L_n + grad_Q_lambda
    
    return grad_L_tilde

def compute_subgradient_nuclear_norm(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Upsilon_prime = U @ Vt
    return Upsilon_prime

def LineSearch(lambda_now, X_now, L_now, index):
    phi = compute_phi_lambda(X_now, lambda_now)
    psi = compute_psi_L_lambda(X_now, X_now, L_now, lambda_now)
    
    while phi > psi:
        X_now = singular_value_thresholding(X_now, lambda_now / (2 * L_now))
        phi = compute_phi_lambda(X_now, lambda_now)
        psi = compute_psi_L_lambda(X_now, X_now, L_now, lambda_now)
        if phi > psi:
            L_now = 2 * L_now
    
    XGROUP[index] = X_now
    return X_now, L_now

def ProxGrad(lambda_now, epsilon_now, X_now, L_now, index):
    global LMin
    k = 0
    omega = compute_omega_lambda(X_now, lambda_now)
    judge = 100
    
    while judge > epsilon_now:
        k += 1
        X_next, L_now = LineSearch(lambda_now, X_now, L_now, index)
        L_now = max(LMin, L_now / 2)
        LGROUP[index] = L_now  # 更新 LGROUP
        LMin = min([x for x in LGROUP if x is not None])  # 过滤掉 None 值
        omega = compute_omega_lambda(X_now, lambda_now)
        judge = np.linalg.norm(X_next - X_now, 'fro')
        X_now = X_next


    return X_now, L_now

# 主循环
for t in range(K):
    LAMBDAGROUP[t + 1] = eta * LAMBDAGROUP[t]
    EPSILONGROUP[t + 1] = LAMBDAGROUP[t] / 4
    XGROUP[t + 1], LGROUP[t + 1] = ProxGrad(LAMBDAGROUP[t + 1], EPSILONGROUP[t + 1], XGROUP[t], LGROUP[t], t + 1)
LGROUP = [LMin for _ in LGROUP]

# 最终步骤
XGROUP[K + 1], LGROUP[K + 1] = ProxGrad(LambdaTarget, EpsilonOptional, XGROUP[K], LGROUP[K], K + 1)

error = np.linalg.norm(matrix_X - XGROUP[K + 1], 'fro')

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
with open('sss', 'w') as f:
    sys.stdout = f
    print(format_matrix(XGROUP[K + 1]))
    sys.stdout = sys.__stdout__  # 恢复标准输出

