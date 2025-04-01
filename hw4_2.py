import numpy as np
from scipy.integrate import quad

# 定義被積函數
def f(x):
    return x**2 * np.log(x)

# 高斯求積法實現
def gauss_quad(f, a, b, n):
    # 獲取高斯節點和權重
    t, w = np.polynomial.legendre.leggauss(n)
    # 變量替換到[a,b]
    x = 0.5*(b-a)*t + 0.5*(a+b)
    integral = 0.5*(b-a) * np.sum(w * f(x))
    return integral

# 計算
a, b = 1, 1.5
exact, _ = quad(f, a, b)
result_n3 = gauss_quad(f, a, b, 3)
result_n4 = gauss_quad(f, a, b, 4)

print(f"精確值: {exact:.10f}")
print(f"高斯求積 (n=3): {result_n3:.10f}")
print(f"高斯求積 (n=4): {result_n4:.10f}")