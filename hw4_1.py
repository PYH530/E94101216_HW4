import numpy as np
from scipy.integrate import quad  # 引入 SciPy 用於計算精確積分

# 定義被積函數 f(x) = e^x * sin(4x)
def f(x):
    return np.exp(x) * np.sin(4 * x)

# 設定積分區間 [a, b] 和步長 h
a, b = 1, 2  # 積分範圍從 1 到 2
h = 0.1  # 步長
n = int((b - a) / h)  # 計算區間內的分割點數量

# 1. 梯形法則
def composite_trapezoidal(f, a, b, n):
    h = (b - a) / n  # 計算每個小區間的寬度
    x = np.linspace(a, b, n+1)  # 產生等距的 x 值
    y = f(x)  # 計算對應的函數值
    integral = h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))  # 使用梯形法則計算積分
    return integral

# 2. 辛普森法則
def composite_simpson(f, a, b, n):
    if n % 2 != 0:  
        n += 1  # 辛普森法則要求 n 必須是偶數
    h = (b - a) / n  # 計算步長
    x = np.linspace(a, b, n+1)  # 產生等距的 x 值
    y = f(x)  # 計算函數值
    integral = h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))  # 套用辛普森法則公式
    return integral

# 3. 中點法則
def composite_midpoint(f, a, b, n):
    h = (b - a) / n  # 計算步長
    x_mid = np.linspace(a + h/2, b - h/2, n)  # 產生區間中點
    y_mid = f(x_mid)  # 計算對應的函數值
    integral = h * np.sum(y_mid)  # 套用中點法則計算積分
    return integral

# 計算積分結果
trap_result = composite_trapezoidal(f, a, b, n)  # 梯形法結果
simp_result = composite_simpson(f, a, b, n)  # 辛普森法結果
mid_result = composite_midpoint(f, a, b, n)  # 中點法結果

# 輸出數值積分結果
print(f"梯形法則結果: {trap_result:.6f}")
print(f"辛普森法結果: {simp_result:.6f}")
print(f"中點法結果: {mid_result:.6f}")

# 使用 SciPy 計算精確積分值作為參考
exact_result, _ = quad(f, a, b)
print(f"精確值結果: {exact_result:.6f}")
