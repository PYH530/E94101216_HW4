import numpy as np
from scipy import integrate

# ====================== 核心配置 ======================
def f(y, x):
    """严格匹配的被积函数：2y*sinx + cos²x"""
    return 2*y*np.sin(x) + np.cos(x)**2

x_lower, x_upper = 0, np.pi/4
y_lower = lambda x: np.sin(x)
y_upper = lambda x: np.cos(x)

# ====================== 1. 理论精确值 ======================
def theoretical_value():
    """手工推导验证值：(5√2)/6 - 2/3"""
    return (5*np.sqrt(2))/6 - 2/3

# ====================== 2. Scipy验证 ======================
def scipy_compute():
    result, _ = integrate.dblquad(f, x_lower, x_upper, y_lower, y_upper)
    return result

# ====================== 3. 精确Simpson实现 ======================
def simpson_accurate(n=4, m=4):
    hx = (x_upper - x_lower)/n
    integral = 0
    
    for i in range(n+1):
        x = x_lower + i*hx
        y_low, y_high = y_lower(x), y_upper(x)
        hy = (y_high - y_low)/m
        
        # y方向积分
        sum_y = 0
        for j in range(m+1):
            y = y_low + j*hy
            weight = 4 if j%2 else 2
            weight = 1 if j in (0,m) else weight
            sum_y += weight * f(y, x)
        
        # x方向权重
        wx = 4 if i%2 else 2
        wx = 1 if i in (0,n) else wx
        integral += wx * (sum_y * hy/3)
    
    return integral * hx/3

# ====================== 4. 精确高斯实现 ======================
def gauss_accurate(n=3, m=3):
    # 获取高斯点
    x_pts, x_wts = np.polynomial.legendre.leggauss(n)
    y_pts, y_wts = np.polynomial.legendre.leggauss(m)
    
    total = 0
    for i in range(n):
        # x方向映射
        x = 0.5*(x_pts[i] + 1)*(x_upper - x_lower) + x_lower
        y_low, y_high = y_lower(x), y_upper(x)
        
        # y方向积分
        sum_y = 0
        for j in range(m):
            y = 0.5*(y_pts[j] + 1)*(y_high - y_low) + y_low
            sum_y += y_wts[j] * f(y, x)
        
        total += x_wts[i] * sum_y * 0.5*(y_high - y_low)
    
    return total * 0.5*(x_upper - x_lower)

# ====================== 执行验证 ======================
if __name__ == "__main__":
    exact = theoretical_value()
    scipy_val = scipy_compute()
    simp_val = simpson_accurate(4,4)
    gauss_val = gauss_accurate(3,3)
    
    print("方法".ljust(20), "结果值".center(25), "绝对误差".rjust(15))
    print("-"*60)
    print(f"{'理论精确值':<20}{exact:^25.15f}{'0':>15}")
    print(f"{'Scipy':<20}{scipy_val:^25.15f}{abs(scipy_val-exact):>15.2e}")
    print(f"{'Simpson 4x4':<20}{simp_val:^25.15f}{abs(simp_val-exact):>15.2e}")
    print(f"{'Gauss 3x3':<20}{gauss_val:^25.15f}{abs(gauss_val-exact):>15.2e}")
