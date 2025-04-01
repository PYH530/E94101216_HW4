import numpy as np
from scipy.integrate import quad

def composite_simpsons(f, a, b, n):
    """
    Composite Simpson's rule for numerical integration
    with handling of endpoint singularities
    """
    if n % 2 != 0:
        n += 1  # Make sure n is even
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    
    # Handle potential singularities at endpoints
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] == a and not np.isfinite(f(a)):
            y[i] = 0  # lim t→0+ t² sin(1/t) = 0
        else:
            y[i] = f(x[i])
    
    weights = np.array([1 if (i==0 or i==n) else 4 if i%2==1 else 2 for i in range(n+1)])
    integral = h/3 * np.sum(weights * y)
    return integral

# Part a: ∫₀¹ x^(-1/4) sin(x) dx
def integrand_a(x):
    return x**(-1/4) * np.sin(x)

def transformed_a(t):
    return 4 * t**2 * np.sin(t**4)  # Well-behaved at t=0

# Part b: ∫₁^∞ x^(-4) sin(x) dx
def integrand_b(x):
    return x**(-4) * np.sin(x)

def transformed_b(t):
    if t == 0:
        return 0  # lim t→0+ t² sin(1/t) = 0
    return t**2 * np.sin(1/t)

# Compute approximations
n = 4

# Part a approximation
approx_a = composite_simpsons(transformed_a, 0, 1, n)
exact_a, _ = quad(integrand_a, 0, 1)

# Part b approximation
approx_b = composite_simpsons(transformed_b, 0, 1, n)
exact_b, _ = quad(integrand_b, 1, np.inf)

# Results
print(f"Part a approximation (n={n}): {approx_a:.6f}")
print(f"Part a exact value: {exact_a:.6f}")
print(f"Part a error: {abs(approx_a - exact_a):.2e}\n")

print(f"Part b approximation (n={n}): {approx_b:.6f}")
print(f"Part b exact value: {exact_b:.6f}")
print(f"Part b error: {abs(approx_b - exact_b):.2e}")