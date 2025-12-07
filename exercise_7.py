from math import log
from scipy.stats import chi2

# Overall class frequencies
C = [59, 71, 48]
n = sum(C)

# Rules (Observed counts)
rules = {
    "R1": [2, 0, 18],
    "R2": [16, 6, 1],
    "R3": [12, 7, 6]
}

def expected_counts(nR):
    return [nR * ci / n for ci in C]

def g_statistic(m, e):
    g = 0
    for m, e in zip(m, e):
        if m > 0:
            g += 2 * m * log(m / e)
    return g

def chi_square(m, e):
    return sum((m - e) ** 2 / e for m, e in zip(m, e))

df = 2  # degrees of freedom

for name, m in rules.items():
    nR = sum(m)
    e = expected_counts(nR)
    
    G = g_statistic(m, e)
    X = chi_square(m, e)
    
    # p-values
    p_G    = chi2.sf(G, df)
    p_chi2 = chi2.sf(X, df)
    
    print(f"\n{name}:")
    print(f"Observed counts: {m}")
    print(f"Expected counts: {[round(e,4) for e in e]}")
    print(f"G statistic     = {G:.6f}")
    print(f"Chi-square      = {X:.6f}")
    print(f"p-value (G)     = {p_G:.6g}")
    print(f"p-value (χ²)    = {p_chi2:.6g}")
