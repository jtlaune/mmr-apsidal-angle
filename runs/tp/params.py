import numpy as np

j = 2
e0 = 0.0
ap = 1.0
theta0 = 0
g0 = 0.0
eta0 = 1.0
a0 = 1.34
lambda0 = 0.0

N = 11
eccs = np.zeros(N)
eccs[1:] = np.logspace(-2, -1, 10)
eps = eccs
eeqs = eccs[1:]
ratios = eeqs ** 2 * 2 * j
