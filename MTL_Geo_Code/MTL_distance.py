
# --- IMPORTS ---

import autograd.numpy as np  # Thinly-wrapped version of Numpy
from numpy.linalg import norm, svd

# --- DISTANCE FUNCTIONS ---
def avg_distance(beta_hat, beta):
    s = 0
    T = beta.shape[1]
    for t in range(T):
        s = s + norm(beta_hat[:, t] - beta[:, t]) / T
    return (s)

def max_distance(beta_hat, beta):
    T = beta.shape[1]
    s = np.zeros(T)
    for t in range(T):
        s[t] = norm(beta_hat[:, t] - beta[:, t])
    return (max(s))

def all_distance(beta_hat, beta):
    T = beta.shape[1]
    s = np.zeros(T)
    for t in range(T):
        s[t] = norm(beta_hat[:, t] - beta[:, t])
    return (s)

pass