# --- IMPORTS ---
# import numpy as nps
import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd import grad

# --- SEEDING ---
np.random.seed(0)
max_iter_num = 200

def MTL(x, y, r, eta=0.05, delta=0.05, max_iter=max_iter_num):
    T = x.shape[0]
    n = x.shape[1]
    p = x.shape[2]
    A_hat = np.zeros((p, r))

    for t in range(T):
        A_hat[0:r, 0:r] = np.identity(r)

    theta_hat = np.zeros((r, T))

    ## initialization
    t = 0

    def ftotal(A, theta):
        return (1 / n * np.dot(y[t, :] - x[t, :, :] @ A @ theta, y[t, :] - x[t, :, :] @ A @ theta))

    ftotal_grad = grad(ftotal, argnum=[0, 1])

    for i in range(200):
        S = ftotal_grad(A_hat, theta_hat[:, t])
        A_hat = A_hat - eta * S[0]
        theta_hat[:, t] = theta_hat[:, t] - eta * S[1]

    ## Step 1

    # lam1 = 2
    for j in range(max_iter):
        def ftotal(A, theta):
            s = 0
            for t in range(T):
                s = s + 1 / n * 1 / T * np.dot(y[t, :] - x[t, :, :] @ A @ theta[:, t],
                                               y[t, :] - x[t, :, :] @ A @ theta[:, t])
            s = s + delta * max(abs(np.linalg.eigh(A.T @ A - theta @ theta.T)[0]))
            return (s)

        ftotal_grad = grad(ftotal, argnum=[0, 1])

        S = ftotal_grad(A_hat, theta_hat)
        A_hat = A_hat - eta * S[0]
        theta_hat = theta_hat - eta * S[1]
        print(f'MTL {j}/{max_iter} iteration finished.... ')

    beta_hat_step1 = np.zeros((p, T))
    for t in range(T):
        beta_hat_step1[:, t] = A_hat @ theta_hat[:, t]

    return (beta_hat_step1)

    pass