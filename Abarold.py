
# --- IMPORTS ---
# import numpy as nps
import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd import grad
from sklearn.linear_model import LinearRegression
# from autograd.numpy import sqrt
from math import sqrt
from numpy.linalg import norm, svd
from numpy import log
from joblib import Parallel, delayed
import csv

# --- SEEDING ---
np.random.seed(0)


def write_abar_old(Abar_old: np.ndarray) -> None:
    """Write the serialized Abar_old."""
    with open("./AbarOld", "wb") as f:
        f.write(Abar_old.tobytes())

class MTLModel:

    def __init__(self):
        self.Abar_old = None

    def MTL(self, x, y, r, eta=0.05, delta=0.05, max_iter=2000):
        T = x.shape[0]  # T tasks
        n = x.shape[1]  # n samples
        p = x.shape[2]  # p features
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

    def MTL_ours(self, x, y, r=3, T1=1, T2=0.05, R=5, r_bar=5, eta=0.05, max_iter=2000, C1=1, C2=0.5, delta=0.05,
                 adaptive=False):
        T = x.shape[0]
        n = x.shape[1]
        p = x.shape[2]

        ## adaptive or not
        if (adaptive == True):
            threshold = T1 * sqrt((p + log(T)) / n) + T2 * R * (r_bar ** (-3 / 4))
            # single-task linear regression
            beta_hat_single_task = np.zeros((p, T))
            for t in range(T):
                beta_hat_single_task[:, t] = LinearRegression().fit(x[t, :, :], y[t, :]).coef_
                length = norm(beta_hat_single_task[:, t])
                if (length > R):
                    beta_hat_single_task[:, t] = beta_hat_single_task[:, t] / R

            r = max(np.where(svd(beta_hat_single_task / sqrt(T))[1] > threshold)[0]) + 1

        A_hat = np.zeros((T, p, r))
        A_bar = np.zeros((p, r), dtype='float64')
        A_bar[0:r, 0:r] = np.identity(r, dtype='float64')

        for t in range(T):
            A_hat[t, 0:r, 0:r] = np.identity(r)

        theta_hat = np.zeros((r, T))

        ## initialization
        for t in range(T):
            def ftotal(A, theta):
                return (1 / n * np.dot(y[t, :] - x[t, :, :] @ A @ theta, y[t, :] - x[t, :, :] @ A @ theta))

            ftotal_grad = grad(ftotal, argnum=[0, 1])

            for i in range(200):
                S = ftotal_grad(A_hat[t, :, :], theta_hat[:, t])
                A_hat[t, :, :] = A_hat[t, :, :] - eta * S[0]
                theta_hat[:, t] = theta_hat[:, t] - eta * S[1]

        ## Step 1
        # lam = sqrt(r*(p+log(T)))*1
        lam = sqrt(r * (p + log(T))) * C1
        # lam1 = 2

        # loss = np.zeros(1000)
        for j in range(max_iter):
            def ftotal(A, theta, A_bar):
                s = 0
                for t in range(T):
                    s = s + 1 / n * 1 / T * np.dot(y[t, :] - x[t, :, :] @ A[t, :, :] @ theta[:, t],
                                                   y[t, :] - x[t, :, :] @ A[t, :, :] @ theta[:, t]) + lam / sqrt(
                        n) * 1 / T * max(abs(np.linalg.eigh(A[t, :, :] @ A[t, :, :].T - A_bar @ A_bar.T)[0]))
                s = s + delta * max(abs(np.linalg.eigh(A_bar.T @ A_bar - theta @ theta.T)[0]))
                return (s)

            ftotal_grad = grad(ftotal, argnum=[0, 1, 2])

            S = ftotal_grad(A_hat, theta_hat, A_bar)
            A_hat = A_hat - eta * S[0]
            theta_hat = theta_hat - eta * S[1]
            A_bar = A_bar - eta * S[2]

            # beta_hat_step1 = np.zeros((p, T))
            print(f'MTL ours {j}/{max_iter} iteration finished.... ')

        beta_hat_step1 = np.zeros((p, T))
        for t in range(T):
            beta_hat_step1[:, t] = A_hat[t, :, :] @ theta_hat[:, t]

        # Step 2
        gamma = sqrt(p + log(T)) * C2

        # Replacing global reference
        self.Abar_old = A_bar

        # Write abar old
        write_abar_old(self.Abar_old)


        beta_hat_step2 = np.zeros((p, T))
        for t in range(T):
            def f(beta):
                return (1 / n * np.dot(y[t, :] - x[t, :, :] @ beta, y[t, :] - x[t, :, :] @ beta) + gamma / sqrt(n) * (
                    sum((beta - beta_hat_step1[:, t]) ** 2)) ** 0.5)

            f_grad = grad(f)
            for j in range(max_iter):
                S = f_grad(beta_hat_step2[:, t])
                beta_hat_step2[:, t] = beta_hat_step2[:, t] - eta * S


        return (beta_hat_step2)

    def avg_distance(self, beta_hat, beta):
        s = 0
        T = beta.shape[1]
        for t in range(T):
            s = s + norm(beta_hat[:, t] - beta[:, t]) / T
        return (s)

    def max_distance(self, beta_hat, beta):
        T = beta.shape[1]
        s = np.zeros(T)
        for t in range(T):
            s[t] = norm(beta_hat[:, t] - beta[:, t])
        return (max(s))

    def all_distance(self, beta_hat, beta):
        T = beta.shape[1]
        s = np.zeros(T)
        for t in range(T):
            s[t] = norm(beta_hat[:, t] - beta[:, t])
        return (s)

    def our_task_outlier(self, h):
        n = 100;
        p = 20;
        r = 3;
        T = 6

        ## parameter setting: 1 outlier
        theta = np.array([[1, 0.5, 0], [1, -1, 1], [1.5, 1.5, 0], [1, 1, 0], [1, 0, 1], [-1, -1, -1]]).T * 2
        R = np.random.normal(0, 1, p * p).reshape((p, p))
        A_center = (np.linalg.svd(R)[0])[0:r, ]
        A_center = A_center.T  # p*r matrix
        A = np.zeros((T, p, r))
        beta = np.zeros((p, T))
        for t in range(T):
            Delta_A = np.zeros((p, r))
            Delta_A[0:r, 0:r] = np.random.uniform(low=-h, high=h, size=1) * np.identity(r)
            A[t, :, :] = A_center + Delta_A
            beta[:, t] = A[t, :, :] @ theta[:, t]

        # beta_outlier = np.random.uniform(-1, 1, p)
        # beta = np.hstack((beta, beta_outlier.reshape(p, 1)))
        #
        # T = 7

        ## data generation
        x = np.zeros((T, n, p))
        y = np.zeros((T, n))

        for t in range(T):
            x[t, :, :] = np.random.normal(0, 1, n * p).reshape((n, p))
            y[t, :] = x[t, :, :] @ beta[:, t] + np.random.normal(0, 1, n)

        # single-task linear regression
        beta_hat_single_task = np.zeros((p, T))
        for t in range(T):
            beta_hat_single_task[:, t] = LinearRegression().fit(x[t, :, :], y[t, :]).coef_

        # MTL ours
        beta_hat_ours = self.MTL_ours(x=x, y=y, r=3, max_iter=2000, eta=0.05)

        # MTL ours: adaptive
        beta_hat_ours_ad = self.MTL_ours(x=x, y=y, r=3, r_bar=sqrt(T), max_iter=2000, eta=0.05, adaptive=True)

        # MTL the same representation
        beta_hat = self.MTL(x=x, y=y, r=3, eta=0.05, max_iter=2000)

        result = np.zeros(4)
        result[0] = max(self.all_distance(beta_hat_single_task, beta)[0:(T - 1)])
        result[1] = max(self.all_distance(beta_hat, beta)[0:(T - 1)])
        result[2] = max(self.all_distance(beta_hat_ours, beta)[0:(T - 1)])
        result[3] = max(self.all_distance(beta_hat_ours_ad, beta)[0:(T - 1)])

        return (result)


# --- Part 3 MAIN EXECUTION ---
model_instance = MTLModel()

h_list = np.arange(0.1, 0.9, 0.1)

mse_outlier = np.zeros((h_list.size, 4))
mse_outlier = np.array(Parallel(n_jobs=4)(delayed(model_instance.our_task_outlier)(h) for h in h_list))
mse_outlier = mse_outlier.reshape((1, h_list.size * 4))



