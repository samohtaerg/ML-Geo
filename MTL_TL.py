# --- IMPORTS ---
# import numpy as nps
import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd import grad
from sklearn.linear_model import LinearRegression
from math import sqrt
from numpy.linalg import norm, svd
from numpy import log
from joblib import Parallel, delayed
import csv


# --- SEEDING ---
np.random.seed(0)

# Define the read_in function
def read_abar_old() -> np.ndarray:
    with open("./AbarOld", "rb") as f:
        serialized_content = f.read()
        return np.frombuffer(serialized_content)
        # flat_array = np.frombuffer(serialized_content)
        # return flat_array.reshape(20, 3)

class MTL_TL:

    def __init__(self):
        self.A_old = None

    def MTL(self, x, y, r, eta=0.05, delta=0.05, max_iter=2):
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

    def MTL_transfer(self, x, y, r=3, T1=1, T2=0.05, R=5, r_bar=5, eta=0.05, max_iter=2, C1=1, C2=0.5, delta=0.05
    , transfer=False):
        T = x.shape[0]
        n = x.shape[1]
        p = x.shape[2]


        # A_hat = np.zeros((T, p, r))
        # A_bar_tf = np.zeros((p, r), dtype='float64')
        # A_bar_tf[0:r, 0:r] = np.identity(r, dtype='float64')

        # for t in range(T):
        #     A_hat[t, 0:r, 0:r] = np.identity(r)

        theta_hat = np.zeros((r, T))

        # Read in A_bar old file
        abar_old_array = read_abar_old()
        self.A_old = abar_old_array

        ## initialization
        for t in range(T):
            # def ftotal(A, theta):
            #     return (1 / n * np.dot(y[t, :] - x[t, :, :] @ A @ theta, y[t, :] - x[t, :, :] @ A @ theta))

            def ftotal(theta):
                return (1 / n * np.dot(y[0, :] - x[0, :, :] @ self.A_old @ theta, y[0, :] - x[0, :, :] @
                                        self.A_old @ theta))

            # ftotal_grad = grad(ftotal, argnum=[0, 1])
            ftotal_grad = grad(ftotal, argnum=0)

            for i in range(200):
                # S = ftotal_grad(A_hat[t, :, :], theta_hat[:, t])
                # A_hat[t, :, :] = A_hat[t, :, :] - eta * S[0]
                # theta_hat[:, t] = theta_hat[:, t] - eta * S[1]
                S = ftotal_grad(theta_hat)
                theta_hat = theta_hat - eta * S

        ## Step 1
        lam = sqrt(r * (p + log(T))) * C1
        for j in range(max_iter):
            def ftotal(A, theta, A_bar):
                s = 0
                for t in range(T):
                    s = s + 1 / n * 1 / T * np.dot(y[t, :] - x[t, :, :] @ A[t, :, :] @ theta[:, t],
                                                   y[t, :] - x[t, :, :] @ A[t, :, :] @ theta[:, t]) + lam / sqrt(
                        n) * 1 / T * max(abs(np.linalg.eigh(A[t, :, :] @ A[t, :, :].T - A_bar @ A_bar.T)[0]))
                s = s + delta * max(abs(np.linalg.eigh(A_bar.T @ A_bar - theta @ theta.T)[0]))
                return (s)

            # S = ftotal_grad(A_hat, theta_hat, A_bar_tf)
            # A_hat = A_hat - eta * S[0]
            # theta_hat = theta_hat - eta * S[1]
            # A_bar_tf = A_bar_tf - eta * S[2]
            # ftotal_grad = grad(ftotal, argnum=[0, 1, 2])
            ftotal_grad = grad(ftotal)
            S = ftotal_grad(theta_hat)
            theta_hat = theta_hat - eta * S
            print(f'MTL transfer {j}/{max_iter} iteration finished.... ')

        # Step 2: Use Ahat_old instead of A_hat for regularization

        # Initialize
        beta_hat_step1_tf = np.zeros((p, T))

        beta_hat_step1_tf[:, t] = self.A_old @ theta_hat
        gamma = sqrt(p + log(T)) * C2
        beta_hat_step2_tf = np.zeros((p, T))

        for t in range(T):
            def f(beta):
                return (1 / n * np.dot(y[t, :] - x[t, :, :] @ beta, y[t, :] - x[t, :, :] @ beta) + gamma / sqrt(n) * (
                    sum((beta - beta_hat_step1_tf[:, t]) ** 2)) ** 0.5)

            f_grad = grad(f)
            for j in range(max_iter):
                S = f_grad(beta_hat_step2_tf[:, t])
                beta_hat_step2_tf[:, t] = beta_hat_step2_tf[:, t] - eta * S

        return beta_hat_step2_tf

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

    def our_task_tl(self, h, noise_factor=1):
        n = 100
        p = 20
        r = 3
        T = 1

        # parameter setting
        theta = np.array([[1, 0.5, 0], [1, -1, 1], [1.5, 1.5, 0], [1, 1, 0], [1, 0, 1], [-1, -1, -1]]).T * 2
        R = np.random.normal(0, 1, p * p).reshape((p, p))
        A_center = (np.linalg.svd(R)[0])[0:r, ]
        A_center = A_center.T  # p*r matrix
        A = np.zeros((T, p, r))
        beta = np.zeros((p, T))
        beta_hat_transfer = np.zeros((p, T))
        for t in range(T):
            Delta_A = np.zeros((p, r))
            Delta_A[0:r, 0:r] = np.random.uniform(low=-h, high=h, size=1) * np.identity(r)
            A[t, :, :] = A_center + Delta_A
            beta[:, t] = A[t, :, :] @ theta[:, t]

        # data generation with increased noise
        x = np.random.normal(0, 1, n * p).reshape((n, p))
        y = x @ beta + np.random.normal(0, noise_factor, n)

        # single-task linear regression
        beta_hat_single_task = LinearRegression().fit(x, y).coef_

        # MTL ours
        beta_hat_ours = self.MTL_transfer(x, y, r=3, T1=1, T2=0.05, R=5, r_bar=5, eta=0.05, max_iter=2, C1=1,
                                                 C2=0.5, delta=0.05, transfer=False)

        # MTL the same representation
        beta_hat = self.MTL(x =x, y=y, r=3, eta=0.05, max_iter=2)

        # MTL_transfer (need to create this function)
        beta_hat_transfer = self.MTL_transfer(x, y, r=3, T1=1, T2=0.05, R=5, r_bar=5, eta=0.05, max_iter=2, C1=1,
                                         C2=0.5, delta=0.05, transfer=True)

        result = np.zeros(4)
        result[0] = max(self.all_distance(beta_hat_single_task, beta)[0:(T - 1)])
        result[1] = max(self.all_distance(beta_hat, beta)[0:(T - 1)])
        result[2] = max(self.all_distance(beta_hat_ours, beta)[0:(T - 1)])
        result[3] = max(self.all_distance(beta_hat_transfer, beta)[0:(T - 1)])

        return result

# --- Part 3 MAIN EXECUTION ---
model_instance = MTL_TL()

h_list = np.arange(0.1, 0.9, 0.1)

mse_tl = np.zeros((h_list.size, 4))
mse_tl = np.array(Parallel(n_jobs=4)(delayed(model_instance.our_task_tl)(h) for h in h_list))
mse_tl = mse_tl.reshape((1, h_list.size * 4))

with open("C:/Users/samoh/PycharmProjects/MTLa/venv/MTL_TL.py" + str(0) + "_1.csv", "w",
            newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(mse_tl)