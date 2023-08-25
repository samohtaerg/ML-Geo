def MTL_ours(self, x, y, r=3, T1=1, T2=0.05, R=5, r_bar=5, eta=0.05, max_iter=2000, C1=1, C2=0.5, delta=0.05):
    T = x.shape[0]
    n = x.shape[1]
    p = x.shape[2]

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