import os
import time

import numpy as np
import tensorflow as tf
import torch


def gd_linreg(X, y, lr=0.2, epoki=50):
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    historia = []

    for ep in range(epoki):
        y_pred = X @ w + b
        err = y_pred - y

        historia.append(np.mean(err**2))

        dw = (2.0 / n) * (X.T @ err)
        db = (2.0 / n) * np.sum(err)

        w -= lr * dw
        b -= lr * db

    return w, b, historia

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def poly_features(x, stopien):
    return np.vander(x, stopien + 1, increasing=True)

def dopasuj_wielomian(x, y, stopien):
    Phi = poly_features(x, stopien)
    return np.linalg.lstsq(Phi, y, rcond=None)[0]


def zad_1_1():
    # Standaryzacja w TensorFlow (NHWC)
    rng = np.random.default_rng(0)
    batch_np = rng.normal(loc=[0.5, 0.2, -0.3], scale=[2.0, 0.5, 1.0],
                          size=(16, 32, 32, 3)).astype(np.float32)

    x_tf = tf.convert_to_tensor(batch_np)
    mu_tf = tf.reduce_mean(x_tf, axis=[0, 1, 2], keepdims=True)
    std_tf = tf.math.reduce_std(x_tf, axis=[0, 1, 2], keepdims=True)
    z_tf = (x_tf - mu_tf) / std_tf

    # Standaryzacja w PyTorch (NCHW)
    x_pt = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
    mu_pt = x_pt.mean(dim=(0, 2, 3), keepdim=True)
    std_pt = x_pt.std(dim=(0, 2, 3), correction=0, keepdim=True)
    z_pt = (x_pt - mu_pt) / std_pt

    if z_tf is None or z_pt is None:
        print("Uzupełnij z_tf i z_pt.")
    else:
        print("TF  średnie per kanał:", np.round(tf.reduce_mean(z_tf, axis=[0, 1, 2]).numpy(), 4))
        print("TF  std per kanał    :", np.round(tf.math.reduce_std(z_tf, axis=[0, 1, 2]).numpy(), 4))
        print("PT  średnie per kanał:", np.round(z_pt.mean(dim=(0, 2, 3)).numpy(), 4))
        z_pt_nhwc = z_pt.permute(0, 2, 3, 1).numpy()  # powrót do NHWC
        print("Zgodność TF vs PyTorch:", np.allclose(z_tf.numpy(), z_pt_nhwc, atol=1e-5))


def lab_1_2():
    SZYBKI_TRYB = os.environ.get("DL_FAST", "0") == "1"

    def pairwise_sq_dist_np(X, Y):
        x_norm = np.sum(X ** 2, axis=1, keepdims=True) # (n, 1)
        y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T # (1, m)
        D = x_norm + y_norm - 2 * (X @ Y.T) # (n, m)
        return np.maximum(D, 0)

    def pairwise_sq_dist_torch(X, Y):
        x_norm = (X ** 2).sum(dim=1, keepdim=True) # (n, 1)
        y_norm = (Y ** 2).sum(dim=1, keepdim=True).T # (1, m)
        D = x_norm + y_norm - 2 * (X @ Y.T) # (n, m)
        return torch.clamp(D, min=0)

    def pairwise_sq_dist_loop(X, Y):
        D = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                D[i, j] = np.sum((X[i] - Y[j]) ** 2)
        return D

    Xs, Ys = np.random.rand(50, 8), np.random.rand(40, 8)
    D_ref = pairwise_sq_dist_loop(Xs, Ys)
    D_np = pairwise_sq_dist_np(Xs, Ys)
    D_pt = pairwise_sq_dist_torch(torch.from_numpy(Xs), torch.from_numpy(Ys))
    if D_np is not None and D_pt is not None:
        print("NumPy - zgodność z wersją pętlową:", np.allclose(D_ref, D_np))
        print("Torch - zgodność z wersją pętlową:", np.allclose(D_ref, D_pt.numpy()))

        # Pomiar czasu: pętla vs wektoryzacja
        n_big = 150 if SZYBKI_TRYB else 400
        Xb, Yb = np.random.rand(n_big, 32), np.random.rand(n_big, 32)
        t0 = time.perf_counter()
        pairwise_sq_dist_loop(Xb, Yb)
        t_loop = time.perf_counter() - t0
        t0 = time.perf_counter()
        pairwise_sq_dist_np(Xb, Yb)
        t_vec = time.perf_counter() - t0
        print(f"n = {n_big}: pętla {t_loop:.3f} s, wektorowo {t_vec:.5f} s, przyspieszenie ~{t_loop / max(t_vec, 1e-9):.0f}x")


def zad_2_1():
    def df_dx_analit(x, y):
        return np.cos(x) * y ** 2 + y * np.exp(x * y)

    def df_dy_analit(x, y):
        return 2 * y * np.sin(x) + x * np.exp(x * y)

    x0, y0 = 0.5, -1.0
    grad_analit = np.array([df_dx_analit(x0, y0), df_dy_analit(x0, y0)])

    # tf
    x_tf = tf.Variable(x0, dtype=tf.float64)
    y_tf = tf.Variable(y0, dtype=tf.float64)

    with tf.GradientTape() as tape:
        f_tf = tf.sin(x_tf) * y_tf ** 2 + tf.exp(x_tf * y_tf)

    grad_tf = tape.gradient(f_tf, [x_tf, y_tf])
    grad_tf = np.array([g.numpy() for g in grad_tf])

    # pyTorch
    x_pt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    y_pt = torch.tensor(y0, dtype=torch.float64, requires_grad=True)

    f_pt = torch.sin(x_pt) * y_pt ** 2 + torch.exp(x_pt * y_pt)
    f_pt.backward()

    grad_pt = np.array([x_pt.grad.item(), y_pt.grad.item()])

    # Porównanie wyników
    print("Gradient analityczny:", grad_analit)
    print("Gradient TensorFlow: ", grad_tf)
    print("Gradient PyTorch:    ", grad_pt)

    print("TF zgodny:", np.allclose(grad_analit, grad_tf, atol=1e-6))
    print("PT zgodny:", np.allclose(grad_analit, grad_pt, atol=1e-6))
    print("TF vs PT:  ", np.allclose(grad_tf, grad_pt, atol=1e-6))


def zad_2_2():
    import matplotlib.pyplot as plt

    def grad_numeryczny(f, w, h=1e-5):
        w = np.asarray(w, dtype=np.float64)
        grad = np.zeros_like(w)

        for i in range(len(w)):
            w_plus = w.copy()
            w_minus = w.copy()

            w_plus[i] += h
            w_minus[i] -= h

            grad[i] = (f(w_plus) - f(w_minus)) / (2 * h)

        return grad

    rng = np.random.default_rng(1)
    A_mat = rng.normal(size=(6, 4))
    w0 = rng.normal(size=4)

    def f_np(w):  # wersja NumPy (do różnic skończonych)
        z = A_mat @ w
        return np.log(np.sum(np.exp(z))) + 0.5 * np.sum(w ** 2)

    # Gradient z autograd (PyTorch, float64)
    w_t = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    f_t = torch.logsumexp(torch.tensor(A_mat) @ w_t, dim=0) + 0.5 * (w_t ** 2).sum()
    f_t.backward()
    g_auto = w_t.grad.numpy()

    def blad_wzgledny(a, b):
        return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-300)

    g_num = grad_numeryczny(f_np, w0, h=1e-5)
    print("autograd :", np.round(g_auto, 6))
    print("numeryczny:", np.round(g_num, 6))
    print("błąd względny (h=1e-5):", blad_wzgledny(g_num, g_auto))

    # Błąd w funkcji kroku h
    hs = np.logspace(-14, 0, 29)
    bledy = [blad_wzgledny(grad_numeryczny(f_np, w0, h), g_auto) for h in hs]
    plt.figure(figsize=(7, 4))
    plt.loglog(hs, np.maximum(bledy, 1e-17), "o-")
    plt.title("Gradient check: błąd względny vs krok h")
    plt.xlabel("krok h")
    plt.ylabel("błąd względny")
    plt.grid(alpha=0.3, which="both")
    plt.show()


def zad_3_1():
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    SZYBKI_TRYB = True
    SEED = 42
    EPOKI_GD = 50 if SZYBKI_TRYB else 500

    def gd_linreg(X, y, lr=0.2, epoki=EPOKI_GD):
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        historia = []

        for ep in range(epoki):
            # Predykcja
            y_pred = X @ w + b

            # Błąd predykcji
            err = y_pred - y

            # MSE
            mse_val = np.mean(err ** 2)
            historia.append(mse_val)

            # Gradienty MSE
            dw = (2 / n) * (X.T @ err)
            db = (2 / n) * np.sum(err)

            # Aktualizacja parametrów
            w -= lr * dw
            b -= lr * db

        return w, b, historia

    dom = fetch_california_housing()
    X_all, y_all = dom.data.astype(np.float64), dom.target.astype(np.float64)

    if SZYBKI_TRYB:  # tryb szybki: tylko 2000 pierwszych próbek
        X_all, y_all = X_all[:2000], y_all[:2000]

    print("cechy:", list(dom.feature_names))
    print("X:", X_all.shape, " y:", y_all.shape)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED)
    mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0)
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    n_tr, _ = X_tr_s.shape

    # Baseline: stała = średnia z treningu
    print("Baseline (średnia) MSE test:", round(mse(y_te, np.full_like(y_te, y_tr.mean())), 4))

    # Rozwiązanie analityczne (najmniejsze kwadraty) - dokładne minimum na train
    Xb_tr = np.hstack([X_tr_s, np.ones((n_tr, 1))])  # kolumna jedynek = bias
    theta_ls = np.linalg.lstsq(Xb_tr, y_tr, rcond=None)[0]
    w_ls, b_ls = theta_ls[:-1], theta_ls[-1]
    loss_opt = mse(y_tr, X_tr_s @ w_ls + b_ls)

    t0 = time.perf_counter()
    w_gd, b_gd, hist_gd = gd_linreg(X_tr_s, y_tr, lr=0.2, epoki=EPOKI_GD)
    print(f"czas: {time.perf_counter() - t0:.2f} s")
    if len(hist_gd) > 0:
        print(f"MSE train GD = {hist_gd[-1]:.4f}  vs  optimum = {loss_opt:.4f}  (różnica {100 * (hist_gd[-1] / loss_opt - 1):.2f}%)")
        print("MSE test GD  =", round(mse(y_te, X_te_s @ w_gd + b_gd), 4))
        print("max |w_gd - w_ls| =", np.abs(w_gd - w_ls).max().round(4))

        plt.figure(figsize=(7, 4))
        plt.plot(hist_gd, label="GD (NumPy)")
        plt.axhline(loss_opt, color="k", ls="--", label="optimum (lstsq)")
        plt.yscale("log")
        plt.title("California Housing: krzywa uczenia GD")
        plt.xlabel("epoka")
        plt.ylabel("MSE train (log)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        # Które cechy mają największy wpływ? (cechy są standaryzowane, więc wagi są porównywalne)
        plt.figure(figsize=(7, 3.5))
        plt.barh(dom.feature_names, w_gd)
        plt.title("Wagi regresji liniowej (cechy standaryzowane)")
        plt.xlabel("waga")
        plt.ylabel("cecha")
        plt.grid(alpha=0.3)
        plt.show()


def zad_3_2():
    import numpy as np
    import tensorflow as tf
    import torch
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    EPOKI_GD = 50
    lr = 0.2

    # Dane
    dom = fetch_california_housing()
    X_all = dom.data.astype(np.float64)
    y_all = dom.target.astype(np.float64)

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma

    # NumPy
    w_gd, b_gd, hist_gd = gd_linreg(X_tr_s, y_tr, lr=lr, epoki=EPOKI_GD)

    # PyTorch
    Xtr_pt = torch.tensor(X_tr_s, dtype=torch.float64)
    ytr_pt = torch.tensor(y_tr, dtype=torch.float64)

    w_pt = torch.zeros(X_tr_s.shape[1], dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    hist_pt = []

    for ep in range(EPOKI_GD):
        y_pred_pt = Xtr_pt @ w_pt + b_pt
        loss_pt = torch.mean((y_pred_pt - ytr_pt) ** 2)
        hist_pt.append(loss_pt.item())

        loss_pt.backward()

        with torch.no_grad():
            w_pt -= lr * w_pt.grad
            b_pt -= lr * b_pt.grad
            w_pt.grad.zero_()
            b_pt.grad.zero_()

    # TensorFlow
    Xtr_tf = tf.constant(X_tr_s, dtype=tf.float32)
    ytr_tf = tf.constant(y_tr, dtype=tf.float32)

    d = X_tr_s.shape[1]
    w_tf = tf.Variable(tf.zeros([d], dtype=tf.float32))
    b_tf = tf.Variable(0.0, dtype=tf.float32)
    hist_tf = []

    for ep in range(EPOKI_GD):
        with tf.GradientTape() as tape:
            y_pred_tf = tf.linalg.matvec(Xtr_tf, w_tf) + b_tf
            loss_tf = tf.reduce_mean((y_pred_tf - ytr_tf) ** 2)

        dw_tf, db_tf = tape.gradient(loss_tf, [w_tf, b_tf])

        hist_tf.append(float(loss_tf.numpy()))

        w_tf.assign_sub(lr * dw_tf)
        b_tf.assign_sub(lr * db_tf)

    print(f"MSE train: NumPy {hist_gd[-1]:.5f} \n PyTorch {hist_pt[-1]:.5f} |\n TF {hist_tf[-1]:.5f}")
    print("MSE test PyTorch:", round(torch.mean((torch.tensor(X_te_s) @ w_pt.detach() + b_pt.detach() - torch.tensor(y_te))**2).item(), 5))
    print("MSE test TF:", round(tf.reduce_mean((tf.linalg.matvec(tf.constant(X_te_s, dtype=tf.float32), w_tf) + b_tf - tf.constant(y_te, dtype=tf.float32))**2).numpy().item(), 5))

    plt.figure(figsize=(7, 4))
    plt.plot(hist_gd, lw=3, alpha=0.6, label="NumPy")
    plt.plot(hist_pt, "--", label="PyTorch autograd")
    plt.plot(hist_tf, ":", lw=2, label="TF GradientTape")
    plt.yscale("log")
    plt.title("Ten sam algorytm w trzech bibliotekach")
    plt.xlabel("Epoka")
    plt.ylabel("MSE train (log)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def zad_4_1():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    import numpy as np
    import matplotlib.pyplot as plt

    SZYBKI_TRYB = True
    dom = fetch_california_housing()
    X_all = dom.data.astype(np.float64)
    y_all = dom.target.astype(np.float64)

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    X_tr_s = (X_tr - mu) / sigma

    n = X_tr_s.shape[0]
    Xb = np.column_stack([X_tr_s, np.ones(n)])

    H = (2.0 / n) * (Xb.T @ Xb)

    eigvals = np.linalg.eigvalsh(H)
    lam_min = eigvals.min()
    lam_max = eigvals.max()

    uwarunkowanie = lam_max / lam_min
    lr_kryt = 2.0 / lam_max

    print("=== Analiza hesjanu ===")
    print(f"lambda_min = {lam_min:.6f}")
    print(f"lambda_max = {lam_max:.6f}")
    print(f"Uwarunkowanie = {uwarunkowanie:.2f}")
    print(f"Teoretyczny próg rozbieżności lr_kryt = 2/lambda_max = {lr_kryt:.6f}")

    EPOKI_LR = 40 if SZYBKI_TRYB else 200

    lrs = [0.001, 0.01, 0.1, 0.5 * lr_kryt, 0.98 * lr_kryt, 1.05 * lr_kryt]
    historie = {}

    for lr_ in lrs:
        _, _, hist = gd_linreg(X_tr_s, y_tr, lr=lr_, epoki=EPOKI_LR)

        historie[lr_] = hist

        if np.all(np.isfinite(hist)):
            print(f"lr = {lr_:.6f}\nloss początkowy = {hist[0]:.6f}\nloss końcowy = {hist[-1]:.6f}")
        else:
            print(f"lr = {lr_:.6f} | loss rozbiega się")

    plt.figure(figsize=(11, 6))
    for lr_, hist in historie.items():
        hist = np.asarray(hist)
        hist[hist <= 0] = np.nan

        plt.plot(range(1, len(hist) + 1), hist, label=f"lr = {lr_:.5f}")

    plt.yscale("log")
    plt.xlabel("Epoka")
    plt.ylabel("MSE (skala logarytmiczna)")
    plt.title("Wpływ learning rate na zbieżność gradient descent")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def zad_5_1(SZYBKI_TRYB=True):
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    import numpy as np
    import time
    import math
    import matplotlib.pyplot as plt

    def minibatch_gd(X, y, lr=0.05, epoki=20, batch_size=32, seed=0):
        rng = np.random.default_rng(seed)
        n, d = X.shape

        w = np.zeros(d)
        b = 0.0

        historia, czasy = [], []
        t0 = time.perf_counter()

        for ep in range(epoki):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]

                X_batch = X[idx]
                y_batch = y[idx]
                m = len(idx)

                y_pred = X_batch @ w + b
                err = y_pred - y_batch

                dw = (2.0 / m) * (X_batch.T @ err)
                db = (2.0 / m) * np.sum(err)

                w -= lr * dw
                b -= lr * db

            y_pred_all = X @ w + b
            mse = np.mean((y_pred_all - y) ** 2)

            historia.append(mse)
            czasy.append(time.perf_counter() - t0)

        return w, b, historia, czasy

    dom = fetch_california_housing()
    X_all = dom.data.astype(np.float64)
    y_all = dom.target.astype(np.float64)

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    X_tr_s = (X_tr - mu) / sigma

    n_tr, _ = X_tr_s.shape
    Xb_tr = np.hstack([X_tr_s, np.ones((n_tr, 1))])  # kolumna jedynek = bias
    theta_ls = np.linalg.lstsq(Xb_tr, y_tr, rcond=None)[0]
    w_ls, b_ls = theta_ls[:-1], theta_ls[-1]

    loss_opt = mse(y_tr, X_tr_s @ w_ls + b_ls)

    # Porównanie różnych rozmiarów batcha
    EPOKI_MB = 4 if SZYBKI_TRYB else 30
    n_tr = X_tr_s.shape[0]

    wyniki_mb = {}

    for bs in [16, 64, 256, n_tr]:
        _, _, h, c = minibatch_gd(X_tr_s, y_tr, lr=0.05, epoki=EPOKI_MB, batch_size=bs, seed=0)

        nazwa = "full-batch" if bs == n_tr else f"B = {bs}"
        wyniki_mb[nazwa] = (h, c)

        print(f"{nazwa:>10}: kroków/epokę = {math.ceil(n_tr / bs):5d}, MSE końcowe = {h[-1]:.4f}, czas = {c[-1]:.2f} s")

    # Wykresy: MSE względem epoki oraz czasu
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    for nazwa, (h, c) in wyniki_mb.items():
        ax[0].plot(range(1, len(h) + 1), h, "o-", ms=3, label=nazwa)
        ax[1].plot(c, h, "o-", ms=3, label=nazwa)

    for a_, xl in zip(ax, ["epoka", "czas [s]"]):
        a_.axhline(loss_opt, color="k", ls="--", lw=1, label="optimum")
        a_.set_yscale("log")
        a_.set_xlabel(xl)
        a_.set_ylabel("MSE train (log)")
        a_.grid(alpha=0.3, which="both")
        a_.legend()

    ax[0].set_title("Mini-batch SGD: strata vs epoka")
    ax[1].set_title("Mini-batch SGD: strata vs czas")

    plt.tight_layout()
    plt.show()


def zad_6_1():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)

    X_all = np.sort(rng.uniform(0, 1, 120))
    y_all = np.sin(2 * np.pi * X_all) + rng.normal(0, 0.3, 120)

    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    stopnie = list(range(16))
    mse_train_D, mse_val_D = [], []

    for D in stopnie:
        wsp = dopasuj_wielomian(X_tr, y_tr, D)

        y_tr_pred = poly_features(X_tr, D) @ wsp
        y_val_pred = poly_features(X_val, D) @ wsp

        mse_train_D.append(np.mean((y_tr_pred - y_tr) ** 2))
        mse_val_D.append(np.mean((y_val_pred - y_val) ** 2))

    najlepszy_D = int(np.argmin(mse_val_D))

    print(f"Najlepszy stopień: D = {najlepszy_D}, MSE val = {mse_val_D[najlepszy_D]:.4f}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(stopnie, mse_train_D, "o-", label="MSE train")
    plt.plot(stopnie, mse_val_D, "s-", label="MSE walidacja")
    plt.axvline(najlepszy_D, color="gray", ls=":", label=f"najlepszy D = {najlepszy_D}")
    plt.axhline(0.3 ** 2, color="k", ls="--", lw=1, label="wariancja szumu (0.09)")

    plt.yscale("log")
    plt.title("Pojemność modelu: underfitting vs overfitting")
    plt.xlabel("Stopień wielomianu D")
    plt.ylabel("MSE (log)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def zad_6_2():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)

    X_all = np.sort(rng.uniform(0, 1, 120))
    y_all = np.sin(2 * np.pi * X_all) + rng.normal(0, 0.3, 120)

    X_trp, X_vap, y_trp, y_vap = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    def dopasuj_ridge(x, y, stopien, lam):
        Phi = poly_features(x, stopien)
        n = len(y)

        I = np.eye(stopien + 1)
        I[0, 0] = 0

        A = Phi.T @ Phi + n * lam * I
        rhs = Phi.T @ y

        return np.linalg.solve(A, rhs)

    D_duzy = 15
    lambdy = np.logspace(-12, 1, 27)

    mse_tr_l = []
    mse_va_l = []
    for lam in lambdy:
        wl = dopasuj_ridge(X_trp, y_trp, D_duzy, lam)

        y_tr_pred = poly_features(X_trp, D_duzy) @ wl
        y_va_pred = poly_features(X_vap, D_duzy) @ wl

        mse_tr_l.append(mse(y_trp, y_tr_pred))
        mse_va_l.append(mse(y_vap, y_va_pred))

    best_idx = int(np.argmin(mse_va_l))
    best_lam = lambdy[best_idx]

    print(f"Najlepsza lambda = {best_lam:.2e}")
    print(f"MSE train = {mse_tr_l[best_idx]:.4f}")
    print(f"MSE val   = {mse_va_l[best_idx]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    ax[0].semilogx(lambdy, mse_tr_l, "o-", label="MSE train")
    ax[0].semilogx(lambdy, mse_va_l, "s-", label="MSE walidacja")
    ax[0].axvline(
        best_lam, color="gray", ls=":",
        label=f"best lambda = {best_lam:.1e}"
    )
    ax[0].set_yscale("log")
    ax[0].set_title(f"Ridge, D = {D_duzy}: błąd vs regularyzacja")
    ax[0].set_xlabel("lambda (log)")
    ax[0].set_ylabel("MSE (log)")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")

    xx = np.linspace(0, 1, 400)
    prawdziwa_f = lambda x: np.sin(2 * np.pi * x)

    w_plain = dopasuj_wielomian(X_trp, y_trp, D_duzy)
    w_ridge = dopasuj_ridge(X_trp, y_trp, D_duzy, best_lam)

    ax[1].plot(xx, prawdziwa_f(xx), "g--", label="prawdziwa funkcja")
    ax[1].plot(
        xx, poly_features(xx, D_duzy) @ w_plain, "r", alpha=0.6, label="D=15 bez regularyzacji"
    )
    ax[1].plot(
        xx, poly_features(xx, D_duzy) @ w_ridge, "b", lw=2, label=f"Ridge, lambda={best_lam:.1e}"
    )
    ax[1].scatter(X_trp, y_trp, edgecolors="k", facecolors="none", label="train")

    ax[1].set_ylim(-2, 2)
    ax[1].set_title("Dopasowanie z regularyzacją L2")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

