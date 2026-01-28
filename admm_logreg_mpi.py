import os
import argparse
import numpy as np
from mpi4py import MPI

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    y = data[:, 0]
    X = data[:, 1:]
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # add intercept
    return X, y

def local_update(X, y, z, u, rho, max_inner=5):
    w = z - u
    for _ in range(max_inner):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) + rho * (w - z + u)
        H = X.T @ np.diag(p * (1 - p)) @ X + rho * np.eye(X.shape[1])
        w -= np.linalg.solve(H, grad)
    return w

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=100)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.endswith(".csv")
    ])

    local_files = files[rank::size]

    Xs, ys = [], []
    for f in local_files:
        X, y = load_csv(f)
        Xs.append(X)
        ys.append(y)

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    d = X.shape[1]
    z = np.zeros(d)
    u = np.zeros(d)
    w = np.zeros(d)

    for _ in range(args.max_iters):
        w = local_update(X, y, z, u, args.r_
