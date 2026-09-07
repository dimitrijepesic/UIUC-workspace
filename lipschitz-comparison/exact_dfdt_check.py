"""Ground-truth check: which bound is loose at eps -> 0 on FFNNBig?

On `big`, the Pasado/auto_LiRPA ratio does NOT converge to 1 as eps shrinks
(it flattens near 1.46: precise ~168.11 vs lirpa ~245.59 at eps=1.12e-4),
unlike 3/4/5layer where both methods converge. Since both are sound upper
bounds on max_j |dF_j/dt| over t in [0, eps], and that quantity converges to
the exact |dF_j/dt| at the point as eps -> 0, they cannot converge to
different limits unless one is not tight -- or one is unsound.

This computes the exact value with autograd at the same point, on the same
image, so the disagreement is settled by a number rather than an argument:

  exact ~= 168  -> Pasado tight, auto_LiRPA carries ~46% irreducible slack
  exact ~= 245  -> Pasado's bound sits BELOW the truth, i.e. unsound
  exact > 245   -> both are loose
"""
import os
import struct

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
SEC = os.path.normpath(os.path.join(HERE, "..", "Pasado", "Section_5_4"))
EPSILONS = [10 ** (-k / 4) * 2 for k in range(2, 18)]


def read_images():
    path = os.path.join(SEC, "MNIST_Data", "MNIST", "raw",
                        "t10k-images-idx3-ubyte")
    with open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)
    return torch.from_numpy(data.copy()).to(torch.float64) / 255.0


def build_big():
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1024), nn.Sigmoid(),
        nn.Linear(1024, 1024), nn.Sigmoid(),
        nn.Linear(1024, 1024), nn.Sigmoid(),
        nn.Linear(1024, 1024), nn.Sigmoid(),
        nn.Linear(1024, 10),
    )
    net.load_state_dict(torch.load(os.path.join(SEC, "trained", "model_big.pth"),
                                   map_location="cpu"))
    return net.to(torch.float64).eval()


def main():
    net = build_big()
    images = read_images()
    correct = set(int(i) for i in torch.load(
        os.path.join(SEC, "trained", "indices_big.pth"), map_location="cpu"))
    idx = next(i for i in range(images.shape[0]) if i in correct)
    img = images[idx].unsqueeze(0)
    print(f"image index {idx} (first correctly classified, same as the comparison)")

    def max_abs_dfdt(t_val):
        t = torch.tensor([[t_val]], requires_grad=True)
        y = net(img + t * (1.0 - img))          # [1, 10]
        # One backward per output class: dF_j/dt for each j.
        grads = []
        for j in range(y.shape[1]):
            g, = torch.autograd.grad(y[0, j], t, retain_graph=True)
            grads.append(g.item())
        return max(abs(g) for g in grads), grads

    # Sample across t in [0, eps] for the smallest and a mid epsilon, so we
    # report the true max over the interval, not just at one point.
    for eps in (EPSILONS[-1], EPSILONS[6], EPSILONS[2]):
        ts = torch.linspace(0.0, eps, 21).tolist()
        best = max(max_abs_dfdt(t)[0] for t in ts)
        print(f"eps={eps:.6g}: exact max_j |dF_j/dt| over 21 samples in "
              f"[0, eps] = {best:.4f}")

    m0, grads0 = max_abs_dfdt(0.0)
    print(f"\nat t=0 exactly: max_j |dF_j/dt| = {m0:.4f}")
    print("per-class dF_j/dt:", " ".join(f"{g:.2f}" for g in grads0))
    print("\nreference bounds from the comparison at eps=1.12e-4: "
          "Pasado precise=168.11, auto_LiRPA=245.59")


if __name__ == "__main__":
    main()
