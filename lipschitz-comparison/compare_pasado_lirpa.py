"""Apples-to-apples comparison: Pasado Section 5.4 precise dual zonotope vs
auto_LiRPA (current master, CROWN-Optimized) on the SAME quantity.

The quantity. Pasado 5.4 bounds the derivative of the network output with
respect to a scalar haze parameter t:  x(t) = img + t*(1 - img),  t in
[0, epsilon]  (interpolation toward white). Its reported number per image is
    lc = max_j max(|lb_j|, |ub_j|)   over dF_j/dt for t in [0, epsilon],
averaged over 30 correctly-classified MNIST test images, for 16 epsilons
(get_lipschitz.py: n_splits=1, float64, no input normalization).

The reduction. x(t) is affine in t, so instead of bounding the full 784-dim
input Jacobian we express the haze map inside the model and ask auto_LiRPA
for the Jacobian w.r.t. the SCALAR t -- which IS dF/dt. The image enters as
a second, unperturbed input (their own local-Lipschitz example uses the same
multi-input pattern for its class mask), so one BoundedModule serves all
images and epsilons of a network.

Fairness notes, stated rather than hidden:
  - auto_LiRPA runs its default optimize=True (CROWN-Optimized) -- their
    strongest standard mode, per-neuron optimized relaxations.
  - Everything runs in float64, same as Pasado.
  - Pasado's lc_* numbers are read from Section_5_4/results/*.pth (produced
    by the real benchmark on this same machine). Its saved time_precise is
    reported alongside our measured per-image auto_LiRPA time, but timing
    comparability across sessions is weak -- tightness is the primary metric.

Run (from lipschitz-comparison/):
    .venv\\Scripts\\python.exe compare_pasado_lirpa.py --network 3layer
"""
import argparse
import csv
import gc
import os
import struct
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
PASADO_54 = os.path.normpath(os.path.join(HERE, "..", "Pasado", "Section_5_4"))

# Same 16 epsilons as get_lipschitz.py line 124.
EPSILONS = [10 ** (-k / 4) * 2 for k in range(2, 18)]
NUM_IMAGES = 30


def read_mnist_test_images():
    """Parse the raw MNIST idx file directly (16-byte header, then uint8
    pixels) and normalize by 255 -- bit-identical to torchvision's ToTensor,
    without needing torchvision in this venv."""
    path = os.path.join(PASADO_54, "MNIST_Data", "MNIST", "raw",
                        "t10k-images-idx3-ubyte")
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051 and rows == 28 and cols == 28, "not MNIST images"
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return torch.from_numpy(data.copy()).to(torch.float64) / 255.0


def build_network(name):
    """Replicate model.py's architectures exactly (self-contained here to
    avoid importing across repos), then load Pasado's trained weights."""
    if name == "big":
        net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024), nn.Sigmoid(),
            nn.Linear(1024, 1024), nn.Sigmoid(),
            nn.Linear(1024, 1024), nn.Sigmoid(),
            nn.Linear(1024, 1024), nn.Sigmoid(),
            nn.Linear(1024, 10),
        )
    else:
        layers = {"3layer": 3, "4layer": 4, "5layer": 5}[name]
        mods = [nn.Flatten(), nn.Linear(784, 100), nn.Sigmoid(),
                nn.Linear(100, 100), nn.Sigmoid()]
        if layers >= 4:
            mods += [nn.Linear(100, 100), nn.Sigmoid()]
        if layers == 5:
            mods += [nn.Linear(100, 100), nn.Sigmoid()]
        mods += [nn.Linear(100, 10)]
        net = nn.Sequential(*mods)

    state = torch.load(os.path.join(PASADO_54, "trained", f"model_{name}.pth"),
                       map_location="cpu")
    if name == "big":
        net.load_state_dict(state)
    else:
        # FCN checkpoints use attribute names (fc1.weight, ...); map them to
        # our Sequential indices in definition order.
        seq_linears = [m for m in net if isinstance(m, nn.Linear)]
        fc_names = ["fc1", "fc2", "fc3", "fc4"][:len(seq_linears) - 1] + ["fc_final"]
        mapped = {}
        for seq_layer, fc in zip(seq_linears, fc_names):
            prefix = None
            for i, m in enumerate(net):
                if m is seq_layer:
                    prefix = str(i)
            mapped[f"{prefix}.weight"] = state[f"{fc}.weight"]
            mapped[f"{prefix}.bias"] = state[f"{fc}.bias"]
        net.load_state_dict(mapped)
    net = net.to(torch.float64)
    net.eval()
    return net


def load_pasado_results(name):
    """Pasado's saved per-epsilon averages (the benchmark's own output)."""
    def load(kind):
        p = os.path.join(PASADO_54, "results", f"{kind}_{name}.pth")
        return torch.load(p, map_location="cpu") if os.path.exists(p) else None
    return {
        "interval": load("lc_intervals"),
        "zono": load("lc_zonos"),
        "precise": load("lc_precise"),
        "time_precise": load("time_precise"),
    }


class HazeJacobian(nn.Module):
    """t (1,1) perturbed scalar; img (1,784) plain tensor. x = img + t(1-img)
    reproduces get_lipschitz.py's hazed input exactly; JacobianOP w.r.t. t
    makes compute_jacobian_bounds return bounds on dF/dt -- Pasado's dual."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, img):
        from auto_LiRPA.jacobian import JacobianOP
        x = img + t * (1.0 - img)
        y = self.net(x)
        return JacobianOP.apply(y, t)


def run_one_pair(network, img_index, eps):
    """Compute one (image, epsilon) bound. Used as a subprocess entry point.

    Isolation is required, not stylistic. auto_LiRPA retains cached
    intermediate bounds and CROWN-Optimized alpha parameters per call, and on
    `big` one call peaks near 10.7 GB of this machine's 15.7 GB. Rebuilding the
    module in-process does not help: `del` + `gc.collect()` return the memory
    to Python's and torch's allocator pools rather than to the OS, so the next
    call still allocates on top of the retained ~10 GB and gets killed. Only
    process exit actually frees it, so the driver runs each pair in its own
    process.
    """
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    net = build_network(network)
    img = read_mnist_test_images()[img_index].unsqueeze(0)
    model = BoundedModule(HazeJacobian(net), (torch.zeros(1, 1), img),
                          device="cpu")
    t0 = torch.full((1, 1), eps / 2)
    bt = BoundedTensor(t0, PerturbationLpNorm(norm=float("inf"), eps=eps / 2))
    begin = time.perf_counter()
    lb, ub = model.compute_jacobian_bounds((bt, img))
    elapsed = time.perf_counter() - begin
    lc = torch.maximum(lb.abs(), ub.abs()).max().item()
    # Parsed by the driver; keep on one line and distinctive.
    print(f"PAIR_RESULT {lc!r} {elapsed!r}", flush=True)


def main():
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="3layer",
                    choices=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--num-images", type=int, default=NUM_IMAGES)
    ap.add_argument("--isolate", action="store_true",
                    help="Run every (image, epsilon) pair in its own "
                         "subprocess. Required for `big`, whose peak RSS makes "
                         "two consecutive solves impossible in one process.")
    ap.add_argument("--worker-image", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--worker-eps-index", type=int, default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker_image is not None:
        run_one_pair(args.network, args.worker_image,
                     EPSILONS[args.worker_eps_index])
        return

    net = build_network(args.network)
    images = read_mnist_test_images()
    correct_indices = torch.load(
        os.path.join(PASADO_54, "trained", f"indices_{args.network}.pth"),
        map_location="cpu")
    correct_set = set(int(i) for i in correct_indices)
    # Same selection rule as get_lipschitz.py: walk the test set in order,
    # keep the first N whose index is in correct_indices.
    selected = [i for i in range(images.shape[0]) if i in correct_set]
    selected = selected[:args.num_images]
    print(f"network={args.network}, images={len(selected)}, "
          f"epsilons={len(EPSILONS)}, dtype={torch.get_default_dtype()}")

    saved = load_pasado_results(args.network)

    out_path = os.path.join(HERE, f"results_{args.network}.csv")
    lc_sums = [0.0] * len(EPSILONS)
    t_sums = [0.0] * len(EPSILONS)
    done = 0

    # A fresh BoundedModule per (image, epsilon) pair.
    #
    # BoundedModule accumulates cached intermediate bounds and CROWN-Optimized
    # alpha parameters on every compute_jacobian_bounds call and never releases
    # them. On `big` (1024-wide) one call already peaks near 10.7 GB of this
    # machine's 15.7 GB, so the second call on the same module is OOM-killed --
    # which is what killed three earlier attempts, first across images and then
    # across epsilons within one image. Rebuilding per pair is the only order
    # that keeps peak memory at one call's worth. It costs a graph rebuild each
    # time, which is minor next to the ~108 s solve.
    counts = [0] * len(EPSILONS)
    total_pairs = len(selected) * len(EPSILONS)
    pair = 0
    for img_no, idx in enumerate(selected, start=1):
        img = images[idx].unsqueeze(0)
        for ei, eps in enumerate(EPSILONS):
            pair += 1
            # t in [0, eps] == Linf ball of radius eps/2 around eps/2.
            t0 = torch.full((1, 1), eps / 2)
            bt = BoundedTensor(t0, PerturbationLpNorm(
                norm=float("inf"), eps=eps / 2))
            begin = time.perf_counter()
            lc = None
            if args.isolate:
                proc = subprocess.run(
                    [sys.executable, os.path.abspath(__file__),
                     "--network", args.network,
                     "--worker-image", str(idx),
                     "--worker-eps-index", str(ei)],
                    capture_output=True, text=True)
                for line in proc.stdout.splitlines():
                    if line.startswith("PAIR_RESULT "):
                        _, lc_s, el_s = line.split()
                        lc, elapsed_inner = float(lc_s), float(el_s)
                if lc is None:
                    tail = (proc.stderr or proc.stdout)[-400:].replace("\n", " ")
                    print(f"  [img {img_no} eps={eps:.6f}] worker failed "
                          f"(exit {proc.returncode}): ...{tail}", flush=True)
            else:
                try:
                    model = BoundedModule(HazeJacobian(net),
                                          (torch.zeros(1, 1), img), device="cpu")
                    lb, ub = model.compute_jacobian_bounds((bt, img))
                    lc = torch.maximum(lb.abs(), ub.abs()).max().item()
                except Exception as e:
                    # Record the loss and keep going: one unlucky pair should
                    # not discard every result already collected.
                    print(f"  [img {img_no} eps={eps:.6f}] FAILED "
                          f"{type(e).__name__}: {e}", flush=True)
                    model = None
                finally:
                    del model
                    gc.collect()
            elapsed = time.perf_counter() - begin
            if lc is not None:
                lc_sums[ei] += lc
                t_sums[ei] += elapsed
                counts[ei] += 1
            print(f"  [{pair}/{total_pairs}] img {img_no} eps={eps:.6f} "
                  f"lirpa={lc if lc is None else round(lc, 3)} "
                  f"({elapsed:.1f}s)", flush=True)

            # Rewritten after every single pair, so a kill costs at most the
            # pair in flight. `images_averaged` records how many images stand
            # behind each row, since rows fill in at different rates.
            rows = []
            for k, e_k in enumerate(EPSILONS):
                rows.append({
                    "epsilon": e_k,
                    "images_averaged": counts[k],
                    "pasado_interval": saved["interval"][k] if saved["interval"] else None,
                    "pasado_zono": saved["zono"][k] if saved["zono"] else None,
                    "pasado_precise": saved["precise"][k] if saved["precise"] else None,
                    "lirpa_opt": (lc_sums[k] / counts[k]) if counts[k] else None,
                    "pasado_time_precise": saved["time_precise"][k] if saved["time_precise"] else None,
                    "lirpa_time": (t_sums[k] / counts[k]) if counts[k] else None,
                })
            write_csv(out_path, rows)

        print(f"--- image {img_no}/{len(selected)} done (idx {idx}) ---", flush=True)
        for r in rows:
            pp, lo = r["pasado_precise"], r["lirpa_opt"]
            if lo is None:
                continue
            ratio = (lo / pp) if pp else float("nan")
            print(f"  eps={r['epsilon']:.6f}  precise={pp:.6g}  lirpa={lo:.6g}  "
                  f"ratio={ratio:.3f}  n={r['images_averaged']}", flush=True)

    print(f"\nwrote {out_path}")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
