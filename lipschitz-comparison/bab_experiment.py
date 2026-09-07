"""Does Branch-and-Bound close auto_LiRPA's gap to Pasado -- and at what cost?

The comparison so far ran auto_LiRPA in its strongest *single-pass* mode
(CROWN-Optimized) and found Pasado tighter by 1.3-2.8x. The obvious objection,
and the first thing a reviewer will raise, is that Shi et al.'s method is meant
to be used with Branch-and-Bound: split the input domain, bound each piece, take
the max. Splitting always tightens, so with enough branching auto_LiRPA must
eventually match or beat any fixed single-pass bound.

So the interesting question is not "can BaB close the gap" (it can, in the
limit) but **how many subproblems it takes**. That number is the honest price
of the gap, and it is what this script measures:

    Pasado, 1 solve, no splitting        ->  bound P
    auto_LiRPA, K solves with BaB        ->  bound L(K)
    smallest K with L(K) <= P            ->  the cost of catching up

Why input-domain branching is the right form here. Pasado's own benchmark has
the same knob -- `get_lipschitz.py` splits t into `n_splits` sub-intervals and
takes the max -- and runs it at `n_splits = 1`. The perturbed input is the
single scalar t, so bisecting [0, eps] is the natural and complete branching
strategy; there are no ReLU neurons to split on in a sigmoid network. Giving
auto_LiRPA the splits while Pasado stays unsplit deliberately favours
auto_LiRPA.

Branching is best-first, not uniform: at each step the sub-interval with the
current largest upper bound is bisected, since only that one can be responsible
for the global max. Uniform bisection would be a weaker strawman.
"""
import argparse
import csv
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
PASADO_54 = os.path.normpath(os.path.join(HERE, "..", "Pasado", "Section_5_4"))
EPSILONS = [10 ** (-k / 4) * 2 for k in range(2, 18)]

# Reuse the exact model/data/quantity definitions from the single-pass
# comparison so the two experiments cannot silently diverge.
from compare_pasado_lirpa import (  # noqa: E402
    build_network, read_mnist_test_images, load_pasado_results, HazeJacobian)


def solve_interval(net, img, t_lo, t_hi, device="cpu"):
    """max_j max(|lb_j|, |ub_j|) over dF_j/dt for t in [t_lo, t_hi].

    Same quantity as the single-pass comparison, just over an arbitrary
    sub-interval instead of all of [0, eps]. A fresh BoundedModule per call is
    required, not merely tidy: auto_LiRPA accumulates cached intermediate
    bounds and alpha parameters per call and never frees them, and BaB makes
    many calls.
    """
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    center = (t_lo + t_hi) / 2.0
    radius = (t_hi - t_lo) / 2.0
    model = BoundedModule(HazeJacobian(net), (torch.zeros(1, 1), img),
                          device=device)
    bt = BoundedTensor(torch.full((1, 1), center),
                       PerturbationLpNorm(norm=float("inf"), eps=radius))
    lb, ub = model.compute_jacobian_bounds((bt, img))
    return torch.maximum(lb.abs(), ub.abs()).max().item()


def bab_curve(net, img, eps, budget, device="cpu"):
    """Best-first BaB over t in [0, eps]; returns {solves_used: bound}.

    The global bound is the max over all live sub-intervals, so it is sound at
    every step -- the curve can be read off at any budget, not just the last.
    Bisecting the current worst interval is what makes this BaB rather than a
    uniform grid: effort goes where the maximum actually is.
    """
    b0 = solve_interval(net, img, 0.0, eps, device)
    domains = [(b0, 0.0, eps)]     # (upper bound on this piece, lo, hi)
    curve = {1: b0}
    solves = 1

    while solves + 2 <= budget:
        # Only the worst piece can hold the global max, so split that one.
        i = max(range(len(domains)), key=lambda k: domains[k][0])
        _, lo, hi = domains.pop(i)
        mid = (lo + hi) / 2.0
        left = solve_interval(net, img, lo, mid, device)
        right = solve_interval(net, img, mid, hi, device)
        solves += 2
        domains.append((left, lo, mid))
        domains.append((right, mid, hi))
        curve[solves] = max(d[0] for d in domains)

    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="5layer",
                    choices=["3layer", "4layer", "5layer", "big"])
    # Default is the epsilon where the single-pass gap peaked for 3/4/5layer.
    ap.add_argument("--eps-index", type=int, default=2,
                    help=f"index into {['%.4g' % e for e in EPSILONS]}")
    ap.add_argument("--budget", type=int, default=31,
                    help="max solves per image; the schedule is 1,3,5,... so "
                         "an odd budget is used exactly")
    ap.add_argument("--num-images", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    eps = EPSILONS[args.eps_index]
    net = build_network(args.network).to(args.device)
    images = read_mnist_test_images()
    correct = set(int(i) for i in torch.load(
        os.path.join(PASADO_54, "trained", f"indices_{args.network}.pth"),
        map_location="cpu"))
    selected = [i for i in range(images.shape[0]) if i in correct][:args.num_images]

    saved = load_pasado_results(args.network)
    pasado = saved["precise"][args.eps_index]

    # Pasado's reference number is an average over exactly 30 images, so any
    # other count compares different quantities -- the same trap that
    # invalidated the first `big` runs.
    if args.num_images != 30:
        print(f"\n*** WARNING: --num-images {args.num_images} != 30; the "
              f"Pasado reference is a 30-image average, so the reported "
              f"crossover K is NOT valid. ***\n", flush=True)

    print(f"network={args.network}  eps={eps:.6g}  budget={args.budget} "
          f"solves/image  images={len(selected)}")
    print(f"Pasado precise (1 solve, n_splits=1): {pasado:.6g}\n", flush=True)

    # Solve counts every image's curve passes through: 1, 3, 5, ...
    schedule = [k for k in range(1, args.budget + 1, 2)]
    sums = {k: 0.0 for k in schedule}
    done = 0
    out_path = os.path.join(HERE, f"bab_{args.network}_eps{args.eps_index}.csv")

    for img_no, idx in enumerate(selected, start=1):
        img = images[idx].unsqueeze(0).to(args.device)
        begin = time.perf_counter()
        curve = bab_curve(net, img, eps, args.budget, args.device)
        for k in schedule:
            sums[k] += curve[k]
        done += 1

        rows = [{
            "solves": k,
            "images_averaged": done,
            "lirpa_bab": sums[k] / done,
            "pasado_precise_1_solve": pasado,
            "ratio": (sums[k] / done) / pasado,
        } for k in schedule]
        # Written per image: a kill costs one image, not the whole run.
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        crossover = next((r["solves"] for r in rows if r["ratio"] <= 1.0), None)
        print(f"[image {img_no}/{len(selected)}] {time.perf_counter()-begin:.0f}s  "
              f"K=1: {rows[0]['ratio']:.3f}x  "
              f"K={schedule[-1]}: {rows[-1]['ratio']:.3f}x  "
              f"crossover K={crossover}", flush=True)

    print(f"\n=== {args.network}, eps={eps:.6g}, {done} images ===")
    print(f"{'solves':>7} {'lirpa+BaB':>12} {'ratio':>8}")
    for r in rows:
        print(f"{r['solves']:>7} {r['lirpa_bab']:>12.6g} {r['ratio']:>8.3f}")

    crossover = next((r["solves"] for r in rows if r["ratio"] <= 1.0), None)
    if crossover is None:
        print(f"\nauto_LiRPA does NOT reach Pasado's single-solve bound within "
              f"{args.budget} solves per image "
              f"(best {rows[-1]['ratio']:.3f}x at {schedule[-1]} solves).")
    else:
        print(f"\nauto_LiRPA needs {crossover} solves per image to match "
              f"Pasado's 1 solve.")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
