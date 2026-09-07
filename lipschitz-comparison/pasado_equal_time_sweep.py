"""Pasado side of the EQUAL-TIME comparison: large n_splits values, on CPU.

The equal-budget sweep (pasado_splits.csv vs bab_sweep_gpu.csv) compares the
two methods at the same number of subproblems, which favours the expensive
side: one auto_LiRPA subproblem on an A100 costs ~0.57/0.97/1.49 s
(3/4/5layer) while one Pasado subproblem costs ~0.019/0.028/0.041 s on this
CPU -- a ~30x asymmetry. At equal wall-clock, Pasado can afford ~30x more
splits. This sweep measures Pasado at n_splits {16, 64, 256}; k=256 lands at
roughly auto_LiRPA's K=7..9 time budget per network, so if Pasado at k=256
beats auto_LiRPA at K=9, equal-time domination holds across the measured
range. Together with pasado_splits.csv (k in {1,3,5,7,9}) this gives the
tightness-vs-time curve.

Reuses the machinery of pasado_splits_sweep.py: get_lipschitz.py runs as a
subprocess, unmodified apart from its --n-splits / --eps-indices flags, so
there is no second implementation of Pasado's quantity.

    python pasado_equal_time_sweep.py
    python pasado_equal_time_sweep.py --networks 3layer --splits 16 --eps-indices 2
"""
import argparse
import ast
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEC = os.path.normpath(os.path.join(HERE, "..", "Pasado", "Section_5_4"))
PY = os.path.normpath(os.path.join(HERE, "..", "Pasado", ".venv", "Scripts", "python.exe"))
EPSILONS = [10 ** (-k / 4) * 2 for k in range(2, 18)]


def run_pasado(network, n_splits, eps_indices):
    """Return (lc_precise list, time_precise list), one entry per eps index."""
    env = dict(os.environ)
    env["PASADO_VECTORIZED_PRECISE"] = "1"
    env["PASADO_BATCHED_LSTSQ"] = "1"
    proc = subprocess.run(
        [PY, "get_lipschitz.py", "--network", network,
         "--n-splits", str(n_splits), "--no-save",
         "--eps-indices", *[str(i) for i in eps_indices]],
        cwd=SEC, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"get_lipschitz.py failed for {network} k={n_splits}:\n"
                           + proc.stdout[-2000:] + proc.stderr[-2000:])

    def grab(tag):
        m = re.search(rf"^{tag} (\[.*?\])", proc.stdout, re.S | re.M)
        if not m:
            raise RuntimeError(f"could not parse '{tag}' from get_lipschitz.py output")
        return ast.literal_eval(m.group(1))

    lc, tm = grab("lc_precise"), grab("time_precise")
    if len(lc) != len(eps_indices):
        raise RuntimeError(f"expected {len(eps_indices)} epsilons, got {len(lc)} "
                           f"-- is --eps-indices wired correctly?")
    return lc, tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+",
                    default=["3layer", "4layer", "5layer"],
                    choices=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--splits", nargs="+", type=int, default=[16, 64, 256])
    ap.add_argument("--eps-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="must be a subset of what the auto_LiRPA sweep covered")
    ap.add_argument("--out", default=os.path.join(HERE, "pasado_equal_time.csv"))
    args = ap.parse_args()

    rows = []
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            rows = list(csv.DictReader(f))
    done = {(r["network"], int(r["n_splits"])) for r in rows}

    for network in args.networks:
        for k in args.splits:
            if (network, k) in done:
                print(f"skip {network} k={k} (already done)")
                continue
            begin = time.perf_counter()
            lc, tm = run_pasado(network, k, args.eps_indices)
            elapsed = time.perf_counter() - begin
            for j, ei in enumerate(args.eps_indices):
                rows.append({
                    "network": network, "n_splits": k, "eps_index": ei,
                    "epsilon": EPSILONS[ei], "pasado_precise": lc[j],
                    "seconds_per_image_per_eps": tm[j],
                })
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"[{network} k={k}] {elapsed:.0f}s total, "
                  f"first eps -> {lc[0]:.6g} ({tm[0]:.3f}s/img)", flush=True)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
