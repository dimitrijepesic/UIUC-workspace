"""Pasado side of the equal-budget sweep: n_splits x networks, on CPU.

The GPU sweep (experiments/colab_bab_sweep.py) measures auto_LiRPA+BaB at
budgets 1,3,5,... This produces the matching Pasado column by running the
benchmark's own splitting knob at the same budgets, so the two can be compared
subproblem-for-subproblem rather than one-split-against-many.

Runs get_lipschitz.py as a subprocess -- the real benchmark, unmodified apart
from the --n-splits flag and the printed result lists -- so there is no second
implementation of Pasado's quantity that could drift from the first.

    python pasado_splits_sweep.py --networks 3layer 4layer 5layer --splits 1 3 5 7 9
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


def run_pasado(network, n_splits):
    """Return (lc_precise list, time_precise list) for all 16 epsilons."""
    env = dict(os.environ)
    # Match the optimized configuration the rest of the project measures.
    env["PASADO_VECTORIZED_PRECISE"] = "1"
    env["PASADO_BATCHED_LSTSQ"] = "1"
    proc = subprocess.run(
        [PY, "get_lipschitz.py", "--network", network,
         "--n-splits", str(n_splits), "--no-save"],
        cwd=SEC, env=env, capture_output=True, text=True,
        # tqdm's progress bar emits bytes the console's cp1250 codec cannot
        # decode, which otherwise kills the read with UnicodeDecodeError before
        # the result lines are seen.
        encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"get_lipschitz.py failed for {network} k={n_splits}:\n"
                           + proc.stdout[-2000:] + proc.stderr[-2000:])

    def grab(tag):
        # The lists are printed across several lines, so match up to the bracket.
        m = re.search(rf"^{tag} (\[.*?\])", proc.stdout, re.S | re.M)
        if not m:
            raise RuntimeError(f"could not parse '{tag}' from get_lipschitz.py output")
        return ast.literal_eval(m.group(1))

    return grab("lc_precise"), grab("time_precise")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+",
                    default=["3layer", "4layer", "5layer"],
                    choices=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 3, 5, 7, 9],
                    help="must match the budgets used on the auto_LiRPA side")
    ap.add_argument("--out", default=os.path.join(HERE, "pasado_splits.csv"))
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
            lc, tm = run_pasado(network, k)
            elapsed = time.perf_counter() - begin
            for ei, eps in enumerate(EPSILONS):
                rows.append({
                    "network": network, "n_splits": k, "eps_index": ei,
                    "epsilon": eps, "pasado_precise": lc[ei],
                    "seconds_per_image_per_eps": tm[ei],
                })
            # Written after each (network, k): a long sweep never loses finished work.
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"[{network} k={k}] {elapsed:.0f}s total, "
                  f"eps=0.2 -> {lc[2]:.6g} ({tm[2]:.3f}s/img)", flush=True)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
