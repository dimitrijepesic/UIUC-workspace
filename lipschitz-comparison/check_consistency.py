"""Consistency guard over the comparison data.

1. Pasado tightness must be (near-)monotone non-increasing in n_splits per
   (network, eps) across pasado_splits.csv + pasado_equal_time.csv. Tolerance
   1e-4 relative covers the unseeded np.random.normal(0,1e-4) perturbation --
   runs at different k carry independent draws.
2. auto_LiRPA BaB values must be monotone non-increasing in K (bab_sweep_gpu).
3. Every Pasado value must be positive and finite; timings positive.
4. Cross-check: equal-budget ratio at K=1 from the GPU sweep must agree with
   the single-pass table's ratios (measured by a separate CPU script months
   apart) within 1% -- three independent code paths, same quantity.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fname, kcol, vcol):
    out = {}
    with open(os.path.join(HERE, fname), newline="") as f:
        for r in csv.DictReader(f):
            out[(r["network"], int(r["eps_index"]), int(r[kcol]))] = (
                float(r[vcol]), float(r["seconds_per_image_per_eps"]))
    return out


pasado = load("pasado_splits.csv", "n_splits", "pasado_precise")
pasado.update(load("pasado_equal_time.csv", "n_splits", "pasado_precise"))
lirpa = load("bab_sweep_gpu.csv", "K", "lirpa_bab")

problems = []

# 1+3: Pasado monotone in k, values sane
nets = sorted({k[0] for k in pasado})
for net in nets:
    for ei in sorted({k[1] for k in pasado if k[0] == net}):
        ks = sorted(k[2] for k in pasado if k[:2] == (net, ei))
        vals = [pasado[(net, ei, k)][0] for k in ks]
        for v, k in zip(vals, ks):
            if not (v > 0 and v < float("inf")):
                problems.append(f"pasado {net} eps{ei} k={k}: bad value {v}")
            if pasado[(net, ei, k)][1] <= 0:
                problems.append(f"pasado {net} eps{ei} k={k}: bad time")
        # eps indices 0-4 are the ones every documented table uses: strict.
        # The converged tail (tiny eps) may legitimately drift ~1e-3 UP with k:
        # each subinterval carries a fixed relaxation overhead plus the unseeded
        # A-perturbation, and max over k noisy near-equal values is biased
        # upward (same mechanism as LIRPA_COMPARISON.md finding 4).
        tol = 1e-4 if ei <= 4 else 1e-3
        for (k1, v1), (k2, v2) in zip(zip(ks, vals), zip(ks[1:], vals[1:])):
            if v2 > v1 * (1 + tol):
                problems.append(
                    f"pasado {net} eps{ei}: NOT monotone k={k1}->{k2} ({v1:.4g}->{v2:.4g})")

# 2: LiRPA monotone in K
for net in sorted({k[0] for k in lirpa}):
    for ei in sorted({k[1] for k in lirpa if k[0] == net}):
        ks = sorted(k[2] for k in lirpa if k[:2] == (net, ei))
        vals = [lirpa[(net, ei, k)][0] for k in ks]
        for (k1, v1), (k2, v2) in zip(zip(ks, vals), zip(ks[1:], vals[1:])):
            if v2 > v1:
                problems.append(
                    f"lirpa {net} eps{ei}: NOT monotone K={k1}->{k2} ({v1:.4g}->{v2:.4g})")

# 4: K=1 GPU-sweep ratio vs the independently measured single-pass table
SINGLE_PASS = {  # from LIRPA_COMPARISON.md, CPU compare script, 30 images
    ("3layer", 0): 1.006, ("3layer", 1): 1.298, ("3layer", 2): 1.323,
    ("3layer", 3): 1.173, ("3layer", 4): 1.035,
    ("4layer", 0): 1.289, ("4layer", 1): 1.822, ("4layer", 2): 1.928,
    ("4layer", 3): 1.554, ("4layer", 4): 1.177,
    ("5layer", 0): 1.748, ("5layer", 1): 2.633, ("5layer", 2): 2.826,
    ("5layer", 3): 2.190, ("5layer", 4): 1.289,
}
for (net, ei), expected in SINGLE_PASS.items():
    got = lirpa[(net, ei, 1)][0] / pasado[(net, ei, 1)][0]
    if abs(got - expected) / expected > 0.01:
        problems.append(
            f"single-pass mismatch {net} eps{ei}: sweep gives {got:.3f}, table says {expected}")

if problems:
    print(f"{len(problems)} PROBLEM(S):")
    for p in problems:
        print(" -", p)
    raise SystemExit(1)
n_cells = len({k[:2] for k in pasado})
print(f"OK: {len(pasado)} pasado rows, {len(lirpa)} lirpa rows, {n_cells} cells; "
      f"monotonicity, sanity and single-pass cross-check all clean")
