"""Equal-time comparison: Pasado (CPU, uniform n_splits) vs auto_LiRPA+BaB (A100).

Inputs:
  pasado_splits.csv      Pasado k in {1,3,5,7,9}   (value + measured s/img/eps)
  pasado_equal_time.csv  Pasado k in {16,64,256}   (value + measured s/img/eps)
  bab_sweep_gpu.csv      auto_LiRPA K in {1..9}; its seconds_per_image_per_eps
                         column is the A100 total for all 25 solves of the K
                         sweep, so per-solve = that/25 and time(K) = K*per-solve.

For each (network, eps): find the smallest Pasado k whose bound beats
auto_LiRPA's best measured budget (K=9), and compare the wall-clock both sides
spend at that point. Also a head-to-head at auto_LiRPA's K=9 time budget.

Writes equal_time_summary.csv and prints the tables.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KS_LIRPA = [1, 3, 5, 7, 9]
SOLVES_PER_SWEEP = sum(KS_LIRPA)  # 25: the GPU s/img/eps covers all budgets


def load(fname, key_val):
    out = {}
    with open(os.path.join(HERE, fname), newline="") as f:
        for r in csv.DictReader(f):
            out[(r["network"], int(r[key_val]), int(r["eps_index"]))] = (
                float(r["pasado_precise" if "pasado" in fname else "lirpa_bab"]),
                float(r["seconds_per_image_per_eps"]))
    return out


pasado = load("pasado_splits.csv", "n_splits")
pasado.update(load("pasado_equal_time.csv", "n_splits"))
lirpa = load("bab_sweep_gpu.csv", "K")

NETS = ["3layer", "4layer", "5layer"]
EPS_IDX = [0, 1, 2, 3, 4]
PASADO_KS = sorted({k for (_, k, _) in pasado})

rows = []
for net in NETS:
    per_solve = lirpa[(net, 1, 0)][1] / SOLVES_PER_SWEEP
    for ei in EPS_IDX:
        eps = [10 ** (-k / 4) * 2 for k in range(2, 18)][ei]
        l9_val = lirpa[(net, 9, ei)][0]
        l9_time = 9 * per_solve
        # smallest Pasado k that already beats auto_LiRPA's best budget
        beat = next(((k,) + pasado[(net, k, ei)] for k in PASADO_KS
                     if (net, k, ei) in pasado and pasado[(net, k, ei)][0] <= l9_val),
                    None)
        # head-to-head at (roughly) the same wall-clock: largest Pasado k
        # whose measured time still fits within auto_LiRPA's K=9 budget
        fit = max((k for k in PASADO_KS if (net, k, ei) in pasado
                   and pasado[(net, k, ei)][1] <= l9_time), default=None)
        fit_val, fit_time = pasado[(net, fit, ei)] if fit else (float("nan"),) * 2
        k_star, p_val, p_time = beat if beat else (None, float("nan"), float("nan"))
        rows.append({
            "network": net, "eps_index": ei, "epsilon": round(eps, 5),
            "lirpa_K9": round(l9_val, 1), "lirpa_K9_time_s": round(l9_time, 2),
            "pasado_k_beats_K9": k_star,
            "pasado_val_at_k": round(p_val, 1),
            "pasado_time_at_k_s": round(p_time, 3),
            "time_advantage_x": round(l9_time / p_time, 1) if beat else None,
            "pasado_k_within_K9_time": fit,
            "pasado_val_within_K9_time": round(fit_val, 1),
            "tightness_ratio_at_equal_time": round(l9_val / fit_val, 2) if fit else None,
        })

out = os.path.join(HERE, "equal_time_summary.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("Smallest Pasado k beating auto_LiRPA K=9, and head-to-head at K=9's time budget")
print(f"{'net':8}{'eps':>8} | {'L K=9':>7} {'@s':>6} | {'k*':>4} {'val':>7} {'@s':>7} "
      f"{'adv':>6} | {'k<=T':>5} {'val':>7} {'ratio':>6}")
for r in rows:
    print(f"{r['network']:8}{r['epsilon']:8.4g} | {r['lirpa_K9']:7} {r['lirpa_K9_time_s']:6} | "
          f"{str(r['pasado_k_beats_K9']):>4} {r['pasado_val_at_k']:7} {r['pasado_time_at_k_s']:7} "
          f"{str(r['time_advantage_x']):>6} | {str(r['pasado_k_within_K9_time']):>5} "
          f"{r['pasado_val_within_K9_time']:7} {str(r['tightness_ratio_at_equal_time']):>6}")
print(f"\nwrote {out}")
