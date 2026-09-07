"""Merge pasado_splits.csv (Pasado n_splits sweep, CPU) with bab_sweep_gpu.csv
(auto_LiRPA BaB sweep, A100) into a single ratio table and find, per
(network, eps), the budget K at which the curves cross.

ratio = lirpa_bab / pasado_precise  (>1 means Pasado is tighter)

Crossover: last K where Pasado leads and first K where auto_LiRPA leads,
plus a linear interpolation of ratio(K)=1 between those two budgets.
Writes bab_crossover.csv and prints a readable table.
"""
import csv
from collections import defaultdict

pasado = {}
with open("pasado_splits.csv", newline="") as f:
    for row in csv.DictReader(f):
        key = (row["network"], int(row["eps_index"]), int(row["n_splits"]))
        pasado[key] = float(row["pasado_precise"])

lirpa = {}
eps_of = {}
with open("bab_sweep_gpu.csv", newline="") as f:
    for row in csv.DictReader(f):
        key = (row["network"], int(row["eps_index"]), int(row["K"]))
        lirpa[key] = float(row["lirpa_bab"])
        eps_of[(row["network"], int(row["eps_index"]))] = float(row["epsilon"])

KS = [1, 3, 5, 7, 9]
NETS = ["3layer", "4layer", "5layer"]
EPS_IDX = sorted({k[1] for k in lirpa})

rows = []
for net in NETS:
    for ei in EPS_IDX:
        ratios = {}
        for k in KS:
            if (net, ei, k) in pasado and (net, ei, k) in lirpa:
                ratios[k] = lirpa[(net, ei, k)] / pasado[(net, ei, k)]
        if not ratios:
            continue
        # crossover: ratio drops below 1 between consecutive budgets
        cross = None
        ks = sorted(ratios)
        if ratios[ks[0]] <= 1.0:
            cross = f"<={ks[0]}"
        elif ratios[ks[-1]] > 1.0:
            cross = f">{ks[-1]}"
        else:
            for a, b in zip(ks, ks[1:]):
                ra, rb = ratios[a], ratios[b]
                if ra > 1.0 >= rb:
                    # linear interpolation in K for ratio = 1
                    kx = a + (ra - 1.0) / (ra - rb) * (b - a)
                    cross = f"~{kx:.1f}"
                    break
        rows.append((net, ei, eps_of[(net, ei)], ratios, cross))

with open("bab_crossover.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["network", "eps_index", "epsilon"]
               + [f"ratio_K{k}" for k in KS] + ["crossover_K"])
    for net, ei, eps, ratios, cross in rows:
        w.writerow([net, ei, f"{eps:.6g}"]
                   + [f"{ratios.get(k, float('nan')):.3f}" for k in KS]
                   + [cross])

hdr = f"{'net':8}{'eps':>9} |" + "".join(f"{'K='+str(k):>9}" for k in KS) + " | cross"
print(hdr)
print("-" * len(hdr))
for net, ei, eps, ratios, cross in rows:
    cells = "".join(f"{ratios.get(k, float('nan')):9.3f}" for k in KS)
    print(f"{net:8}{eps:9.4g} |{cells} | {cross}")
