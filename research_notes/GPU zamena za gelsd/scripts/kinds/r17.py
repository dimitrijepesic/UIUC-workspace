# Backward-style ratio R17 and residual ratio, computed exactly (Fractions) for gelsd and the closed form.
import sys
from fractions import Fraction as F
import torch
sys.path.insert(0, "/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/kinds")
dt = torch.float32 if sys.argv[1] == "f32" else torch.float64
torch.set_default_dtype(dt)
import importlib.util
spec = importlib.util.spec_from_file_location("kc", "/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/kinds/kinds_check.py")
src = open(spec.origin).read().split("torch.manual_seed(0)")[0]
ns = {}; exec(src, ns)
eps = F(torch.finfo(dt).eps)
def r17(g, z, b):
    X = [[F(1), F(p[0]), F(p[1])] for p in g.tolist()]
    zz = [F(v) for v in z.tolist()]; bb = [F(v) for v in b]
    r = [zz[i] - sum(X[i][j]*bb[j] for j in range(3)) for i in range(25)]
    Xtr = [sum(X[i][j]*r[i] for i in range(25)) for j in range(3)]
    n1X = max(sum(abs(X[i][j]) for i in range(25)) for j in range(3))
    n1r = sum(abs(v) for v in r)
    if n1r == 0: return 0.0, 0.0
    return float(sum(abs(v) for v in Xtr) / (n1X * n1r * 25 * eps)), float(sum(v*v for v in r))
n = 100
print(f"dtype={dt}")
print(f"{'kind':16s} {'R17 gelsd max':>14s} {'R17 CF max':>12s} {'res2 CF/exact max':>18s} {'res2 gelsd/exact max':>21s} {'max|coef| CF':>13s} {'max|coef| gelsd':>15s}")
for kind in ["normal", "degenerate_y", "near_degenerate", "mixed_scale"]:
    g, z = ns["make"](kind, n, 777, dt)
    X = torch.cat([torch.ones(n, 25, 1, dtype=dt), g], 2)
    sol = torch.linalg.lstsq(X, z.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
    cf = ns["closed_form"](g, z)
    rg = []; rc = []; qg = []; qc = []
    for i in range(n):
        ex = ns["exact"](g[i], z[i])
        a, ra = r17(g[i], z[i], sol[i].tolist()); b, rb = r17(g[i], z[i], cf[i].tolist())
        _, re = r17(g[i], z[i], [float(v) for v in ex])
        rg.append(a); rc.append(b)
        qc.append(rb / re if re > 0 else 1.0); qg.append(ra / re if re > 0 else 1.0)
    print(f"{kind:16s} {max(rg):14.3g} {max(rc):12.3g} {max(qc):18.6g} {max(qg):21.6g} {cf.abs().max().item():13.3e} {sol.abs().max().item():15.3e}")
