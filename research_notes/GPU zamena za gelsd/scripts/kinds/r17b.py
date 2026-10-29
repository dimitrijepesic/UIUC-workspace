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
    n1b = sum(abs(v) for v in bb)
    a = float(sum(abs(v) for v in Xtr) / (n1X * n1r * 25 * eps)) if n1r else 0.0
    bnorm = float(sum(abs(v) for v in Xtr) / (n1X * n1X * n1b * 25 * eps)) if n1b else 0.0
    return a, bnorm
n = 100
print(f"dtype={dt}")

for kind in ["normal", "degenerate_y", "near_degenerate", "mixed_scale"]:
    g, z = ns["make"](kind, n, 777, dt)
    X = torch.cat([torch.ones(n, 25, 1, dtype=dt), g], 2)
    sol = torch.linalg.lstsq(X, z.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
    cf = ns["closed_form"](g, z)
    out = {"gelsd": [], "CF": [], "round(exact)": []}
    for i in range(n):
        ex = [float(v) for v in ns["exact"](g[i], z[i])]
        if dt == torch.float32: ex = torch.tensor(ex, dtype=torch.float32).tolist()
        for name, b in (("gelsd", sol[i].tolist()), ("CF", cf[i].tolist()), ("round(exact)", ex)):
            out[name].append(r17(g[i], z[i], b))
    cells = "  ".join(f"{k}: R17r={max(v[0] for v in out[k]):.2g} R17b={max(v[1] for v in out[k]):.2g}" for k in out)
    print(f"{kind:16s} {cells}")
