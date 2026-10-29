import sys, math, torch
src = open("/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/kinds/kinds_check.py").read().split("def exact")[0]
ns = {}; exec(src, ns); cf = ns["closed_form"]
def box(lx, ux, ly, uy, dt):
    xs = torch.linspace(lx, ux, 5, dtype=dt); ys = torch.linspace(ly, uy, 5, dtype=dt)
    g = torch.cartesian_prod(xs, ys).unsqueeze(0)
    s = torch.sigmoid(g[..., 0]); z = (1 - s) * (s * g[..., 1])
    return g, z
def gelsd(g, z):
    X = torch.cat([torch.ones(1, 25, 1, dtype=g.dtype), g], 2)
    r = torch.linalg.lstsq(X, z.unsqueeze(-1), driver="gelsd")
    return r.solution.squeeze(-1), r.rank
cases = [
  ("f32 tiny nonzero width (sxx underflow)", 1e-30, 1e-30 + 1e-37, -1.0, 1.0, torch.float32),
  ("f64 tiny nonzero width", 1e-200, 1e-200 + 1e-210, -1.0, 1.0, torch.float64),
  ("f64 degenerate x0=1e155", 1e155, 1e155, -1.0, 1.0, torch.float64),
  ("f32 degenerate x0=1e20", 1e20, 1e20, -1.0, 1.0, torch.float32),
  ("f64 endpoints 1 ulp apart, x~30", 30.0, math.nextafter(30.0, 31.0), -1.0, 1.0, torch.float64),
  ("f64 -0.0 / +0.0 x", -0.0, 0.0, -1.0, 1.0, torch.float64),
]
for name, lx, ux, ly, uy, dt in cases:
    g, z = box(lx, ux, ly, uy, dt)
    b = cf(g, z)
    try:
        s, rk = gelsd(g, z); gs = f"gelsd={s[0].tolist()} rank={int(rk[0])}"
    except Exception as e:
        gs = f"gelsd raised {type(e).__name__}: {e}"
    print(f"{name}: CF={b[0].tolist()}  {gs}")
# NaN isolation: one NaN box in a batch of 3
g1, z1 = box(0.1, 0.5, -1.0, 1.0, torch.float64); g2, z2 = box(1.0, 2.0, 0.0, 3.0, torch.float64)
g = torch.cat([g1, g1, g2]); z = torch.cat([z1, z1, z2]); z[1, 7] = float("nan")
print("NaN isolation CF rows finite:", torch.isfinite(cf(g, z)).all(1).tolist())
X = torch.cat([torch.ones(3, 25, 1, dtype=g.dtype), g], 2)
try:
    r = torch.linalg.lstsq(X, z.unsqueeze(-1), driver="gelsd"); print("NaN isolation gelsd rows finite:", torch.isfinite(r.solution.squeeze(-1)).all(1).tolist())
except Exception as e:
    print("gelsd on NaN batch raised", type(e).__name__, e)
