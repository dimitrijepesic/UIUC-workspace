# Check existing box kinds (+ degenerate_y) against gelsd and an exact rational oracle.
import sys, os
from fractions import Fraction as F
import torch
P = "/Users/dimitrije/prakse/UIUC/Pasado"
sys.path.insert(0, os.path.join(P, "experiments"))
sys.path.insert(0, os.path.join(P, "forward_mode_tensorized_src"))
import test_batched_lin_reg as T
import precise_transformer as pt

def closed_form(grid, z):
    x = grid[..., 0]; y = grid[..., 1]
    xdeg = x[:, 0] == x[:, -1]
    ydeg = y[:, 0] == y[:, 4]
    xb = x.mean(1); yb = y.mean(1); zb = z.mean(1)
    rx = x - x[:, :1]; ry = y - y[:, :1]
    dx = rx - rx.mean(1, keepdim=True); dy = ry - ry.mean(1, keepdim=True)
    dz = z - zb[:, None]
    sxx = (dx*dx).sum(1); syy = (dy*dy).sum(1)
    one = torch.ones_like(sxx)
    A0 = torch.where(xdeg, torch.zeros_like(sxx), (dx*dz).sum(1) / torch.where(xdeg, one, sxx))
    B0 = torch.where(ydeg, torch.zeros_like(syy), (dy*dz).sum(1) / torch.where(ydeg, one, syy))
    k = zb - A0*xb - B0*yb
    denom = 1 + xdeg*xb*xb + ydeg*yb*yb
    return torch.stack((k/denom, torch.where(xdeg, k*xb/denom, A0), torch.where(ydeg, k*yb/denom, B0)), 1)

def exact(grid, z):
    xs = [F(v) for v in grid[:, 0].tolist()]; ys = [F(v) for v in grid[:, 1].tolist()]; zs = [F(v) for v in z.tolist()]
    n = 25
    xdeg = len(set(xs)) == 1; ydeg = len(set(ys)) == 1
    mx = sum(xs)/n; my = sum(ys)/n; mz = sum(zs)/n
    def slope(us, mu):
        su = sum((u-mu)**2 for u in us)
        return sum((u-mu)*(w-mz) for u, w in zip(us, zs)) / su
    a = F(0) if xdeg else slope(xs, mx)
    b = F(0) if ydeg else slope(ys, my)
    k = mz - a*mx - b*my
    den = 1 + (mx*mx if xdeg else 0) + (my*my if ydeg else 0)
    C = k/den
    A = k*mx/den if xdeg else a
    B = k*my/den if ydeg else b
    return [C, A, B]

def relerr(b, ref):
    r = max(abs(v) for v in ref)
    if r == 0: return float(max(abs(F(v)) for v in b))
    return float(max(abs(F(v) - rv) for v, rv in zip(b, ref)) / r)

def krank(grid, rcond):
    x = grid[..., 0]; y = grid[..., 1]
    rx = x - x[:, :1]; ry = y - y[:, :1]
    mdx = rx.mean(1); mdy = ry.mean(1)
    sx = ((rx - mdx[:, None])**2).mean(1).sqrt(); sy = ((ry - mdy[:, None])**2).mean(1).sqrt()
    xb = x[:, 0] + mdx; yb = y[:, 0] + mdy
    K = torch.zeros(x.shape[0], 3, 3, dtype=x.dtype)
    K[:, 0, 0] = 1; K[:, 0, 1] = xb; K[:, 0, 2] = yb; K[:, 1, 1] = sx; K[:, 2, 2] = sy
    s = torch.linalg.svdvals(5*K)
    return (s > rcond * s[:, :1]).sum(1), s[:, 2] / s[:, 0]

def make(kind, n, seed, dtype):
    if kind == "degenerate_y":
        g, z = T.realistic_grids(n, seed, dtype, "degenerate_x")
        g2 = g.flip(-1).reshape(n, 5, 5, 2).transpose(1, 2).reshape(n, 25, 2).contiguous()  # x<->y, keep x slowest
        z2 = torch.stack([pt.sigmoid_prime_times_y(gr) for gr in g2])
        return g2, z2
    return T.realistic_grids(n, seed, dtype, kind)

torch.manual_seed(0); torch.set_default_dtype(torch.float64)
n = 200
print(f"{'kind':16s} {'gelsd rank hist':22s} {'Krule==gelsd':>12s} {'max|CF-gelsd|rel':>17s} {'gelsd vs exact (med/max)':>26s} {'CF vs exact (med/max)':>24s} {'nonfinite CF':>12s}")
for kind in ["normal", "degenerate_x", "degenerate_y", "point", "near_degenerate", "saturated", "mixed_scale"]:
    g, z = make(kind, n, 12345, torch.float64)
    X = torch.cat([torch.ones(n, 25, 1, dtype=g.dtype), g], 2)
    res = torch.linalg.lstsq(X, z.unsqueeze(-1), driver="gelsd")
    sol = res.solution.squeeze(-1); rank = res.rank
    cf = closed_form(g, z)
    kr, ratio = krank(g, torch.finfo(torch.float64).eps * 25)
    hist = {int(r): int((rank == r).sum()) for r in rank.unique()}
    eg = []; ec = []
    for i in range(n):
        ref = exact(g[i], z[i])
        eg.append(relerr(sol[i].tolist(), ref)); ec.append(relerr(cf[i].tolist(), ref))
    eg.sort(); ec.sort()
    d = ((cf - sol).abs().max(1).values / sol.abs().max(1).values.clamp_min(1e-300)).max().item()
    print(f"{kind:16s} {str(hist):22s} {int((kr == rank).sum()):>5d}/{n:<6d} {d:17.3e} {eg[n//2]:12.2e}/{eg[-1]:<12.2e} {ec[n//2]:11.2e}/{ec[-1]:<11.2e} {int((~torch.isfinite(cf)).any(1).sum()):>12d}")
