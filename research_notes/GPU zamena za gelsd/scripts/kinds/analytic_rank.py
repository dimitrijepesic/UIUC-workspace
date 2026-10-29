# Sync-free analytic rank test from the invariants e1, e2, e3 of (5K)^T(5K), compared with gelsd's rank.
import sys, math, torch
dt = torch.float32 if sys.argv[1] == "f32" else torch.float64
torch.set_default_dtype(dt)
src = open("/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/kinds/kinds_check.py").read().split("torch.manual_seed(0)")[0]
ns = {}; exec(src, ns)
def analytic_rank(grid, rcond, iters=8):
    x = grid[..., 0]; y = grid[..., 1]
    rx = x - x[:, :1]; ry = y - y[:, :1]
    mdx = rx.mean(1); mdy = ry.mean(1)
    vx = ((rx - mdx[:, None])**2).mean(1); vy = ((ry - mdy[:, None])**2).mean(1)   # sigma^2
    xb = x[:, 0] + mdx; yb = y[:, 0] + mdy
    e1 = 25*(1 + xb*xb + yb*yb + vx + vy)
    e2 = 625*(vx*(1 + yb*yb) + vy*(1 + xb*xb) + vx*vy)
    e3 = 15625*vx*vy
    lam = e1.clone()                                   # Newton from the right: monotone to the largest root
    for _ in range(iters):
        f = ((lam - e1)*lam + e2)*lam - e3
        fp = (3*lam - 2*e1)*lam + e2
        lam = lam - f/fp
    q = e3/lam; p = (e2 - q)/lam
    disc = torch.clamp(p*p - 4*q, min=0)
    l2 = (p + disc.sqrt())/2
    l3 = torch.where(l2 > 0, q/torch.where(l2 > 0, l2, torch.ones_like(l2)), torch.zeros_like(l2))
    thr = (rcond**2)*lam
    return 1 + (l2 > thr).long() + (l3 > thr).long()
rcond = 25*torch.finfo(dt).eps
tot = 0; ok = 0
for kind in ["normal", "degenerate_x", "degenerate_y", "point", "near_degenerate", "saturated", "mixed_scale"]:
    n = 200; g, z = ns["make"](kind, n, 4242, dt)
    X = torch.cat([torch.ones(n, 25, 1), g], 2)
    rk = torch.linalg.lstsq(X, z.unsqueeze(-1), driver="gelsd").rank
    ar = analytic_rank(g, rcond); kr, _ = ns["krank"](g, rcond)
    tot += n; ok += int((ar == rk).sum())
    print(f"{dt} {kind:15s} analytic==gelsd {int((ar == rk).sum())}/{n}  svdvals==gelsd {int((kr == rk).sum())}/{n}")
# band sweep: x-width swept log-uniformly around the cutoff at several centres
for c, yb in [(0.3, 0.5), (2.0, 3.0), (10.0, 1.0), (1e3, 1.0)]:
    ws = torch.logspace(-17 if dt == torch.float64 else -9, -6 if dt == torch.float64 else 1, 2000)
    n = ws.numel()
    lx = torch.full((n,), c) - ws/2; ux = lx + ws
    t = torch.linspace(0, 1, 5)
    xs = lx[:, None] + (ux - lx)[:, None]*t; ys = (yb - 0.5) + t.expand(n, 5)
    g = torch.stack([xs[:, :, None].expand(n, 5, 5), ys[:, None, :].expand(n, 5, 5)], -1).reshape(n, 25, 2).contiguous()
    s = torch.sigmoid(g[..., 0]); z = (1 - s)*(s*g[..., 1])
    X = torch.cat([torch.ones(n, 25, 1), g], 2)
    rk = torch.linalg.lstsq(X, z.unsqueeze(-1), driver="gelsd").rank
    ar = analytic_rank(g, rcond); kr, _ = ns["krank"](g, rcond)
    bad = (ar != rk).nonzero().flatten()
    band = f"mismatch widths {ws[bad].min().item():.3g}..{ws[bad].max().item():.3g}" if bad.numel() else "no mismatch"
    lastdef = ws[(rk < 3).nonzero().flatten().max()].item() if (rk < 3).any() else float('nan')
    print(f"{dt} sweep c={c:g}: gelsd largest rank-deficient width {lastdef:.3g}; analytic mismatches {bad.numel()}/{n} ({band}); svdvals mismatches {int((kr != rk).sum())}")
