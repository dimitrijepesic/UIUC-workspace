# "gelsd-compat" closed form: K-rule decides rank, flagged axes use the exact-degenerate min-norm formulas.
import sys, torch
dt = torch.float32 if sys.argv[1] == "f32" else torch.float64
torch.set_default_dtype(dt)
src = open("/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/kinds/kinds_check.py").read().split("torch.manual_seed(0)")[0]
ns = {}; exec(src, ns)
def cf_compat(grid, z, rcond):
    x = grid[..., 0]; y = grid[..., 1]
    rx = x - x[:, :1]; ry = y - y[:, :1]
    mdx = rx.mean(1); mdy = ry.mean(1)
    dx = rx - mdx[:, None]; dy = ry - mdy[:, None]
    sxx = (dx*dx).sum(1); syy = (dy*dy).sum(1)
    xb = x[:, 0] + mdx; yb = y[:, 0] + mdy; zb = z.mean(1); dz = z - zb[:, None]
    rank, _ = ns["krank"](grid, rcond)
    sx = (sxx/25).sqrt(); sy = (syy/25).sqrt()   # per-point std of the 25-point columns
    ax = sx / (1 + xb*xb).sqrt(); ay = sy / (1 + yb*yb).sqrt()
    xdeg = (rank <= 1) | ((rank == 2) & (ax <= ay)) | (sxx == 0)
    ydeg = (rank <= 1) | ((rank == 2) & (ay < ax)) | (syy == 0)
    one = torch.ones_like(sxx); zero = torch.zeros_like(sxx)
    A0 = torch.where(xdeg, zero, (dx*dz).sum(1) / torch.where(xdeg, one, sxx))
    B0 = torch.where(ydeg, zero, (dy*dz).sum(1) / torch.where(ydeg, one, syy))
    k = zb - A0*xb - B0*yb
    den = 1 + xdeg*xb*xb + ydeg*yb*yb
    return torch.stack((k/den, torch.where(xdeg, k*xb/den, A0), torch.where(ydeg, k*yb/den, B0)), 1), rank
rcond = 25*torch.finfo(dt).eps
for kind in ["mixed_scale", "near_degenerate", "degenerate_y", "normal"]:
    n = 200
    g, z = ns["make"](kind, n, 12345, dt)
    X = torch.cat([torch.ones(n, 25, 1), g], 2)
    r = torch.linalg.lstsq(X, z.unsqueeze(-1), driver="gelsd")
    sol = r.solution.squeeze(-1)
    b, rank = cf_compat(g, z, rcond)
    trunc = r.rank < 3
    rel = (b - sol).abs().max(1).values / sol.abs().max(1).values
    plain = ns["closed_form"](g, z); relp = (plain - sol).abs().max(1).values / sol.abs().max(1).values
    print(f"{dt} {kind:15s} rank-match {int((rank == r.rank).sum())}/{n}  gelsd-truncated {int(trunc.sum())}  compat-vs-gelsd max rel (truncated boxes) {rel[trunc].max().item() if trunc.any() else float('nan'):.2e}  (all) {rel.max().item():.2e}   plain-CF-vs-gelsd (truncated) {relp[trunc].max().item() if trunc.any() else float('nan'):.2e}  max|coef| compat {b.abs().max().item():.3g}")
