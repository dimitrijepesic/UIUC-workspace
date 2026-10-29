"""Numerical sanity check of the 'any plane is sound' argument for g(x,y)=sigmoid'(x)*y.

Real-arithmetic (float64) experiment only; not a soundness proof.
"""
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(0)
sig = lambda x: 1.0 / (1.0 + np.exp(-x))
h = lambda x: sig(x) * (1.0 - sig(x))          # sigmoid'
g = lambda x, y: h(x) * y
logit = lambda s: np.log(s / (1.0 - s))


def D_formula(A, B, C, lx, ux, ly, uy):
    """Corners + stationary points on the two y-edges (exact cubic roots via np.roots)."""
    pts = [(lx, ly), (lx, uy), (ux, ly), (ux, uy)]
    for ye in (ly, uy):
        if ye == 0.0:
            continue
        c = A / ye
        # s(1-s)(1-2s) = c  <=>  2s^3 - 3s^2 + s - c = 0
        for r in np.roots([2.0, -3.0, 1.0, -c]):
            if abs(r.imag) < 1e-12 and 0.0 < r.real < 1.0:
                x = logit(r.real)
                if lx <= x <= ux:
                    pts.append((x, ye))
    return max(abs(g(x, y) - (A * x + B * y + C)) for x, y in pts)


def D_formula_upstream_emulated(A, B, C, lx, ux, ly, uy):
    """Emulates upstream Pasado root path: complex64 Cardano, x**0.33333, exact isreal filter."""
    pts = [(lx, ly), (lx, uy), (ux, ly), (ux, uy)]
    lim = 1.0 / (6.0 * np.sqrt(3.0))
    for ye in (ly, uy):
        if ye == 0.0:
            continue
        cval = A / ye
        if not (-lim <= cval <= lim):
            continue
        Q = np.complex64(-1.0 / 12.0)
        R = np.complex64(0.25 * cval)
        sq = np.sqrt(np.complex64(Q ** 3 + R ** 2))
        S = (R + sq) ** np.float32(0.33333)
        T = (R - sq) ** np.float32(0.33333)
        J = np.complex64(1j)
        k = np.float32(np.sqrt(3.0) / 2.0)
        roots = [S + T + 0.5,
                 -0.5 * (S + T) + 0.5 + (k * J) * (S - T),
                 -0.5 * (S + T) + 0.5 - (k * J) * (S - T)]
        for r in roots:
            if r.imag == 0 and 0.0 < r.real < 1.0:
                x = float(logit(np.float32(r.real)))
                if lx <= x <= ux:
                    pts.append((x, ye))
    return max(abs(g(x, y) - (A * x + B * y + C)) for x, y in pts)


def D_dense(A, B, C, lx, ux, ly, uy, n=4001):
    xs = np.linspace(lx, ux, n)
    best = 0.0
    for ye in np.linspace(ly, uy, 41):  # interior y lines too, to test the y-edge claim
        best = max(best, np.max(np.abs(g(xs, ye) - (A * xs + B * ye + C))))
    return best


def ls_plane(lx, ux, ly, uy, steps=5):
    xs = np.linspace(lx, ux, steps); ys = np.linspace(ly, uy, steps)
    X, Y = np.meshgrid(xs, ys, indexing="ij"); X = X.ravel(); Y = Y.ravel()
    M = np.stack([np.ones_like(X), X, Y], 1)
    c, *_ = np.linalg.lstsq(M, g(X, Y), rcond=None)
    return c[1], c[2], c[0]


def ls_closed_form(lx, ux, ly, uy, steps=5):
    """Tensor-grid LS for separable g = h(x)*y: normal equations decouple."""
    xs = np.linspace(lx, ux, steps); ys = np.linspace(ly, uy, steps)
    hx = h(xs); xm = xs.mean(); ym = ys.mean(); hm = hx.mean()
    A = ym * np.sum((xs - xm) * (hx - hm)) / np.sum((xs - xm) ** 2)
    B = hm
    C = hm * ym - A * xm - B * ym
    return A, B, C


def minimax_plane(lx, ux, ly, uy, n=81):
    xs = np.linspace(lx, ux, n); ys = np.linspace(ly, uy, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij"); X = X.ravel(); Y = Y.ravel(); G = g(X, Y)
    # vars: A, B, C, t ; minimise t s.t. |G - (A X + B Y + C)| <= t
    ones = np.ones_like(X)
    A_ub = np.vstack([np.stack([-X, -Y, -ones, -ones], 1), np.stack([X, Y, ones, -ones], 1)])
    b_ub = np.concatenate([-G, G])
    res = linprog([0, 0, 0, 1], A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 4, method="highs")
    return res.x[0], res.x[1], res.x[2]


def tangent_plane(lx, ux, ly, uy):
    xc = 0.5 * (lx + ux); yc = 0.5 * (ly + uy)
    s = sig(xc); A = s * (1 - s) * (1 - 2 * s) * yc; B = h(xc)
    return A, B, g(xc, yc) - A * xc - B * yc


worst_gap = 0.0
worst_up = 0.0
cf_err = 0.0
stats = {k: [] for k in ["ls", "ls_noise", "minimax", "tangent", "random"]}
for trial in range(300):
    cx = rng.uniform(-6, 6); rx = 10 ** rng.uniform(-3, 0.7)
    cy = rng.uniform(-3, 3); ry = 10 ** rng.uniform(-3, 0.5)
    lx, ux, ly, uy = cx - rx, cx + rx, cy - ry, cy + ry
    planes = {
        "ls": ls_plane(lx, ux, ly, uy),
        "minimax": minimax_plane(lx, ux, ly, uy),
        "tangent": tangent_plane(lx, ux, ly, uy),
    }
    A, B, C = planes["ls"]
    planes["ls_noise"] = (A + rng.normal(0, 1e-4), B, C)
    planes["random"] = (A + rng.normal(0, 0.05), B + rng.normal(0, 0.05), C + rng.normal(0, 0.05))
    Acf, Bcf, Ccf = ls_closed_form(lx, ux, ly, uy)
    cf_err = max(cf_err, abs(Acf - A), abs(Bcf - B), abs(Ccf - C))
    for name, (a, b, c) in planes.items():
        Df = D_formula(a, b, c, lx, ux, ly, uy)
        Dd = D_dense(a, b, c, lx, ux, ly, uy)
        worst_gap = max(worst_gap, (Dd - Df) / max(Dd, 1e-300))
        Du = D_formula_upstream_emulated(a, b, c, lx, ux, ly, uy)
        worst_up = max(worst_up, (Df - Du) / max(Df, 1e-300))
        W = abs(a) * rx + abs(b) * ry + Df  # radius proxy if x,y were independent
        stats[name].append((Df, W))

print(f"max relative (dense - formula) over all planes/boxes: {worst_gap:.3e}  (<=0 means formula >= dense max)")
print(f"max relative underestimate of D by emulated upstream complex64 root path: {worst_up:.3e}")
print(f"max |closed-form LS - lstsq| coefficient diff: {cf_err:.3e}")
base = np.array(stats["ls"])
for name, v in stats.items():
    v = np.array(v)
    rD = v[:, 0] / base[:, 0]; rW = v[:, 1] / base[:, 1]
    print(f"{name:9s} D/D_ls median {np.median(rD):.3f} [p10 {np.percentile(rD,10):.3f}, p90 {np.percentile(rD,90):.3f}]"
          f" | W/W_ls median {np.median(rW):.3f} [p10 {np.percentile(rW,10):.3f}, p90 {np.percentile(rW,90):.3f}]")
