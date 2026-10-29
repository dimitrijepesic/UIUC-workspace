import numpy as np, scipy, scipy.linalg as sl
from fractions import Fraction as F
print("numpy", np.__version__, "scipy", scipy.__version__)
try:
    print(np.show_config(mode="dicts")["Build Dependencies"]["lapack"]["name"])
except Exception as e: print("cfg?", e)

def design(lx, ux, ly, uy, dt=np.float64):
    x = np.linspace(lx, ux, 5).astype(dt); y = np.linspace(ly, uy, 5).astype(dt)
    X = np.array([[1.0, xi, yj] for xi in x for yj in y], dtype=dt)
    sp = lambda t: (1/(1+np.exp(-t)))*(1-1/(1+np.exp(-t)))
    z = np.array([sp(xi)*yj for xi in x for yj in y], dtype=dt)
    return X, z

def gelsd(X, z, dt=np.float64):
    rc = 25*np.finfo(dt).eps
    sol, res, rank, s = sl.lstsq(X, z, cond=rc, lapack_driver='gelsd')
    return sol, rank, s

def exact_fullrank(X, z):
    # exact rational normal equations (assumes full rank)
    Xf = [[F(float(v)) for v in r] for r in X]; zf = [F(float(v)) for v in z]
    M = [[sum(r[i]*r[j] for r in Xf) for j in range(3)] for i in range(3)]
    b = [sum(r[i]*zz for r, zz in zip(Xf, zf)) for i in range(3)]
    # gaussian elimination
    A = [M[i][:] + [b[i]] for i in range(3)]
    for c in range(3):
        p = next(r for r in range(c, 3) if A[r][c] != 0); A[c], A[p] = A[p], A[c]
        for r in range(3):
            if r != c and A[r][c] != 0:
                f = A[r][c]/A[c][c]; A[r] = [a - f*bb for a, bb in zip(A[r], A[c])]
    return np.array([float(A[i][3]/A[i][i]) for i in range(3)])

rng = np.random.default_rng(0)
eps = np.finfo(float).eps
# (a) exactly degenerate y: ratio s3/s1 in units of eps
worst = 0; ranks = {}
for t in range(4000):
    mag = 10**rng.uniform(-3, 6); xc = rng.uniform(-1, 1)*mag
    wx = 10**rng.uniform(-12, 1)*max(1, abs(xc)); y0 = rng.uniform(-1, 1)*10**rng.uniform(-3, 3)
    X, z = design(xc - wx/2, xc + wx/2, y0, y0)
    sol, rank, s = gelsd(X, z)
    worst = max(worst, s[2]/s[0]/eps); ranks[rank] = ranks.get(rank, 0) + 1
print("(a) y exactly degenerate: rank counts", ranks, " max computed s3/(s1*eps) =", worst, " cutoff = 25")
# both degenerate
worst = 0; ranks = {}
for t in range(2000):
    xc = rng.uniform(-1,1)*10**rng.uniform(-3,6); y0 = rng.uniform(-1,1)*10**rng.uniform(-3,3)
    X, z = design(xc, xc, y0, y0); sol, rank, s = gelsd(X, z)
    worst = max(worst, s[1]/s[0]/eps); ranks[rank] = ranks.get(rank, 0)+1
print("(a2) both degenerate: rank counts", ranks, " max s2/(s1*eps) =", worst)

def svK(xbar, ybar, sx, sy):
    K = 5*np.array([[1, xbar, ybar], [0, sx, 0], [0, 0, sy]])
    return np.linalg.svd(K, compute_uv=False)

# (b) transition in wy, and prediction
def transition(xc, wx, y0, var='y', dt=np.float64):
    rc = 25*np.finfo(dt).eps
    ws = np.logspace(-20, 2, 2000)
    last_def = None; first_full = None
    for w in ws:
        if var == 'y': X, z = design(xc-wx/2, xc+wx/2, y0-w/2, y0+w/2, dt)
        else: X, z = design(xc-w/2, xc+w/2, y0-wx/2, y0+wx/2, dt)
        sol, rank, s = gelsd(X, z, dt)
        if rank < 3: last_def = w
        elif first_full is None: first_full = w
    # prediction from exact 3x3 K (derived): rank drop iff s3 <= rc*s1, sigma = w/(2*sqrt2)
    def deficient(w):
        sx, sy = (wx, w) if var=='y' else (w, wx)
        s = svK(xc if var=='y' else xc, y0, sx/(2*np.sqrt(2)), sy/(2*np.sqrt(2))); return s[2] <= rc*s[0]
    pred = max([w for w in ws if deficient(w)], default=None)
    # crude formula
    if var == 'y':
        T = 1 + xc**2 + y0**2 + (wx**2)/8
        s1 = 5*np.sqrt(T)  # upper-ish approx
        crude = 2*np.sqrt(2)*rc*np.sqrt(1+y0**2)*np.sqrt(T)
    else:
        T = 1 + xc**2 + y0**2 + (wx**2)/8
        crude = 2*np.sqrt(2)*rc*np.sqrt(1+xc**2)*np.sqrt(T)
    return last_def, first_full, pred, crude

for (xc, wx, y0, var, dt) in [(0.3, 1.0, 0.5, 'y', np.float64), (2.0, 0.5, 3.0, 'y', np.float64),
                              (10.0, 1.0, 1.0, 'x', np.float64), (1e3, 1.0, 1.0, 'x', np.float64),
                              (1e4, 1.0, 0.2, 'x', np.float64),
                              (0.3, 1.0, 0.5, 'y', np.float32), (10.0, 1.0, 1.0, 'x', np.float32),
                              (100.0, 1.0, 1.0, 'x', np.float32)]:
    ld, ff, pr, cr = transition(xc, wx, y0, var, dt)
    print(f"(b) {np.dtype(dt).name} var={var} center={xc} other_w={wx} y0={y0}: gelsd last-deficient w={ld:.3e} first-full w={ff:.3e} | K-svd pred={pr:.3e} crude={cr:.3e}")

# (d) jump: box just inside deficient band vs exact full-rank
xc, wx, y0 = 2.0, 0.5, 3.0
for w in [1e-16, 5e-15, 2e-14, 1e-13, 1e-12]:
    X, z = design(xc-wx/2, xc+wx/2, y0-w/2, y0+w/2)
    sol, rank, s = gelsd(X, z)
    if w > 0 and np.ptp(X[:,2]) > 0:
        ex = exact_fullrank(X, z)
    print(f"(d) wy={w:.0e} rank={rank} gelsd={sol} exact_fullrank={ex} diff={np.abs(sol-ex).max():.3e}")
# predicted jump (derived): (sbar + a*xbar*y0^2)/(1+y0^2) * (y0, 0, -1)
x = np.linspace(xc-wx/2, xc+wx/2, 5); spv = (1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x)))
sbar = spv.mean(); a = np.sum((x-x.mean())*(spv-sbar))/np.sum((x-x.mean())**2)
print("(d) predicted jump vector", (sbar + a*x.mean()*y0**2)/(1+y0**2)*np.array([y0, 0, -1]))
