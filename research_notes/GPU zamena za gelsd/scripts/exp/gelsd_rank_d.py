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
# (d) jump: box just inside deficient band vs exact full-rank
xc, wx, y0 = 2.0, 0.5, 3.0
for w in [2e-15, 5e-15, 2e-14, 1e-13, 1e-12]:
    X, z = design(xc-wx/2, xc+wx/2, y0-w/2, y0+w/2)
    sol, rank, s = gelsd(X, z)
    if w > 0 and np.ptp(X[:,2]) > 0:
        ex = exact_fullrank(X, z)
    print(f"(d) wy={w:.0e} rank={rank} gelsd={sol} exact_fullrank={ex} diff={np.abs(sol-ex).max():.3e}")
# predicted jump (derived): (sbar + a*xbar*y0^2)/(1+y0^2) * (y0, 0, -1)
x = np.linspace(xc-wx/2, xc+wx/2, 5); spv = (1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x)))
sbar = spv.mean(); a = np.sum((x-x.mean())*(spv-sbar))/np.sum((x-x.mean())**2)
print("(d) predicted jump vector", (sbar + a*x.mean()*y0**2)/(1+y0**2)*np.array([y0, 0, -1]))

# float32 exact-degenerate ratio
worst=0; r={}
for t in range(2000):
    xc=rng.uniform(-1,1)*10**rng.uniform(-3,3); wx=10**rng.uniform(-2,1); y0=rng.uniform(-1,1)*10**rng.uniform(-3,2)
    X,z=design(xc-wx/2,xc+wx/2,y0,y0,np.float32); sol,rank,s_=gelsd(X,z,np.float32); worst=max(worst,s_[2]/s_[0]/np.finfo(np.float32).eps); r[int(rank)]=r.get(int(rank),0)+1
print("(a32) float32 y-degenerate ranks",r,"max s3/(s1*eps32)",worst)
# hazards of unified min-norm formula
with np.errstate(all='ignore'):
    for x0 in [1e8, 1e154, 1e155, 1e200, 1e-160]:
        k=1.0; den=1+x0*x0; C=k/den; A=k*x0/den
        # scaled variant (derived): for |x0|>1 use r=1/x0
        if abs(x0)>1: r_=1/x0; Cs=k*r_*r_/(1+r_*r_); As=k*r_/(1+r_*r_)
        else: Cs=k/(1+x0*x0); As=k*x0/(1+x0*x0)
        print(f"(h) x0={x0:.0e}: naive C={C:.3e} A={A:.3e} | scaled C={Cs:.3e} A={As:.3e}")
    x0=np.float32(1e19); print("(h32) x0=1e19 f32 x0^2 =", x0*x0, " 1+4096^2==4096^2:", np.float32(1)+np.float32(4096)**2==np.float32(4096)**2)
