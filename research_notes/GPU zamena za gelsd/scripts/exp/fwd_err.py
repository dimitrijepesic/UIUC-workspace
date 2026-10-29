import numpy as np, scipy.linalg as sl, warnings
warnings.filterwarnings('ignore')
import importlib.util, sys
spec = importlib.util.spec_from_file_location('g', sys.argv[1]); 
src = open(sys.argv[1]).read(); src = src[:src.index('rng = np.random')]
g = {}; exec(src, g)
design, exact_fullrank = g['design'], g['exact_fullrank']
eps = np.finfo(float).eps
def closed(X):
    pass
def pivot_closed(lx, ux, ly, uy, z):
    x = np.linspace(lx, ux, 5); y = np.linspace(ly, uy, 5)
    Z = z.reshape(5, 5)  # rows i over x, cols j over y
    dx = x - x[0]; dx = dx - dx.mean(); dy = y - y[0]; dy = dy - dy.mean()
    zb = Z.mean()
    A = (dx[:, None]*(Z - zb)).sum()/(5*(dx*dx).sum())
    B = (dy[None, :]*(Z - zb)).sum()/(5*(dy*dy).sum())
    xm = x[0] + (x - x[0]).mean(); ym = y[0] + (y - y[0]).mean()
    return np.array([zb - A*xm - B*ym, A, B])
print("case: x in [1.75,2.25], y centre 3.0, width wy; rel err = max|theta-exact|/max|exact|")
for wy in [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    X, z = design(1.75, 2.25, 3.0-wy/2, 3.0+wy/2)
    ex = exact_fullrank(X, z); out = []
    for drv in ['gelsd', 'gelss', 'gelsy']:
        sol = sl.lstsq(X, z, cond=25*eps, lapack_driver=drv)[0]; out.append(np.abs(sol-ex).max()/np.abs(ex).max())
    pc = pivot_closed(1.75, 2.25, 3.0-wy/2, 3.0+wy/2, z); out.append(np.abs(pc-ex).max()/np.abs(ex).max())
    r = z - X@ex
    print(f"wy={wy:.0e}: gelsd {out[0]:.1e}  gelss {out[1]:.1e}  gelsy {out[2]:.1e}  pivot-closed {out[3]:.1e}   ||r||={np.linalg.norm(r):.1e}")
print("narrow-x case: x centre 10, width wx; y in [0.5,1.5]")
for wx in [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]:
    X, z = design(10-wx/2, 10+wx/2, 0.5, 1.5)
    ex = exact_fullrank(X, z)
    sol = sl.lstsq(X, z, cond=25*eps, lapack_driver='gelsd')[0]
    pc = pivot_closed(10-wx/2, 10+wx/2, 0.5, 1.5, z)
    print(f"wx={wx:.0e}: gelsd {np.abs(sol-ex).max()/np.abs(ex).max():.1e}  pivot-closed {np.abs(pc-ex).max()/np.abs(ex).max():.1e}")
