import math, sys, torch
torch.set_default_dtype(torch.float64)
sys.path.insert(0, r"C:\everything\prakse\UIUC\Pasado\forward_mode_tensorized_src")
sys.path.insert(0, r"C:\everything\prakse\UIUC\lipschitz-comparison")
import precise_transformer as pt
from probe_cubic_real import inverse_poly_real

def as_sorted(triple):
    """[n,3] with NaNs pushed to the end, valid roots ascending, + valid count."""
    m = torch.stack(triple, dim=1)                      # [n,3]
    valid = ~torch.isnan(m)
    cnt = valid.sum(dim=1)
    big = torch.full_like(m, float("inf"))
    filled = torch.where(valid, m, big)
    return filled.sort(dim=1).values, cnt

LIM = 1.0 / (6.0 * math.sqrt(3.0))
g = torch.Generator().manual_seed(5)

def cmp(name, ys):
    cur, ccnt = as_sorted(pt.inverse_sigmoid_2nd_deriv(ys.clone()))
    new, ncnt = as_sorted(tuple(pt.inv_sigmoid(r) for r in inverse_poly_real(ys.clone())))
    fewer = (ncnt < ccnt).sum().item()   # SOUNDNESS RISK: new drops a candidate
    more  = (ncnt > ccnt).sum().item()   # safe: new is more conservative
    same  = (ncnt == ccnt)
    if same.any():
        a, b = cur[same], new[same]
        fin = torch.isfinite(a) & torch.isfinite(b)
        d = (a[fin] - b[fin]).abs()
        dmax = d.max().item() if fin.any() else float("nan")
        dmed = d.median().item() if fin.any() else float("nan")
    else:
        dmax = dmed = float("nan")
    print(f"\n=== {name} (n={ys.numel()}) ===")
    print(f"  new has FEWER valid roots (soundness risk): {fewer}")
    print(f"  new has MORE valid roots (conservative)   : {more}")
    print(f"  same count: {int(same.sum())}   sorted-root diff: median {dmed:.3e}  max {dmax:.3e}")

cmp("in range", (torch.rand(20000, generator=g)*2-1)*LIM*0.999)
cmp("out of range", torch.sign(torch.randn(20000, generator=g))*(LIM*1.001+torch.rand(20000, generator=g)*5))
cmp("near boundary", torch.sign(torch.randn(20000, generator=g))*LIM*(1+(torch.rand(20000, generator=g)-0.5)*1e-6))
cmp("wide realistic A/ly", torch.randn(20000, generator=g)*0.5)
