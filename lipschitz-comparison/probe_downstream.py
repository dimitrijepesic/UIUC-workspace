import math, sys, torch
torch.set_default_dtype(torch.float64)
sys.path.insert(0, r"C:\everything\prakse\UIUC\Pasado\forward_mode_tensorized_src")
sys.path.insert(0, r"C:\everything\prakse\UIUC\lipschitz-comparison")
import precise_transformer as pt
from probe_cubic_real import inverse_poly_real

def inv_sig_2nd_real(y):
    return tuple(pt.inv_sigmoid(r) for r in inverse_poly_real(y))

LIM = 1.0 / (6.0 * math.sqrt(3.0))
g = torch.Generator().manual_seed(5)

def cmp(name, ys):
    cur = pt.inverse_sigmoid_2nd_deriv(ys.clone())
    new = inv_sig_2nd_real(ys.clone())
    print(f"\n=== {name} (n={ys.numel()}) ===")
    for k in range(3):
        c, n = cur[k], new[k]
        cn, nn = torch.isnan(c), torch.isnan(n)
        dropped = (~cn & nn).sum().item()
        extra   = (cn & ~nn).sum().item()
        both = ~cn & ~nn
        dm = (c[both]-n[both]).abs().max().item() if both.any() else float("nan")
        print(f"  root {k}: DROPPED(prod has, new NaN)={dropped:6d}  "
              f"extra(new has, prod NaN)={extra:6d}  both={int(both.sum()):6d}  max|diff|={dm:.3e}")

cmp("in range", (torch.rand(20000, generator=g)*2-1)*LIM*0.999)
cmp("out of range", torch.sign(torch.randn(20000, generator=g))*(LIM*1.001+torch.rand(20000, generator=g)*5))
cmp("near boundary", torch.sign(torch.randn(20000, generator=g))*LIM*(1+(torch.rand(20000, generator=g)-0.5)*1e-6))
cmp("wide realistic A/ly", torch.randn(20000, generator=g)*0.5)
