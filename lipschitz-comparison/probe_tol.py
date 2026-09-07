import math, sys, torch
torch.set_default_dtype(torch.float64)
sys.path.insert(0, r"C:\everything\prakse\UIUC\Pasado\forward_mode_tensorized_src")
import precise_transformer as pt

def inverse_poly_real_tol(y, tol):
    """Viete form; NaN only when |arg| exceeds 1 by more than tol.

    Within the tolerance band the arg is clamped, which returns the (near-)double
    root instead of discarding it. Keeping a candidate can only make the error
    term larger, i.e. more conservative -- dropping one is what would risk
    unsoundness."""
    arg = 6.0 * math.sqrt(3.0) * y
    phi = torch.acos(arg.clamp(-1.0, 1.0)) / 3.0
    phi = torch.where(arg.abs() <= 1.0 + tol, phi, torch.full_like(phi, float("nan")))
    s = 1.0 / math.sqrt(3.0)
    return tuple(s * torch.cos(phi - k * 2.0 * math.pi / 3.0) + 0.5 for k in (0, 1, 2))

def as_sorted(triple):
    m = torch.stack(triple, dim=1)
    valid = ~torch.isnan(m)
    return torch.where(valid, m, torch.full_like(m, float("inf"))).sort(dim=1).values, valid.sum(dim=1)

LIM = 1.0 / (6.0 * math.sqrt(3.0))
g = torch.Generator().manual_seed(5)
samples = {
  "near boundary": torch.sign(torch.randn(20000, generator=g))*LIM*(1+(torch.rand(20000, generator=g)-0.5)*1e-6),
  "boundary +-1e-4": torch.sign(torch.randn(20000, generator=g))*LIM*(1+(torch.rand(20000, generator=g)-0.5)*1e-4),
  "out of range": torch.sign(torch.randn(20000, generator=g))*(LIM*1.001+torch.rand(20000, generator=g)*5),
  "wide realistic": torch.randn(20000, generator=g)*0.5,
}
print(f"{'tol':>8s}  " + "  ".join(f"{k:>18s}" for k in samples))
print(f"{'':>8s}  " + "  ".join(f"{'fewer/more':>18s}" for _ in samples))
for tol in (0.0, 1e-9, 1e-7, 1e-6, 1e-4):
    cells = []
    for name, ys in samples.items():
        cur, ccnt = as_sorted(pt.inverse_sigmoid_2nd_deriv(ys.clone()))
        new, ncnt = as_sorted(tuple(pt.inv_sigmoid(r) for r in inverse_poly_real_tol(ys.clone(), tol)))
        cells.append(f"{(ncnt<ccnt).sum().item():8d}/{(ncnt>ccnt).sum().item():<9d}")
    print(f"{tol:8.0e}  " + "  ".join(cells))
