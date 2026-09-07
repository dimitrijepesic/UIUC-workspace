import torch

def batched_get_linspace(lx, ux, ly, uy, STEPS=5):
    """[n,STEPS*STEPS,2] equal to torch.stack(get_linspace(...))."""
    def lins(lo, hi):
        t = torch.arange(STEPS, dtype=lo.dtype, device=lo.device)
        step = (hi - lo) / (STEPS - 1)
        out = lo.unsqueeze(1) + t * step.unsqueeze(1)
        tail = torch.arange(STEPS, device=lo.device) >= STEPS // 2
        out[:, tail] = hi.unsqueeze(1) - (STEPS - 1 - t[tail]) * step.unsqueeze(1)
        return out
    xs, ys = lins(lx, ux), lins(ly, uy)
    return torch.stack((xs.repeat_interleave(STEPS, dim=1),
                        ys.repeat(1, STEPS)), dim=2)

def reference(lx, ux, ly, uy, STEPS=5):
    xs = [torch.linspace(lx[i].item(), ux[i].item(), steps=STEPS) for i in range(lx.shape[0])]
    ys = [torch.linspace(ly[i].item(), uy[i].item(), steps=STEPS) for i in range(ly.shape[0])]
    return torch.stack([torch.cartesian_prod(xs[i], ys[i]) for i in range(len(xs))])

def case(name, lx, ux, ly, uy, STEPS=5):
    r = reference(lx, ux, ly, uy, STEPS)
    b = batched_get_linspace(lx, ux, ly, uy, STEPS)
    ok = torch.equal(r, b)
    print(f"  {name:44s} STEPS={STEPES if False else STEPS}  shape={tuple(b.shape)}  bit-exact={ok}"
          + ("" if ok else f"  max|diff|={(r-b).abs().max().item():.3e}"))
    return ok

for dt in (torch.float64, torch.float32):
    torch.set_default_dtype(dt)
    print(f"=== dtype {dt} ===")
    g = torch.Generator().manual_seed(11)
    allok = True
    for STEPS in (3, 5, 7, 9):
        n = 500
        lo = torch.rand(n, generator=g) * 10 - 5
        hi = lo + torch.rand(n, generator=g) * 5 + 1e-9
        ly = torch.rand(n, generator=g) * 10 - 5
        uy = ly + torch.rand(n, generator=g) * 5 + 1e-9
        allok &= case("random boxes", lo, hi, ly, uy, STEPS)
    n = 200
    z = torch.rand(n, generator=g) * 4 - 2
    allok &= case("degenerate lx==ux and ly==uy (point box)", z, z.clone(), z, z.clone())
    allok &= case("degenerate lx==ux only", z, z.clone(), z, z + 1.0)
    w = torch.full((n,), 1e-12)
    allok &= case("width 1e-12", z, z + w, z, z + w)
    big = torch.full((n,), 1e6)
    allok &= case("large magnitude 1e6", big, big + 10, -big, -big + 10)
    allok &= case("negative ranges", -torch.abs(z) - 5, -torch.abs(z), -torch.abs(z) - 5, -torch.abs(z))
    print(f"  ALL BIT-EXACT: {allok}\n")
