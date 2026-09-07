import torch
torch.set_default_dtype(torch.float64)
g = torch.Generator().manual_seed(7)
STEPS = 5
N = 4000
lo = torch.rand(N, generator=g) * 8 - 4
hi = lo + torch.rand(N, generator=g) * 4 + 1e-9

ref = torch.stack([torch.linspace(lo[i].item(), hi[i].item(), steps=STEPS) for i in range(N)])
t = torch.arange(STEPS, dtype=torch.float64)
step = (hi - lo) / (STEPS - 1)

cands = {}
# naive affine
cands["start + i*step"] = lo.unsqueeze(1) + t * step.unsqueeze(1)
# affine with exact endpoint pinned
a = lo.unsqueeze(1) + t * step.unsqueeze(1); a[:, -1] = hi
cands["start + i*step, end pinned"] = a
# symmetric: first half from start, second half from end (numpy-style)
b = lo.unsqueeze(1) + t * step.unsqueeze(1)
half = STEPS // 2
idx = torch.arange(STEPS)
tail = idx >= half
b[:, tail] = hi.unsqueeze(1) - (STEPS - 1 - t[tail]) * step.unsqueeze(1)
cands["symmetric halves"] = b
# weighted / lerp form
w = t / (STEPS - 1)
cands["lerp (1-w)*lo + w*hi"] = lo.unsqueeze(1) * (1 - w) + hi.unsqueeze(1) * w

for name, v in cands.items():
    exact = torch.equal(v, ref)
    bad = (v != ref).any(dim=1).sum().item()
    maxulp = (v - ref).abs().max().item()
    print(f"{name:34s} bit-exact={exact}  rows differing={bad}/{N}  max|diff|={maxulp:.3e}")
