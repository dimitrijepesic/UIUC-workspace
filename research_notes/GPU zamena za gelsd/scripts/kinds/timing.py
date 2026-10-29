import time, torch
torch.set_default_dtype(torch.float64)
src = open("/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/kinds/kinds_check.py").read().split("torch.manual_seed(0)")[0]
ns = {}; exec(src, ns)
n = 65536
g0 = torch.Generator().manual_seed(1)
lx = torch.rand(n, generator=g0)*8-4; ux = lx + torch.rand(n, generator=g0)*4 + 1e-3
ly = torch.rand(n, generator=g0)*8-4; uy = ly + torch.rand(n, generator=g0)*4 + 1e-3
t = torch.linspace(0, 1, 5)
xs = lx[:, None] + (ux-lx)[:, None]*t; ys = ly[:, None] + (uy-ly)[:, None]*t
grid = torch.stack([xs[:, :, None].expand(n, 5, 5), ys[:, None, :].expand(n, 5, 5)], -1).reshape(n, 25, 2).contiguous()
s = torch.sigmoid(grid[..., 0]); z = (1-s)*(s*grid[..., 1])
X = torch.cat([torch.ones(n, 25, 1), grid], 2)
def tm(f, reps=3):
    f(); best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter()-t0)
    return best*1e3
print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, n={n}, float64, Mac CPU")
print(f"gelsd batched        {tm(lambda: torch.linalg.lstsq(X, z.unsqueeze(-1), driver='gelsd')):9.1f} ms")
print(f"closed form          {tm(lambda: ns['closed_form'](grid, z)):9.1f} ms")
print(f"pinv(X) @ z          {tm(lambda: torch.linalg.pinv(X) @ z.unsqueeze(-1)):9.1f} ms")
print(f"svdvals of 3x3 5K    {tm(lambda: ns['krank'](grid, 25*torch.finfo(torch.float64).eps)):9.1f} ms")
b1 = torch.linalg.lstsq(X, z.unsqueeze(-1), driver='gelsd').solution.squeeze(-1)
b2 = (torch.linalg.pinv(X) @ z.unsqueeze(-1)).squeeze(-1); b3 = ns['closed_form'](grid, z)
print("max rel |pinv-gelsd|", ((b2-b1).abs().max(1).values/b1.abs().max(1).values).max().item(), " max rel |CF-gelsd|", ((b3-b1).abs().max(1).values/b1.abs().max(1).values).max().item())
d = (b3-b1).abs().max(1).values/b1.abs().max(1).values
idx = d.argsort(descending=True)[:20]
src2 = open("/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/kinds/kinds_check.py").read().split("def krank")[0]
ns2 = {}; exec(src2, ns2)
eg = []; ec = []
for i in idx.tolist():
    ref = ns2["exact"](grid[i], z[i])
    eg.append(ns2["relerr"](b1[i].tolist(), ref)); ec.append(ns2["relerr"](b3[i].tolist(), ref))
print("worst-20 boxes: gelsd vs exact max", max(eg), " CF vs exact max", max(ec), " widths x:", (ux-lx)[idx].min().item())
