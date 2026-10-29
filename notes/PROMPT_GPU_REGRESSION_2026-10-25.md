# Task brief: replace the CPU `gelsd` bottleneck in batched Pasado with a GPU solver

Private working note, outside the `Pasado` repo. Written 25.10.2026 as a self-contained
brief for a second opinion. Everything below was measured or checked; the status of
each claim is stated. Paths are relative to `/Users/dimitrije/prakse/UIUC/Pasado`.

---

## 1. What the project computes

Pasado bounds the local Lipschitz constant of a small sigmoid MLP (MNIST, networks
`3layer`, `4layer`, `5layer` with 100 neurons per layer, and `big` with 4 hidden layers
of 1024) along a "haze" path

    x(t) = (1 - t) * img + t,    t in [0, eps]

with dual zonotopes (a real zonotope for the activations and a dual zonotope for the
derivative with respect to t). The input interval [0, eps] can be split into `S` equal
pieces (`--n-splits`); the bound is the maximum over the pieces. More pieces give a
tighter bound (on `big`, image 0, eps_idx 0: 9927 at S=1, 250 at S=64).

At every sigmoid layer the "precise" transformer approximates the product
`sigmoid'(x) * y` over the box `[lx, ux] x [ly, uy]` of each neuron by a plane
`A*x + B*y + C` and then adds a fresh noise symbol equal to the worst-case approximation
error over the box. The plane comes from a least squares fit on a 5 x 5 grid.

## 2. What was done and what is now the bottleneck

The S pieces are independent and have identical shapes. They were previously run in a
Python loop (cost linear in S, GPU mostly idle). I stacked them on a leading batch axis
(`forward_mode_tensorized_src/batched_splits.py`; centers `[S, N]`, generators
`[S, G, N]`) and the (split, neuron) boxes are flattened to one batch of `S*N` for the
grid, the regression and the error check.

Verified on an A100 (float64): every per-split bound matches the sequential loop to
`<= 2.3e-14` relative (`experiments/test_batched_splits.py`, `ALL CHECKS PASSED`, plus 7
planted mutants caught on CPU).

Measured speedup of batched over sequential on the A100, eps_idx 0:

| network | S=8 | S=32 | S=64 |
|---|---|---|---|
| 3layer | 4.6x | 6.8x | 7.5x |
| 4layer | 4.4x | 6.4x | 7.0x |
| 5layer | 4.4x | 6.3x | 6.7x |
| big    | 1.6x | 1.6x | 1.6x |

Why `big` stalls: the regression uses `torch.linalg.lstsq(..., driver='gelsd')`, which
exists only on CPU, so inputs are moved to the CPU, solved, and moved back
(`lin_reg_tensor_batched`, `forward_mode_tensorized_src/precise_transformer.py:143`;
called from `batched_splits.py:128`). Measured share of the batched pass on `big`:

| S | total ms | CPU `gelsd` ms | share |
|---|---|---|---|
| 1  | 54   | 35   | 65% |
| 8  | 337  | 283  | 84% |
| 32 | 1252 | 1082 | 86% |
| 64 | 2488 | 2201 | 88% |

`gelsd` costs 34-35 ms per split regardless of S: the batched LAPACK call just loops over
the `S*N` tiny problems serially. The non-`gelsd` remainder at S=64 is 287 ms, so with a
free regression `big` would be about 4067 / 287 = 14x faster than sequential, not 1.6x.

Peak VRAM of the batched path grows linearly with S (big: 503 MB at S=1, 29.8 GB at
S=64); sequential stays at about 480 MB. `--split-chunk` caps it.

## 3. The exact problem to solve

For every box, the 25 grid points are the Cartesian product of 5 equispaced x values and
5 equispaced y values (built by `get_linspace_batched`, which reproduces
`torch.linspace` bit for bit; layout `[n, 25, 2]`, x slowest, y fastest). The targets are

    z = sigmoid'(x) * y = (1 - s(x)) * (s(x) * y)

and the design matrix is `X = [1, x, y]` (25 x 3). The current code returns the `gelsd`
solution `(C, A, B)`, intercept first, i.e. the minimum-norm least squares solution with
torch's default `rcond = eps * max(m, n)` (float64: about 5.5e-15, relative to the
largest singular value of X). After the solve, noise `N(0, 1e-4)` is added to A (kept as
is; it only guarantees `A != 0` for a later division).

Requirement: a solver for `[n, 25, 3]` batches with `n = S*N` (up to about 65 000) that
runs on the GPU, is batched, and reproduces the `gelsd` result.

### 3.1 Known facts about the available solvers

- `torch.linalg.lstsq` on CUDA supports only `driver='gels'` (QR, assumes full rank). It
  is wrong on rank-deficient boxes (observed earlier in this project), so it cannot
  replace `gelsd`.
- `gelsd` is SVD based with a relative cutoff; `torch.linalg.svd` / `pinv` run on CUDA
  for batches of small matrices. `pinv(X, rtol=eps*25) @ z` is the direct GPU analogue.
  Not bit-identical to `gelsd`; it can differ when a singular value is near the cutoff.
  Not tested yet.

### 3.2 Closed form (prototype, tested)

Because the grid is a Cartesian product, the centred columns `x - mean(x)`, `y - mean(y)`
and the constant column are exactly orthogonal (the cross term factorises as
`(sum_i dx_i) * (sum_j dy_j) = 0`). The normal equations are diagonal:

    A = sum(dx * z) / sum(dx^2)
    B = sum(dy * z) / sum(dy^2)
    C = mean(z) - A * mean(x) - B * mean(y)

Degenerate boxes need the minimum-norm solution, which is not "A = 0":

- x constant (lx == ux, value x0): columns `1` and `x` are collinear. With
  `k = mean(z) - B*mean(y)`, the min-norm solution is `C = k/(1+x0^2)`, `A = k*x0/(1+x0^2)`.
- y constant: symmetric (`C = k/(1+y0^2)`, `B = k*y0/(1+y0^2)`).
- both constant: `(C, A, B) = mean(z) * (1, x0, y0) / (1 + x0^2 + y0^2)`.

Unified, with `xdeg`, `ydeg` boolean masks and `A0`, `B0` the OLS slopes (0 where the
axis is degenerate):

    k = mean(z) - A0*mean(x) - B0*mean(y)
    denom = 1 + xdeg*mean(x)^2 + ydeg*mean(y)^2
    C = k / denom
    A = where(xdeg, k*mean(x)/denom, A0)
    B = where(ydeg, k*mean(y)/denom, B0)

Working prototype (kept here, not in the repo):

```python
def closed_form(grid, z):                      # grid [n,25,2], z [n,25]; returns [n,3] = (C, A, B)
    x = grid[..., 0]; y = grid[..., 1]
    xdeg = x[:, 0] == x[:, -1]                 # x runs slowest: first/last are lx/ux
    ydeg = y[:, 0] == y[:, 4]                  # y runs fastest: first 5 are ly..uy
    xb = x.mean(1); yb = y.mean(1); zb = z.mean(1)
    # centre around a data point, not the rounded mean (see risk R2)
    rx = x - x[:, :1]; ry = y - y[:, :1]
    dx = rx - rx.mean(1, keepdim=True); dy = ry - ry.mean(1, keepdim=True)
    dz = z - zb[:, None]
    sxx = (dx*dx).sum(1); syy = (dy*dy).sum(1)
    one = torch.ones_like(sxx)
    A0 = torch.where(xdeg, torch.zeros_like(sxx), (dx*dz).sum(1) / torch.where(xdeg, one, sxx))
    B0 = torch.where(ydeg, torch.zeros_like(syy), (dy*dz).sum(1) / torch.where(ydeg, one, syy))
    k = zb - A0*xb - B0*yb
    denom = 1 + xdeg*xb*xb + ydeg*yb*yb
    return torch.stack((k/denom,
                        torch.where(xdeg, k*xb/denom, A0),
                        torch.where(ydeg, k*yb/denom, B0)), 1)
```

## 4. Evidence gathered so far (CPU, float64, real network data)

All from instrumenting the real batched forward on `3layer` and `big`, images from the
correctly classified MNIST set, eps_idx in {0, 8, 15}.

1. **Degenerate boxes.** In layer 1, `ly == uy` exactly for 100% of the boxes (the dual
   part of the input has zero generators, so its bounds coincide). x is never exactly
   degenerate. Layers 2 and later have no exactly degenerate boxes. So the min-norm
   branch is not an edge case: it is every layer-1 box.
2. **Near-degenerate boxes grow as eps shrinks.** Width relative to `1 + |mean x| +
   |mean y|`: at eps_idx 0 essentially none below 1e-3; at eps_idx 8 about 1%-25% of x
   boxes below 1e-3; at eps_idx 15 all x boxes are below 1e-3 and 0.1%-0.6% below 1e-6,
   and up to 50% of y boxes in later layers are below 1e-3.
3. **Accuracy against an exact rational solution** (`fractions.Fraction` on the very
   same float x, y, z), 150 boxes per layer on `big`, layers 2-4:
   - eps_idx 0: `gelsd` median error 1e-16, max 2e-14; closed form median 4e-17, max 2e-15.
   - eps_idx 8: `gelsd` median 4e-15 to 8e-15, max 6e-13; closed form median 2e-16, max 9e-15.
   - eps_idx 15: `gelsd` median 2e-13 to 3e-13, max 7.6e-12; closed form median 7e-16,
     max 9e-15.
   So the (pivot-centred) closed form is more accurate than `gelsd` on the uncentred
   design as boxes get narrow.
4. **End to end.** Replacing the regression by the closed form (no fallback, no special
   handling of near-degenerate boxes) and comparing per-split bounds against the `gelsd`
   path, seeded, S=4, 3 images each: max relative difference `<= 1.2e-13` on `3layer` and
   `big`, at eps_idx 0, 8 and 15. The reported bounds agree to all printed digits.

Not yet tested: adversarial boxes (`test_batched_lin_reg.py` kinds `saturated`,
`mixed_scale`, `near_degenerate`, `point`), float32 (MPS), CUDA run, larger S, the other
two networks, timing of the closed form on the GPU.

## 5. Risks, in order of how much they worry me

**R1. Soundness vs tightness.** The plane `(A, B, C)` is arbitrary as far as soundness
goes, provided the error term is the true maximum of `|sigmoid'(x) y - (A x + B y + C)|`
over the box: the zonotope then encloses the function whatever plane was chosen. That
error comes from `check_corners_tensor` and `check_nonlinear_boundary_tensor`, which
evaluate the corners and the stationary points on the two y-edges (E is linear in y for
fixed x, so the extremum over y is on the y-edges or corners). So a less accurate solver
should only loosen the bound, not make it unsound. This reasoning should be re-checked
by whoever reviews the change, because it is what makes the solver substitution safe.
The existing autograd check (`lc_precise >= sampled max derivative`) is the empirical
guard.

**R2. Numerical stability of centring.** My first prototype centred around the
floating-point mean. It gave errors up to 8e-8 at small eps and, in one variant, a
factor 2-3.6 difference in the final bound. Two separate causes:
- the mean of 25 identical floats is not exactly that float, so `sum((y-ybar)^2) == 0`
  does not detect a degenerate axis (`dy ~ 1e-17`, then garbage slopes); degeneracy must
  be detected from the exact endpoints (`lx == ux`, `ly == uy`), as in the prototype;
- `x - mean(x)` carries an error of `eps*|mean(x)|` shared by all points, which is large
  compared with a narrow box; centring on `x[:,0]` and then removing the mean of the
  small differences fixes it.
Any production version must keep both properties.

**R3. Near-degenerate boxes and the `rcond` cutoff.** `gelsd` zeroes singular values
below `rcond * sigma_max`, and `sigma_max` grows with the coordinate magnitude, so a box
of width 1e-8 at |y| ~ 1e6 is treated as degenerate by `gelsd` while the closed form
divides by a tiny variance. Where the two disagree, `gelsd` switches to min-norm and the
closed form does not. By R1 this changes tightness only, but it can break "reproduces
gelsd" if that is a hard requirement. Options: (a) accept a tolerance, (b) mask the
boxes with `width / (1 + |mean|) < tau` (tau around 1e-6, scaled by dtype eps) and run
the old CPU `gelsd` on that subset only, (c) apply the same relative cutoff on the
closed-form side. (b) costs one host sync per layer (boolean mask indexing) and a graph
break under `torch.compile`; (a) and (c) stay fully elementwise. At eps_idx 15 the
subset in (b) is small (about 0.1%-0.6% of x boxes) but not empty.

**R4. float32.** The `rcond` cutoff is about 3e-6 in float32, so the near-degenerate
region is orders of magnitude wider (MPS runs use float32). Cancellation in `zb - A0*xb`
and in `dz` is also larger. The closed form has not been tested in float32.

**R5. `A` must stay nonzero.** The caller asserts `A != 0` after adding the noise, and
the nonlinear boundary check divides by `ly`/`uy` and by A-related quantities. The min-norm
formulas give `A = k*x0/(1+x0^2)`, which can be exactly 0 when `k == 0` or `x0 == 0`.
That is the same as `gelsd` (the noise fixes it), but confirm that the order "solve,
then add noise" is preserved and that the seeded noise stream is unchanged (seeded runs
must remain identical to the sequential protocol).

**R6. Division by zero in unused `where` branches.** `torch.where` evaluates both
branches; unguarded `/ sxx` yields NaN/inf in the discarded branch and breaks gradients
or `torch.compile` checks. The prototype guards the denominator with `where(deg, 1, sxx)`.

**R7. Non-finite inputs.** If lower > upper because of rounding in a bound, or if a box
holds inf/NaN (saturated sigmoids), `gelsd` and the closed form may fail differently.
`saturated` boxes (`lx` around 25 to 35) are in the existing test kinds and must be run.

**R8. Strict reproducibility requirement.** The current tests demand the batched pass
agree with the sequential loop to `1e-10` relative on per-split bounds
(`experiments/test_batched_splits.py`). The prototype is far inside that on real data, but
the adversarial cases in R3 and R7 are exactly where it could fail. Decide upfront whether
bit-level agreement with `gelsd` is required (then use fallback (b) or the SVD route) or
agreement of the final bound to a tolerance is enough.

**R9. SVD alternative has its own risks.** Batched `torch.linalg.svd`/`pinv` on CUDA for
`n ~ 65 000` matrices of size 25 x 3: unknown speed (cuSOLVER batched Jacobi for small
matrices, but PyTorch falls back to other paths in some versions), possible NaN on
degenerate input, and the cutoff rule `rtol` must be set to match `gelsd`. Measure before
choosing it as the primary path.

**R10. Scope of the change.** The sequential path and `precise_transformer.py` are the
reference for all tests and should stay untouched; the new solver should live next to
the batched code and be selectable (default off until validated), like the existing
`USE_*` switches.

## 6. Candidate designs

- **A. Closed form + exact degenerate branches, no fallback.** Fully elementwise,
  trivially fuses (relevant for the later TorchInductor/TVM/XLA comparison on `big`).
  Differs from `gelsd` only in near-degenerate boxes (R3).
- **B. Closed form + CPU `gelsd` on a flagged subset.** Keeps `gelsd` behaviour for the
  hard boxes. One sync and one graph break per layer.
- **C. GPU `pinv` via SVD with `rtol = eps * 25`.** Closest in spirit to `gelsd`; speed
  and robustness unmeasured.
- **D. Keep the CPU solve, parallelise it** (threads or processes over chunks of the
  `S*N` problems). Cheap to try, bounded by the CPU core count (Colab has few cores);
  does not remove the D2H/H2D transfers.

My current preference is A with B as the safety net if the adversarial tests fail, and C
as a comparison point.

## 7. Acceptance criteria I would use

1. A new test (against `gelsd`) on the existing box kinds (`normal`, `degenerate_x`,
   `point`, `near_degenerate`, `saturated`, `mixed_scale`) and on boxes captured from real
   runs, in float64 on CPU and CUDA, plus float32: report max coefficient error, and the
   error against an exact rational solution (as in section 4, item 3).
2. `experiments/test_batched_splits.py` still passes with the new solver, and the
   autograd soundness check (`lc_precise >= sampled max derivative`) still passes.
3. Planted mutants in the new solver are caught (wrong sign, wrong degenerate formula,
   mean instead of exact degeneracy test, missing pivot centring).
4. A CUDA sweep (notebook `COLAB_GPU_HYBRID_BATCHED_SPLITS.ipynb`) shows `big` going from
   1.6x to roughly 10x or more at S=32/64, with `lstsq` share well below 20%.

## 8. Constraints from the repository owner

- Never write AI attribution anywhere (commit messages, comments, docstrings, docs, logs).
- Do not run `git commit` from the agent environment; prepare changes and give the owner
  the commands. Commits use the owner's identity.
- Comments and docs inside `Pasado` are in English.
- Python environment: `Pasado/.venv/bin/python` (torch 2.12, CPU only on the Mac); CUDA
  runs happen on Colab through a notebook that writes the files with `%%writefile`.

## 9. Questions for the reviewer

1. Is the soundness argument in R1 airtight, in particular that the error term is exact
   for any plane (and that the `|E|` fix from 22.10 is what makes the boundary part
   correct)?
2. Is A (accept tolerance) good enough, or does the project's "reproduces the sequential
   result" bar require B/C?
3. Is there a cleaner way to apply the `rcond` rule on the closed-form side (R3) than a
   hard threshold, for example by computing the 3 x 3 Gram matrix singular values
   analytically?
4. Any numerical hazard in the unified min-norm formula that the real-data check could
   not reveal (for example `x0` large, so `1 + x0^2` loses the 1)?
