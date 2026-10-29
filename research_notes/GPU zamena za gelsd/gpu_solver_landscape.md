# GPU solvers for batched min-norm, rank-revealing least squares on tiny matrices (25x3): what exists as of Oct 2026

Context for the writer: the workload is N ≈ 1,000–65,000 independent problems `min ||X b - z||`. Each X is 25x3 (`[1, x, y]`) with one right-hand side. The reference is LAPACK `dgelsd` via `torch.linalg.lstsq(X, z, driver='gelsd')` on CPU: SVD-based, minimum-norm, and σ_i ≤ rcond·σ_1 treated as zero, with default rcond = eps·max(m, n) = 25·eps (≈5.55e-15 in float64, ≈2.98e-6 in float32). Every finding below was checked on 2026-10-25 against source code, vendor docs or issue threads, unless it is marked otherwise.

Version landmarks (PyTorch releases): v2.12.0 2026-05-13, v2.12.1 2026-06-18, v2.13.0 2026-07-08, v2.14.0 2026-09-02, v2.14.1 2026-09-30 (latest) — [GitHub releases API / releases page](https://github.com/pytorch/pytorch/releases). Issues already cite nightlies `2.16.0.dev2026100x`, so a 2.15 release branch exists — [issue #200486](https://github.com/pytorch/pytorch/issues/200486).

## Q1. PyTorch: lstsq drivers on CUDA and MPS, the gelsd-on-CUDA requests, pinv and svd dispatch, known issues

### Takeaway
**CUDA:** every PyTorch release up to v2.14.1 and current main accepts only `driver='gels'` on CUDA. That driver is unpivoted QR and assumes full rank. The requests for an SVD (gelsd) driver on CUDA are open, and the one PR doing it (#176746) has gone stale. The usable GPU route inside PyTorch is `torch.linalg.svd` / `torch.linalg.pinv`. For a 25x3 matrix this dispatches to cuSOLVER `gesvdjBatched`, in float64, with the same default cutoff rule as gelsd (`rtol = eps·max(m,n)`). It does force a device-to-host sync on every call.

**MPS:** native MPS `lstsq` and `svd` first appeared in 2.14.0. They are float32 only, solve every driver through SVD, and had serious bugs in 2.14.0, partly fixed in 2.14.1. In torch 2.12 neither has an MPS kernel.

### Cited Findings
**lstsq driver rules (CUDA, CPU, MPS)**
- Current main, `get_default_lstsq_driver`: allowed drivers are `{"gels","gelsy","gelsd","gelss"}`. The code comment reads "CUDA supports only the 'gels' driver; CPU and MPS support all four". On CUDA, any other driver raises `"torch.linalg.lstsq: \`driver\` other than \`gels\` is not supported on CUDA"`. The default driver is `"gels"` on CUDA and `"gelsy"` otherwise — [BatchLinearAlgebra.cpp (main)](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/BatchLinearAlgebra.cpp)
- I diffed the release tags v2.12.0, v2.13.0, v2.14.0, v2.14.1 and main:
  - All of them contain the CUDA "not supported on CUDA" check.
  - The "CPU and MPS" branch appears only from v2.14.0 on.
  - `native_functions.yaml` dispatches `linalg_lstsq.out` and `_linalg_svd.U` to `CPU, CUDA` in v2.12.0, and to `CPU, CUDA, MPS` in v2.14.0.
  - Sources: [v2.12.0 native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml), [v2.14.0 native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/native_functions.yaml)
- Default rcond in code: `rcond.has_value() ? rcond : _get_epsilon(real dtype) * max(m, n)` — [BatchLinearAlgebra.cpp (main)](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/BatchLinearAlgebra.cpp)
- The docs say:
  - "For CUDA input, the only valid driver is 'gels', which assumes that A is full-rank."
  - σ_i "will be rounded down to zero if σ_i ≤ rcond·σ_1".
  - Default rcond is "the machine precision of the dtype of A times max(m, n)".
  - "The default value of rcond may change in a future PyTorch release."
  - The `driver` docstring still says "name of the LAPACK/MAGMA method to be used".
  - Sources: [torch.linalg docstrings (torch/linalg/__init__.py, main)](https://github.com/pytorch/pytorch/blob/main/torch/linalg/__init__.py); [torch.linalg.lstsq docs](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html)

**How CUDA `gels` actually runs**
- CUDA `lstsq_kernel` (identical in v2.12.0 and main):
  - If m < n: `linalg_lstsq_gels`, which is `geqrf` + `ormqr` + `triangular_solve`.
  - If m ≥ n: `if (m <= 256 && batchCount(b) >= max(2, m/16)) gels_batched_cublas(...) else linalg_lstsq_gels(...)`. The heuristic is justified by benchmarks in [PR #54725](https://github.com/pytorch/pytorch/pull/54725#issuecomment-832234456).
  - For 25x3 with batch ≥ 2, this means cuBLAS `gelsBatched`. rank, singular_values and rcond are ignored on CUDA (unused parameters).
  - Source: [cuda/linalg/BatchLinearAlgebra.cpp (main)](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp); [v2.12.0 copy](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp)
- The PyTorch wrapper `apply_gels_batched` calls `cublas<t>gelsBatched` in chunks of `batch_limit = 65535`. It passes the device `infos` array and requires m ≥ n — [BatchLinearAlgebraLibBlas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLibBlas.cpp)
- `linalg_lstsq` then calls `at::_linalg_check_errors(infos, "torch.linalg.lstsq", ...)` — [BatchLinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/BatchLinearAlgebra.cpp)
- In practice, a rank-deficient CUDA lstsq "silently fails in non-batched-mode but throws a `_LinAlgError` in batched mode" — [issue #117122](https://github.com/pytorch/pytorch/issues/117122)
- PR #176746's author found that the cuBLAS `infos` check is unreliable. With `A[...,-1] = A[...,-2] + A[...,-3]` (an exact linear dependence that is not a zero column), the call "gave me incorrect results without throwing any errors" — [PR #176746](https://github.com/pytorch/pytorch/pull/176746)

**Requests for gelsd on CUDA, and what maintainers said**
- [#72711](https://github.com/pytorch/pytorch/issues/72711) "Implement `gelsd` backend for `linalg.lstsq` via `linalg.svd`" (2022-02, OPEN). lezcano: "Absolutely… have a look at… `matrix_rank` which is also implemented in terms of `at::linalg_svd`."
- [#117122](https://github.com/pytorch/pytorch/issues/117122) "Improve behaviour of `torch.linalg.lstsq` on CUDA GPU for rank defficient matrices" (OPEN).
  - lezcano (2024-01-29): "Yep, this makes sense. I had this in mind when we were implementing `torch.linalg` but never got around implementing it."
  - Bumped 2026-02-27; a contributor volunteered 2026-03-04.
- [#72709](https://github.com/pytorch/pytorch/issues/72709) "Use iterative refinement algorithm from cuSOLVER for the least-squares solver on CUDA" (OPEN).
- [PR #126652](https://github.com/pytorch/pytorch/pull/126652) "Allow linalg.lstsq to use svd… for rank deficient matrices" was closed unmerged on 2024-08-21 (stale).
- [PR #176746](https://github.com/pytorch/pytorch/pull/176746) "Added 'gelsd' to CUDA torch.linalg.lstsq driver" (opened 2026-03-06, OPEN).
  - Reviewed by nikitaved on 2026-03-12.
  - Marked stale 2026-05-19, 2026-08-14 and 2026-10-16. The author asked on 2026-08-17 whether to let it close.
  - It needs explicit `driver='gelsd'`. It is not in any release.
- [#180157](https://github.com/pytorch/pytorch/issues/180157) "GPU gives wrong answer for rank-deficient lstsq" (2026-04) was closed by lezcano with the one-word reply "documented". The example is a rank-2 3x3 matrix where GPU vs NumPy L2 = 1.29.

**torch.linalg.pinv**
- Implementation (main):
  - `NoTF32Guard`.
  - If `hermitian=False`: `input.svd()`, then `tol = max(atol, rtol·σ_max)`, then `S_pinv = where(S > tol, 1/S, 0)`, then `V·diag(S_pinv)·Uᴴ`.
  - If `hermitian=True`: `linalg_eigh` with `|λ|`.
  - The code carries this FIXME: "Whenever we have a nice lstsq, we should dispatch this function to simply be `torch.lstsq(A, torch.eye(...))`… with a driver that supports singular inputs".
  - Source: [LinearAlgebra.cpp (main)](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LinearAlgebra.cpp)
- Default tolerances (`get_atol_rtol`):
  - `atol` defaults to 0.
  - `rtol` defaults to `eps(real dtype)·max(m, n)`. If `atol` is given and > 0, `rtol` becomes 0.
  - The tolerance tensors are float64, or float32 on MPS/Metal.
  - Sources: [LinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LinearAlgebra.cpp); docs "singular values … below max(atol, σ_1·rtol) threshold are treated as zero", "For CUDA inputs, this function synchronizes that device with the CPU" — [torch/linalg/__init__.py](https://github.com/pytorch/pytorch/blob/main/torch/linalg/__init__.py)

**torch.linalg.svd on CUDA (identical in v2.12.0 and main)**
- `svd_kernel` always calls `svd_cusolver`. MAGMA is no longer used for SVD. The `_warn_once_magma_deprecation("linalg.svd")` message reads "MAGMA, as a linear algebra backend, is deprecated and will be removed" — [cuda/linalg/BatchLinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp)
- `svd_cusolver` dispatch rule:
  - `driver_v = driver.value_or("gesvdj")`.
  - For `gesvdj`: `if (m <= 32 && n <= 32) svd_cusolver_gesvdjBatched(...) else svd_cusolver_gesvdj(...)`, where the latter loops one matrix at a time.
  - `"gesvd"` uses looped `cusolverDn<t>gesvd`, which requires m ≥ n (transposes otherwise).
  - `"gesvda"` uses `gesvdaStridedBatched`.
  - Source: [cuda/linalg/BatchLinearAlgebraLib.cpp (main)](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp)
- gesvdjBatched settings in PyTorch:
  - `cusolverDnXgesvdjSetTolerance(params, numeric_limits<scalar_t>::epsilon())`.
  - `cusolverDnXgesvdjSetMaxSweeps(params, cusolver_gesvdj_max_sweeps)` with `cusolver_gesvdj_max_sweeps = 400` (cuSOLVER's own default is 100).
  - `SetSortEig(params, 1)`.
  - For `full_matrices=False` with m ≠ n, it allocates a full square U (25x25 here) and copies back the first k columns.
  - Sources: [BatchLinearAlgebraLib.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp), [BatchLinearAlgebraLib.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.h)
- Convergence check:
  - For every driver except `gesvd`, `_check_gesvdj_convergence(info, k+1)` copies `infos` to the CPU. The source says "A device-host sync will be performed".
  - If some batches fail to converge, it warns once.
  - Only when the user did not pass `driver=`, it recomputes those batches with `gesvd`. With an explicit driver it only recommends "redo this SVD with another driver".
  - Source: [BatchLinearAlgebraLib.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp)
- The docs describe the drivers as follows: "By default (driver = None), we call 'gesvdj' and, if it fails, we fallback to 'gesvd'". Options: `None, gesvd, gesvdj, gesvda`, CUDA/cuSOLVER only. 'gesvd' for "not well-conditioned or precision is relevant", 'gesvda' for tall/wide — [torch/linalg/__init__.py](https://github.com/pytorch/pytorch/blob/main/torch/linalg/__init__.py)
- Benchmark that justified gesvdjBatched as the default for m, n ≤ 32 (lezcano, RTX 2060, CUDA 11.7.1): "the batched version is always faster". Example: shape (32,4) is 130 batched vs 290 iterative, unit unstated (likely µs). Only tiny batches were tested — [PR #88502 comment](https://github.com/pytorch/pytorch/pull/88502#issuecomment-1303860789)

**Known CUDA performance and accuracy issues**
- [#200486](https://github.com/pytorch/pytorch/issues/200486) (2026-10-10, open): there is a performance cliff at n = 32 → 33 because the default loops gesvdj per matrix. `svdvals` and `pinv` inherit it.
  - Tesla T4, B = 50, float64: n=32 default 10.5 ms (gesvdjBatched) vs `gesvda` 4.63 ms; n=33 default 484.2 ms.
  - float32: n=32 1.10 ms.
  - A proposed fix makes the default batch-aware and keeps gesvdjBatched for m, n ≤ 32. Our 25x3 stays on the fast batched path either way.
- [#75494](https://github.com/pytorch/pytorch/issues/75494) (open): float32 `pinverse` on CUDA gives non-optimal results for badly column-scaled matrices (`[A b; 0 1]` with ||b|| ≫ ||A||).
  - IvanYashchuk attributes it to the switch to "a faster, but less robust Jacobi-based algorithm" (PR #48436).
  - lezcano: "Whenever possible, it's better to use lstsq than pinv".
- GitHub searches for "svd NaN cuda", "pinv cuda incorrect" and "gesvdj accuracy" returned no relevant open CUDA issue specific to rank-deficient inputs (see Gaps).

**MPS (Apple)**
- 2.14.0 introduced native MPS SVD ([PR #185954](https://github.com/pytorch/pytorch/pull/185954)). Its kernel is a Metal one-sided Jacobi kernel, `svd_jacobi`.
  - It checks "linalg.svd: the MPS backend supports only float32 and complex64". There is no float64 ("Metal has no float64").
  - `driver=` is rejected.
  - Matrices too large for threadgroup memory fall back to the CPU with a warning.
  - In 2.14.0/2.14.1 there is also a gate `batch*m*n < 8192` → CPU.
  - Source: [mps/operations/LinearAlgebra.mm v2.14.0](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/operations/LinearAlgebra.mm)
- MPS `lstsq_kernel_mps` (identical in 2.14.0 and main) always solves through `at::linalg_svd(a, full_matrices=False)`, for every driver including `'gels'`.
  - Cutoff: `S > s_max * rcond_value`.
  - `rcond_value = rcond > 0 ? rcond : eps(real dtype)·max(m, n)`.
  - rank is set for non-gels drivers, and S for gelsd/gelss.
  - Source: [LinearAlgebra.mm (main)](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/LinearAlgebra.mm)
- 2.14.0 bugs, from the cherry-pick [PR #198394](https://github.com/pytorch/pytorch/pull/198394) (merged into release/2.14 for 2.14.1, 2026-09-24): "Native MPS SVD is new in 2.14.0 (#185954) and shipped with a crash and three silent-correctness bugs":
  - [#195937](https://github.com/pytorch/pytorch/issues/195937): `linalg.svd/svdvals/lstsq` raise a Metal pipeline-state error when the input has more than 8192 elements (repro: `(911,3,3)` fails). A 1024x25x3 batch has 76,800 elements, so it would hit this in 2.14.0.
  - [#196112](https://github.com/pytorch/pytorch/issues/196112): non-orthogonal U for rank-deficient and ill-conditioned inputs (max |UᴴU − I| = 1.0 on MPS vs ≤ 1.6e-6 on CPU).
  - [#196113](https://github.com/pytorch/pytorch/issues/196113): wrong complex batched underdetermined lstsq.
  - The `numel < 8192` CPU gate was removed in [PR #195950](https://github.com/pytorch/pytorch/pull/195950). Its M5 Pro benchmarks: (910,3,3) GPU 199 µs vs CPU-fallback 1233 µs, mean over 100 calls. linalg.svd "always runs … infos.any().item(), a mandatory device->host sync on every call".
- Still open in 2.14.1: [PR #200536](https://github.com/pytorch/pytorch/pull/200536) / [#200920](https://github.com/pytorch/pytorch/pull/200920) "Fix MPS SVD dropping valid singular vectors below an absolute epsilon, causing incorrect pseudoinverses and least-squares solutions". Repro: `pinv(1e-8·I)` and `lstsq` return all zeros on MPS. It was merged only into `release/2.15`, on 2026-10-14.
- [PR #200649](https://github.com/pytorch/pytorch/pull/200649): with `full_matrices=True`, the MPS SVD basis completion gave non-orthonormal U. It notes a regression "in 2.15" from small inputs moving to MPS. This does not affect `full_matrices=False`, which is what lstsq uses.

### Inferences
- **On CUDA in torch 2.12**, `torch.linalg.pinv(X) @ z`, or `U,S,Vh = torch.linalg.svd(X, full_matrices=False)` followed by a manual cutoff, is the only built-in route to gelsd's semantics. The cutoff is `σ_i ≤ eps·max(m,n)·σ_1 → 0`, the same as pinv's default.
  - It runs `cusolverDnDgesvdjBatched` in float64.
  - Costs to expect: one device-to-host sync per call, and a full 25x25 U allocated per problem (copied back to 25x3).
  - Results can differ from dgelsd in the last bits. For singular values that land within rounding error of the cutoff, the computed rank can also differ, because Jacobi and bidiagonal QR/D&C round differently.
- **Prefer `driver=None` over `driver='gesvdj'`.** Without an explicit driver, PyTorch silently falls back to `gesvd` for non-converged matrices. With an explicit driver, it only warns.
- **`torch.linalg.lstsq` on CUDA (gels → cuBLAS gelsBatched) cannot be used for the rank-deficient boxes**, and its error signal is unreliable. A constant `x` column gives a computed R diagonal of O(eps), not exactly 0, so `devInfoArray` probably does not fire.
- **On a Mac with torch 2.12**, MPS has no lstsq or svd kernel. Calls would fail unless the CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK=1`) is on; this is inferred from the dispatch tables and was not tested.
- **On torch ≥ 2.14.1, MPS lstsq is SVD-based with gelsd-like semantics, but float32 only.** It also still has the small-singular-value bug until 2.15. The float32 cutoff 25·eps32·σ_1 ≈ 3e-6·σ_1 makes many near-deficient boxes count as rank-deficient, while float64 gelsd would not.

### Gaps
- No PyTorch issue was found that documents gesvdjBatched NaN or accuracy failures on exactly rank-deficient float64 inputs, so its behaviour is undocumented rather than known-good.
- PyTorch publishes no A100 timing for `linalg.svd`/`pinv` on (N, 25, 3) float64.
- I did not find the PyTorch 2.15 release date or final contents. They are not released as of 2026-10-25.

## Q2. cuSOLVER / cuBLAS routines: which can give a rank-revealing, minimum-norm solution?

### Takeaway
The only batched NVIDIA library routines that reveal rank are the SVD (`gesvdjBatched`, `gesvdaStridedBatched`) and the symmetric eigensolvers (`syevjBatched`, `XsyevBatched`, usable through the Gram matrix). None of them solves least squares directly, so the min-norm solve has to be assembled by hand. `cublas<t>gelsBatched` and the IRS `gels` are unpivoted QR and assume full rank. Neither cuSOLVER nor cuBLAS has a batched pivoted QR (`geqp3`).

### Cited Findings
Docs consulted: cuSOLVER 13.4 and cuBLAS 13.4.

**cusolverDn<t>gesvdjBatched** — [cuSOLVER docs](https://docs.nvidia.com/cuda/cusolver/index.html)
- Precisions: S/D/C/Z; float64 is `cusolverDnDgesvdjBatched`.
- Size: "requires that all matrices are of the same size m,n no greater than 32 and are packed in contiguous way". The parameter descriptions say "m is no greater than 32", "n is no greater than 32".
- Sorting: "can either sort the singular values in descending order (default) or chose as-is" via `cusolverDnXgesvdjSetSortEig`.
- Diagnostics: it "cannot report residual and executed sweeps" (GetResidual/GetSweeps return NOT_SUPPORTED).
- info: `info[i] = min(m,n)+1` means no convergence on matrix i.
- **gesvdj accuracy controls** — [cuSOLVER docs](https://docs.nvidia.com/cuda/cusolver/index.html)
  - Stopping rule: "Jacobi method stops if ‖E‖_F ≤ eps·‖A‖_F".
  - Tolerance: "The default value is machine accuracy". Maximum sweeps: "The default value is 100… experiments show 15 sweeps are good enough to converge to machine accuracy".
  - "tolerance should not be smaller than machine accuracy".
  - "gesvdj has the same functionality as gesvd… The parallelism of Jacobi method gives GPU better performance on small and medium size matrices".

**cusolverDn<t>gesvdaStridedBatched** — [cuSOLVER docs](https://docs.nvidia.com/cuda/cusolver/index.html)
- It "approximates the SVD of a tall skinny m×n matrix" by working on AᵀA ("Since the computation of A**T*A … can greatly amplify errors, it is recommended to use gesvda only with well-conditioned data").
- It takes a `rank` parameter (the number of singular values computed).
- An optional `h_R_nrmF` residual.
- Size limit: `batchSize*stride ≤ INT32_MAX`.
- NVIDIA's GTC 2019 talk adds: "left singular vector is not accurate when singular value is small" and "accurate up to 1.e-6" — [GTC 2019 S9226 slides](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9226-fast-singular-value-decomposition-on-gpus-v2.pdf)

**Batched dense routines in cuSOLVER 13.4** — [cuSOLVER docs](https://docs.nvidia.com/cuda/cusolver/index.html)
- The complete set: `potrfBatched`, `potrsBatched`, `syevjBatched`/`heevjBatched`, `gesvdjBatched`, `gesvdaStridedBatched`, `cusolverDnXsyevBatched`.
- No batched `gels`, `geqrf` or `geqp3`. `cusolverDnXgeqrf` is non-batched, and `cusolverDnXgesvdp`/`Xgesvdr` have no batched forms.
- The only batched QR least-squares routine is the sparse `cusolverSpXcsrqrBatched`, documented as "no column pivoting… the solution is valid only if A_j is of full rank".
- `syevjBatched` limit: "n*lda*batchSize ≤ INT32_MAX". This page no longer shows a 32 limit for syevjBatched.
- `cusolverDnXsyevBatched` has a default algorithm that "May switch between algorithms for best performance".

**cusolverDn<t1><t2>gels (IRS, iterative refinement)** — [cuSOLVER docs](https://docs.nvidia.com/cuda/cusolver/index.html)
- It is "based on the QR factorization Xgels… similar… to the full precision LAPACK QR (least squares) solver (Xgels)".
- It is not in the batched list.
- PyTorch issue [#72709](https://github.com/pytorch/pytorch/issues/72709) requesting it is open.

**cublas<t>gelsBatched** — [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html)
- "find the least squares solution of a batch of overdetermined systems… On exit, each Aarray[i] is overwritten with their QR factorization".
- "supports only the non-transpose operation and only solves over-determined systems (m >= n)".
- "intended to be used for matrices of small sizes where the launch overhead is a significant factor".
- `devInfoArray`: "V > 0: the V-th diagonal element of the Aarray[i] is zero. Aarray[i] does not have full rank."

**cublas<t>geqrfBatched** — [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html)
- Householder QR, "supports arbitrary dimension", with no pivoting mentioned. Small-size intent, same as gelsBatched.

### Inferences
- **For exactly gelsd semantics with stock NVIDIA libraries**, run `DgesvdjBatched` on the 25x3 matrices (fits the ≤ 32 limit), then apply the cutoff and the min-norm back-substitution x = V·diag(σ⁺)·Uᵀz yourself. This is what torch `pinv`/`svd` already do.
- **The Gram route has a precision cost.** Use `syevjBatched` (3x3, very cheap) on XᵀX, then x = (XᵀX)⁺Xᵀz. In exact arithmetic it gives the same min-norm solution. Numerically it squares the condition number: with ‖XᵀX‖ ≈ σ_1², anything below ~sqrt(eps)·σ_1 ≈ 1.5e-8·σ_1 becomes noise. It therefore cannot reproduce gelsd's 25·eps cutoff for near-rank-deficient boxes, though exactly rank-deficient ones would still be zeroed with a suitable threshold.
- **`gesvdaStridedBatched` is unsuitable** for near-deficient inputs, for the same Gram-matrix reason (NVIDIA's own caveat).
- **The gesvdj stopping criterion is norm-wise** (‖E‖_F ≤ eps‖A‖_F). A computed "zero" singular value of an exactly deficient X will be O(eps·‖X‖), not 0. gelsd's factor of 25 in rcond gives some, but not guaranteed, margin to zero it consistently. The same is true of LAPACK itself.

### Gaps
- NVIDIA does not document a maximum `batchSize` for `gesvdjBatched` beyond the int type, nor its behaviour on exactly zero columns or repeated singular values.
- I found no NVIDIA numbers for `DgesvdjBatched` on A100 at m ≤ 32.

## Q3. MAGMA batched least squares, QR and SVD, and whether PyTorch exposes them

### Takeaway
MAGMA 2.10.0 (released 2026-02-23 per its ReleaseNotes; GitHub release published 2026-03-02) added a batched one-sided Jacobi SVD (`magma_<T>gesvj_batched`) and a QR+SVD variant (`magma_<T>gesvj_qr_batched`). Both work at any size and in all precisions, with a fused single-kernel path for tiny matrices. MAGMA has no `gels_batched`; QR building blocks exist instead. PyTorch exposes none of this: MAGMA is deprecated as a PyTorch backend, and SVD and lstsq always go to cuSOLVER/cuBLAS.

### Cited Findings
**MAGMA 2.10.0 release notes (dated Feb 23, 2026)** — [MAGMA ReleaseNotes](https://github.com/icl-utk-edu/magma/blob/master/ReleaseNotes)
- "magma_<T>gesvj_batched performs a Jacobi-based SVD on a uniform batch of relatively small matrices".
- "magma_<T>gesvj_qr_batched performs a QR factorization followed by a Jacobi-based SVD… May outperform magma_<T>gesvj_batched on very tall-skinny shapes".
- New `magma_<T>ormqr_batched`/`unmqr_batched`, left side only.
- Also: Blackwell support (CUDA ≥ 12.8), and builds with CUDA 13 / ROCm 7. The GitHub release v2.10.0 was published 2026-03-02.

**magma_zgesvj_batched source** — [src/zgesvj_batched.cpp](https://github.com/icl-utk-edu/magma/blob/master/src/zgesvj_batched.cpp)
- It "computes only the economy size SVD based on the one-sided Jacobi algorithm" and "computes V, not V**H". Singular values are "sorted so that S(i) >= S(i+1)".
- Internal defaults: `gesvj_max_sweeps = 100`; an inner Hermitian Jacobi eigensolver with `heevj_max_sweeps = 1`, a starting `heevj_tol = 10^floor(0.3·log10(1/eps))`, `heevj_tol_min = 30`, `heevj_tol_scal = 10`.
- For small matrices it dispatches to the fused shared-memory kernel `magma_zgesvj_batched_small_sm`. That kernel's rotation threshold is `tol * sqrt(ajj * akk)` and `info = sweep` if not converged — [magmablas/zgesvj_batched_small_sm.cu](https://github.com/icl-utk-edu/magma/blob/master/magmablas/zgesvj_batched_small_sm.cu)

**Batched files present in the MAGMA tree**
- `zgeqrf_batched`, `zgeqrf_batched_smallsq`, `zgeqr2_batched_fused_*`, `zunmqr_batched`, `ztrsm_batched`/`ztrsm_small_batched`, `zgesv_batched`, `zgesvj_batched`, `zgesvj_qr_batched`.
- There is no `zgels_batched` or batched `geqp3`. `zgels*.cpp` exist only in non-batched form.
- Source: [MAGMA repo tree](https://github.com/icl-utk-edu/magma)

**PyTorch and MAGMA** — [cuda/linalg/BatchLinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp)
- `svd_kernel` and `lstsq_kernel` both call `_warn_once_magma_deprecation(...)`. Even with `preferred_linalg_library('magma')` the message says "MAGMA, as a linear algebra backend, is deprecated and will be removed… will try dispatching to cuSOLVER instead".
- The same text is in v2.12.0 — [v2.12.0 source](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp)

**Abdelfattah & Fasi, "An Efficient Batch Solver for the Singular Value Decomposition on GPUs"** (arXiv 2601.17979, v1 2026-01-25, v2 2026-04-09) — [arXiv](https://arxiv.org/abs/2601.17979); [HTML v2](https://arxiv.org/html/2601.17979v2)
- It describes this MAGMA solver: "If the problem is small enough to fit in the shared memory… launch a single kernel for the entire operation".
- Accuracy thresholds use k·u with k = 30 (FP64 threshold 3.3307e-15).
- It describes cuSOLVER's batch SVD as "limited to small matrices of dimensions up to 32".

### Inferences
- **Using MAGMA `gesvj_batched` / `gesvj_qr_batched` from PyTorch** means building MAGMA ≥ 2.10 and writing a small C++/CUDA extension that passes tensor pointers (pointer-array interface). It gives economy SVD with V, and the user still applies the gelsd cutoff.
- **The `gesvj_qr_batched` design (QR first, then Jacobi SVD on the small R) is the natural structure for 25x3.** The SVD of the 3x3 R has the same singular values as X, so the gelsd cutoff semantics carry over.

### Gaps
- No public MAGMA timing for exactly 25x3 batches. The paper's very-small-matrix figure is graphical; only speedup ranges are in the text (Q6).
- Whether conda/pip MAGMA builds ship 2.10 is not verified.

## Q4. JAX, CuPy and TensorFlow

### Takeaway
- **JAX:** `jnp.linalg.lstsq` is SVD-based, NumPy-compatible and gelsd-like, and is batched only through `jax.vmap`. On CUDA, for m, n ≤ 32, the batched SVD lowers to cuSOLVER `gesvdjBatched`.
- **CuPy:** `cupy.linalg.lstsq` is SVD-based with the exact gelsd cutoff, but 2-D only. Its batched `cupy.linalg.pinv` uses `gesvdjBatched` for m, n ≤ 32, with a default rcond of 1e-15.
- **TensorFlow:** `tf.linalg.lstsq(fast=False)`, the min-norm, rank-deficient path, is CPU-only. `fast=True` is Cholesky on the normal equations.

### Cited Findings
**JAX** — [jax/_src/numpy/linalg.py](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py); [jax/_src/lax/linalg.py](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py); [jaxlib/gpu/solver_kernels_ffi.cc](https://github.com/jax-ml/jax/blob/main/jaxlib/gpu/solver_kernels_ffi.cc)
- **lstsq:** `_lstsq` requires `a.ndim == 2` ("Array must be two-dimensional").
  - If `rcond is None`, then `rcond = eps(dtype)·max(n, m)`; a negative rcond becomes eps.
  - It calls `u, s, vt = svd(a, full_matrices=False)` with mask `(s > 0) & (s >= rcond * s[0])`, and `x = vtᴴ (s_inv * uᴴ b)` with `Precision.HIGHEST`. It is wrapped in `jit`.
- **pinv:** works on `(..., M, N)`. Default `rtol = 10·max(rows, cols)·eps`, and it keeps `s > rtol·s[0]`. The docs note this differs from NumPy's default (1e-15).
- **GPU SVD algorithm choice:** with `SvdAlgorithm.DEFAULT` on CUDA/ROCm it uses Jacobi (`gesvdj`) if m, n ≤ 1024, else `gesvd`. Explicit `SvdAlgorithm.QR`, `JACOBI`, `POLAR` (`gesvdp`) are available. `DIVIDE_AND_CONQUER` is ROCm-only.
- **Reaching gesvdjBatched:**
  - For Jacobi the lowering sets `econ = not full_matrices and m > 32 and n > 32`, with the comment "gesvdjbatched: no 'econ' mode; batched path worthwhile up to 32x32".
  - The FFI kernel uses `GesvdjBatched` only when `batch > 1`, m, n ≤ 32 and not econ; otherwise it loops per-matrix `gesvdj`.
  - JAX calls no `SetTolerance`/`SetMaxSweeps`, so the cuSOLVER defaults apply (eps, 100 sweeps).
  - Matrices whose info ≠ 0 are replaced with NaN (`_replace_not_ok_with_nan`).

**CuPy** — [cupy/linalg/_solve.py](https://github.com/cupy/cupy/blob/main/cupy/linalg/_solve.py); [cupy/linalg/_decomposition.py](https://github.com/cupy/cupy/blob/main/cupy/linalg/_decomposition.py)
- **lstsq:** `_assert_2d(a)` and `b.ndim ≤ 2`, so no batching.
  - It uses `svd(a, full_matrices=False)`. `rcond=None` gives `eps·max(m, n)`; `rcond ≤ 0 or ≥ 1` gives eps (the comment reads "some doc of gelss/gelsd says 'rcond < 0', but it's not true!").
  - Values with `s <= rcond·s.max()` are set to 0.
- **pinv:** `pinv(a, rcond=1e-15)` takes `(..., M, N)`; values with `s <= rcond·amax(s)` are zeroed.
- **Batched SVD:** if `runtime.is_hip or (m <= 32 and n <= 32)` it uses `_gesvdj_batched`; otherwise it loops `gesvd`. The docs say: "On CUDA, when a.ndim > 2 and the matrix dimensions <= 32, a fast code path based on Jacobian method (gesvdj) is taken."

**TensorFlow** — [tensorflow/python/ops/linalg_ops.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_ops.py); [linalg_ops_common.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/linalg/linalg_ops_common.h); [matrix_solve_ls_op_double.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/linalg/matrix_solve_ls_op_double.cc)
- `fast=True` solves "the normal equations using Cholesky decomposition… only numerically stable when A is numerically full rank". It uses a composite implementation "that supports GPU".
- `fast=False` uses "the numerically robust complete orthogonal decomposition… minimum-norm least-squares solution, even when A is rank deficient. This path is typically 6-7 times slower". It ignores `l2_regularizer`.
- The fast=False kernel `MatrixSolveLs` is registered via `REGISTER_LINALG_OP`, which expands to `REGISTER_LINALG_OP_CPU` (`DEVICE_CPU`) only. It uses Eigen's `completeOrthogonalDecomposition()`.

**Other wrappers**
- `torch-batch-svd` (KinglittleQ) is a CUDA drop-in for batched SVD that "only works for matrices whose row and column are smaller than 32" (gesvdjBatched-based). This was seen only in a search summary — [ecosyste.ms listing](https://awesome.ecosyste.ms/projects/1368749)

### Inferences
- **JAX:** `jax.vmap(jnp.linalg.lstsq)` over (N, 25, 3) float64 is the closest drop-in for gelsd outside PyTorch.
  - It lowers to `DgesvdjBatched` with full 25x25 U.
  - The cutoff matches gelsd except at exact equality: JAX keeps `s == rcond·s0`, while gelsd zeroes it.
  - Float64 needs `jax_enable_x64`. This is standard JAX behaviour and was not re-verified here.
  - Unlike PyTorch, it does not sync per call. Non-convergence surfaces as NaN, not as a fallback.
- **CuPy:** `cupy.linalg.pinv(X, rcond=25*eps) @ z` (batched) reproduces gelsd's rule exactly (`≤` zeroed). The default rcond (1e-15) is about 5.5x smaller than gelsd's.
- **TensorFlow is not a viable GPU option** for rank-deficient min-norm solves.

### Gaps
- CuPy's `_gesvdj_batched` tolerance and sweep settings were not checked.
- No JAX or CuPy benchmark numbers for this shape were found.

## Q5. Custom approaches for tiny batched SVD/LS on GPU (per-thread Jacobi, Triton, normal equations, 3x3 analytic, cuSOLVERDx)

### Takeaway
Three building blocks are well established:
- **Vendor device-side library:** NVIDIA cuSOLVERDx, with a one-problem-per-thread `Thread()` mode recommended for M×N ≤ 100. It has GELS (unpivoted QR/LQ) and GESVD (QR algorithm) as device functions.
- **Per-thread 3x3 SVD codes** (McAdams et al. 2011), and 3x3 symmetric eigensolver comparisons (Kopp 2008).
- **MAGMA's fused single-kernel Jacobi.**

No reviewed Triton implementation of batched rank-revealing LS was found.

### Cited Findings
**cuSOLVERDx (part of MathDx)**
- Functions include GELS, GEQRF/UNMQR/UNGQR, GELQF, HEEV/HTEV/HEGV, BDSVD and GESVD — [cuSOLVERDx docs](https://docs.nvidia.com/cuda/cusolverdx/index.html)
- **GELS** uses "the QR factorization of A" for M ≥ N and gives the "minimum norm solution of an underdetermined system using the LQ factorization" for M < N. No pivoting or rank handling is described — [GELS](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/gels.html)
- **GESVD** "computes the singular values and, optionally, the left and/or right singular vectors of batched general matrices using the QR algorithm" — [GESVD](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/gesvd.html)
- **Release history** — [release notes](https://docs.nvidia.com/cuda/cusolverdx/release_notes.html)
  - 0.4.0 added singular vectors to BDSVD/GESVD, and "Thread operators… runs the solver in a thread context so that each thread executes the operation independently… intended for small problem sizes that fit in registers".
  - Known bug NVBUG 5908003: CTK 13.2+, SM100+, complex gesvd with vectors.
  - Current version: 0.5.0.
- **Thread operator** — [execution operators](https://docs.nvidia.com/cuda/cusolverdx/api/execution_ops.html)
  - "recommend using the Thread operator only for dense problems with M x N <= 100". 25x3 = 75.
  - The Block-mode example uses `Precision<double>()`.
- **Python access:** "accessed from Python using NVIDIA Warp or nvmath-python" — [Python bindings](https://docs.nvidia.com/cuda/cusolverdx/python_bindings.html)

**McAdams, Selle, Tamstorf, Teran, Sifakis (2011), UW-Madison TR1690, 3x3 SVD**
- "does not require any explicit branching beyond simple conditional assignments". The only operations are add, multiply and "an inexact (yet fast) reciprocal square root".
- It "robustly handles rank-deficient matrices and guarantees orthonormality of the computed rotational factors".
- Sources: [TR1690 PDF](https://research.cs.wisc.edu/techreports/2011/TR1690.pdf); [project page](https://pages.cs.wisc.edu/~sifakis/project_pages/svd.html)
- A CUDA port (Gao et al., SIGGRAPH Asia 2018) reportedly reaches 0.37 ns per 3x3 SVD on a Titan Xp. This comes from a search-result summary and was not verified in the repo — [3x3_SVD_CUDA](https://github.com/littlemine/3x3_SVD_CUDA)

**Kopp (2008), Int. J. Mod. Phys. C 19:523–548, "Efficient numerical diagonalization of hermitian 3x3 matrices"**
- Compares Jacobi, QL, Cuppen and analytic (Cardano). Jacobi is the most accurate and slowest. The analytic method is fastest but "can lose accuracy when matrix entries differ widely in magnitude"; a hybrid that falls back to QL avoids this. C/Fortran code is provided.
- Sources: [arXiv physics/0610206](https://arxiv.org/abs/physics/0610206); [code page](https://www.mpi-hd.mpg.de/personalhomes/globes/3x3)

**Triton**
- The only hit was an unreviewed Hugging Face file, `svd-triton`. Per its docstring: N=2 uses a closed-form 2x2 eigensolve in registers, N=3 a "fused kernel with cyclic Jacobi in registers", and N≥4 a Gram-eigh hybrid — [svd-triton/kernel_profiler.py](https://huggingface.co/AbstractPhil/svd-triton/blob/main/kernel_profiler.py)

**Earlier batched work**
- Abdelfattah, Haidar, Tomov, Dongarra (J. Comput. Sci. 2018), "Batched one-sided factorizations of tiny matrices using GPUs": LU/QR/Cholesky for sizes ≤ 32, not SVD — [ICL](https://icl.utk.edu/node/1209)
- KBLAS `kblas_gesvj_batch` (Boukaram et al., 2018) is a one-sided Jacobi batch SVD, single and double precision only, with an unblocked parallel-Jacobi kernel for very small matrices. Its double-precision tall-skinny path supports only rows ≤ 1024 — described in [arXiv 2601.17979 v2](https://arxiv.org/html/2601.17979v2)

### Inferences
- **Recommended custom kernel for 25x3, if built:** one thread (or a few lanes) per problem, in three steps.
  1. A Householder QR of X (25x3, 3 reflectors), applying Qᵀ to z on the fly.
  2. A one-sided or two-sided Jacobi SVD of the 3x3 R. Its singular values equal X's, so the gelsd cutoff `σ_i ≤ 25·eps·σ_1` applies unchanged.
  3. x = V·diag(σ⁺)·U_Rᵀ·(Qᵀz)[:3].
  - This mirrors MAGMA's `gesvj_qr_batched`. In float64 it is backward stable and avoids the condition-number squaring of the normal-equations or Gram routes.
  - The arithmetic is roughly a few hundred flops per problem, so 65k problems should be memory- or launch-bound rather than compute-bound on an A100. This is an estimate; nothing measured it.
- **It can also be written in pure batched PyTorch** (vectorized Householder over 3 columns plus a 3x3 Jacobi loop with a fixed sweep count). That avoids cuSOLVER's per-call host sync.
- **Normal equations + 3x3 eigh / Cholesky, or closed-form Cardano,** are fastest but square the condition number (see Q2 inferences). Kopp's results warn that analytic 3x3 eigensolvers lose accuracy with widely varying entries.
- **McAdams-style codes are float32-oriented**: they use an approximate rsqrt and a fixed iteration count. Using them in float64 would need validation.
- **Do not centre x and y to simplify the problem.** Centring, or any reparametrization, preserves least squares but **not** the minimum-norm choice among solutions in the original coefficients `[c0, c1, c2]`. A gelsd replica must work with the original X columns for rank-deficient boxes.

### Gaps
- I did not confirm which cuSOLVERDx functions (specifically `gesvd`/`gels`) support `Thread()` execution in float64, or their per-thread register limits.
- No measured timings for cuSOLVERDx on tiny batched SVD/LS were found.
- The 0.37 ns figure for the McAdams CUDA port is from a search summary only.

## Q6. Published benchmarks: batched SVD/QR of ~10^4–10^5 tiny matrices (≈25x3, 32x32) in float64

### Takeaway
No source reports A100 float64 timings for (10^4–10^5) x 25x3 SVD or LS. The best recent data is Abdelfattah & Fasi (2026) on a GH200 (H200 GPU), with batches of 10,000 very small matrices: in double precision, MAGMA's batch Jacobi SVD is 2.2–33.4x faster than cuSOLVER (`gesvdjBatched`) and 2.1–20.5x faster than the CPU. Smaller data points: PyTorch issue measurements on a T4, and NVIDIA's GTC 2019 gesvda numbers.

### Cited Findings
**Abdelfattah & Fasi, arXiv 2601.17979 v2 (2026), GH200/H200 system, CUDA 12.8.1** — [arXiv HTML](https://arxiv.org/html/2601.17979v2)
- Batches of 10,000 very small matrices:
  - Double precision: MAGMA is "2.1–20.5x faster than the CPU implementation, 2.2–33.4x faster than cuSOLVER, and 1.4–2.6x faster than KBLAS".
  - Single precision: 1.9–16.9x over cuSOLVER.
- On MI300A, rocSOLVER is slower than the CPU for very small matrices, and MAGMA is ">2 orders of magnitude" faster than it.
- WcycleSVD's `svd_small_matrix` "failed most of the accuracy checks".
- Tall-skinny case (n = 16, m 100–2000, 1,000 matrices): QR pre-processing helps. KBLAS double precision supports rows ≤ 1024 only.
- Its batch Hermitian eigensolver (1000 matrices, double) is up to 2.5x faster than cuSOLVER `dsyevj` on small sizes.

**W-cycle SVD (Xiao et al., SC22), one-sided Jacobi batched SVD**
- Reported 2.6–10.2x over cuSOLVER and 2.73–3.09x over MAGMA in a data-assimilation application. This is from the SC22 proceedings summary and was not verified in full text — [SC22 paper page](https://sc22.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap493.html)

**PyTorch #200486, Tesla T4 (weak in FP64), B = 50 square matrices, median ms** — [issue #200486](https://github.com/pytorch/pytorch/issues/200486)
- float64 n=32: `linalg.svd` 10.5 (gesvdjBatched), `gesvda` 4.63, `pinv` 10.7.
- float32 n=32: `svd` 1.10, `pinv` 2.39.
- Accuracy at n=33 (gesvdj, float64): max relative singular-value error 1.5e-14.

**PyTorch MPS, M5 Pro (float32)**
- `linalg.svd` on (910,3,3): 199 µs on GPU vs 1233 µs with CPU round-trip. On (64,8,8): 193 vs 817 µs — [PR #195950](https://github.com/pytorch/pytorch/pull/195950)

**NVIDIA GTC 2019 (V100, single precision)**
- Batched `SGESVDA` vs `SGESVDJ` with N = 35: e.g. M = 16,384, batch 32: 0.0076 s vs 0.0562 s (7.41x) — [S9226 slides](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9226-fast-singular-value-decomposition-on-gpus-v2.pdf)

### Inferences
- **For the 84–88%-of-runtime CPU gelsd** (~35 ms per 1024 problems, measured by the user), `DgesvdjBatched` on an A100 is very likely much faster per problem. That is not measured.
- **The main fixed costs in the PyTorch path** are the per-call host sync and the allocation and copy of a 25x25 U per problem.
- **A fused custom kernel** (QR + 3x3 Jacobi, Q5) or MAGMA's fused small-matrix path should beat cuSOLVER by the 2–30x range the paper reports. That range comes from H200 and other shapes and would need to be measured on A100 with 25x3.

### Gaps
- No public benchmark for float64 batched SVD or LS at m×n = 25x3 (or 32x32) with 10^4–10^5 matrices on an A100 was found. The repo's own Colab measurement is the only way to get this number.
- The paper's very-small-matrix timings are only in figures (absolute times not extracted).
- No benchmark was found for `cublasDgelsBatched` on tiny matrices.
