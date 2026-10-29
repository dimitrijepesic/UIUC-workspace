# Testing methodology for a batched GPU replacement of LAPACK dgelsd (25x3 grid design X = [1, x, y])

Scope note: every library fact below was read directly from the upstream source file (downloaded 2026-10-25 from `main`/`master`) unless marked otherwise. Line numbers are approximate, as of that date. "Own run" means a script executed in this session with the project venv `Pasado/.venv` (torch 2.12.1, numpy 2.5.1, mpmath shipped with torch) on the M5 Pro Mac, CPU, float64. Those results are reproducible from the code summarised next to them.

---

## 1. LAPACK's own least-squares test suite (TESTING/LIN): matrix types, test ratios, threshold

### Takeaway
LAPACK tests `DGELSD` in `DDRVLS` with six matrix types: full rank or rank 3*min(m,n)/4, each either unscaled, scaled toward overflow or scaled toward underflow. It computes four scale-free ratios: the singular-value error, the residual, the normal-equation residual, and a check that x is in the row space. A result fails when a ratio is >= THRESH = 30. The generator deliberately keeps a wide gap between "nonzero" singular values (>= 0.1) and exact zeros, and uses RCOND ≈ sqrt(eps)/2. The suite therefore never tests the rank decision near the rcond boundary, which is the riskiest region for a closed-form replacement.

### Cited Findings
- There is no `dchkls.f` on Reference-LAPACK master (the fetch returns 404). The least-squares tester is `DDRVLS`: "DDRVLS tests the least squares driver routines DGELS, DGELST, DGETSLS, DGELSS, DGELSY, and DGELSD." — [ddrvls.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/ddrvls.f)
- Six matrix types (DOTYPE). j=1: A = U*D*V with random orthogonal U, V and D entries > 0.1 drawn from uniform(0,1), full rank. j=2: same, "scaled up". j=3: same, "scaled down". j=4: rank-deficient, with 3*min(M,N)/4 entries > 0.1 and the rest set to 0. j=5: type 4 scaled up. j=6: type 4 scaled down. — [ddrvls.f header](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/ddrvls.f)
- Generator `DQRT15` (rank-deficient / SVD drivers):
  - S(1) = 1. Each other singular value is |DLARND| resampled until it exceeds `SVMIN = 0.1D0`. The values are sorted in decreasing order, and the remaining (MN − RANK) are exactly 0.
  - The RHS is consistent: B = (first RANK columns of a random orthogonal matrix) × random, i.e. "Generate consistent rhs in the range space of A".
  - If RANK = 0, both A and B are null.
  - Scaling: SMLNUM = (safe_min/eps)/eps and BIGNUM = 1/SMLNUM. A, S and B are all rescaled by DLASCL so that max|A| = BIGNUM (scale type 2) or SMLNUM (scale type 3).
  - — [dqrt15.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/dqrt15.f)
- Generator `DQRT13` (full-rank, for DGELS/DGELST/DGETSLS) "generates a full-rank matrix that may be scaled to have large or small norm". It uses SMLNUM = safe_min/eps and BIGNUM = 1/SMLNUM, then rescales with DLASCL. — [dqrt13.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/dqrt13.f)
- The rank threshold used for all rank-revealing drivers in the test is `RCOND = SQRT( EPS ) - ( SQRT( EPS )-EPS ) / 2`, roughly sqrt(eps)/2 ≈ 7.5e-9 in double. — [ddrvls.f ~l.284](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/ddrvls.f)
- Tests for DGELSD (RESULT 15–18). Tests 11–14 are the same for DGELSS, and tests 7–10 for DGELSY:
  - **15**: relative SVD error `sum|S − S_true| / sum|S_true| / (EPS * min(M,N))`, computed only if RANK > 0.
  - **16**: `DQRT16`, RESID = `norm(B − A*X) / (max(m,n) * norm(A) * norm(X) * EPS)` (1-norms, max over RHS columns). Special cases: if ANORM = 0 and BNORM = 0 then 0; if ANORM <= 0 or XNORM <= 0 then 1/EPS.
  - **17**: `DQRT17`, `norm(R^T * A) / (norm(A) * alpha * max(M,N,NRHS) * EPS)` with R = B − A X. alpha = norm(B) if IRESID = 1 (the "zero-residual problem") and norm(R) if IRESID = 2. It uses the 1-norm, rescales R to unit max-norm first when NORMRS > SMLNUM, and is run only if M > CRANK.
  - **18**: `DQRT14`, "checks whether X is in the row space of A". It scales A and X to norms in [sqrt(eps), 1/sqrt(eps)], LQ-factorises [A', X]', and returns the norm of the trailing triangle / (MAX(M,N,NRHS) * eps). It is run only if N > CRANK, i.e. only for rank-deficient problems.
  - — [ddrvls.f l.~945–1000](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/ddrvls.f), [dqrt16.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/dqrt16.f), [dqrt17.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/dqrt17.f), [dqrt14.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/dqrt14.f)
- DGELSY test 7 uses `DQRT12`, which "computes the singular values of the upper trapezoid ... and returns the ratio || svlues − s ||/(||s||*eps*max(M,N))". — [dqrt12.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/dqrt12.f)
- Inside DDRVLS, DQRT17 is always called with IRESID = 1 (alpha = ||B||), because DQRT15 builds consistent right-hand sides. — [ddrvls.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/ddrvls.f)
- Default input `dtest.in`: M ∈ {0,1,2,3,5,10,50}, N ∈ {0,1,2,3,5,10,50}, NRHS ∈ {1,2,15}, NB ∈ {1,3,3,3,20} with NX ∈ {1,0,5,9,1}, "30.0 Threshold value of test ratio", and `DLS 6` (all six types). — [dtest.in](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/dtest.in)
- THRESH semantics: "A result is included in the output file if RESULT >= THRESH". — [ddrvls.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/ddrvls.f)
- The LAPACK Users' Guide error-bound fragment for least squares (xGELSS/xGELSD, full rank only) is:
  - SINT = RNORM/BNORM
  - COST = MAX(SQRT((1−SINT)(1+SINT)), EPSMCH)
  - TANT = SINT/COST
  - ERRBD = EPSMCH*(2/(RCOND*COST) + TANT/RCOND**2), with RCOND = S(N)/S(1)

  The fragment "only computes error bounds if the matrix has been determined to have full rank". For the rank-deficient case it says "computing and interpreting error bounds is more complicated". — [LAPACK Users' Guide, Error Bounds for Linear Least Squares](https://www.netlib.org/lapack/lug/node82.html)

### Inferences
- **The ratios carry over almost unchanged to the 25x3 problem** (m=25, n=3, nrhs=1, so max(m,n,nrhs)=25). Use THRESH = 30 as LAPACK does, with eps = 2^-52 for f64 and 2^-23 for f32 on MPS.
  - `R16 = ||z − Xb||_1 / (25 · ||X||_1 · ||b||_1 · eps)`. Only meaningful for **consistent** data (z = X·b_true). Project boxes with sigmoid data are not consistent, so build a separate "consistent" kind, e.g. z = A0 + B0·x + C0·y with dyadic coefficients.
  - `R17 = ||Xᵀ r||_1 / (||X||_1 · alpha · 25 · eps)`. Use alpha = ||r|| (the IRESID = 2 form) for the real, inconsistent sigmoid data. This is the normal-equation / orthogonality test and is independent of the condition number, which makes it the best primary pass/fail metric.
  - `R18` (row space): applies only to rank-2 or rank-1 results. For lx == ux = c the null space of X is spanned by v = (c, −1, 0), so require |v·b| / (||v||·||b||·25·eps) < 30. For point boxes the null space is span{(cx,−1,0), (cy,0,−1)}, so check both. This directly verifies "minimum norm" rather than merely "a least-squares solution".
  - `R15` (singular values): only if the pinv/SVD variant returns singular values. Compare them to the `singular_values` field that `torch.linalg.lstsq(..., driver='gelsd')` returns.
- **LAPACK's scaled types 2/3 do not carry over literally.** X contains a column of ones that cannot be rescaled. The analogue is to scale the box coordinates and/or z by powers of two toward the overflow and underflow limits (for f32/MPS: ~1e38 and ~1e-38, with subnormals below 1.18e-38). Then check that every ratio stays below 30 and that no inf/NaN appears.
- **LAPACK sizes 0 and 1 map to the batch dimension here.** Test n_problems ∈ {0, 1, 2, 3, 5, 10, 50, 65 000, 65 537}. The odd sizes and the size above 2^16 catch launch-grid and tail-handling bugs, and 0 corresponds to LAPACK's M = 0 / N = 0 cases.
- **Gap in LAPACK's own design that the project must fill.** Because DQRT15 forces σ ≥ 0.1 or σ = 0 and RCOND ≈ sqrt(eps)/2, LAPACK never tests singular values near `rcond·σ1`. `torch.linalg.lstsq` uses rcond = eps·max(m,n) = 25·eps ≈ 5.55e-15, and that is where the closed form and gelsd disagree (see §4).

### Gaps
- I did not read `dchkaa.f` (the main program that dispatches the "DLS" path) or the error-exit tester `derrls.f`. The error-exit tests (invalid LDA, etc.) are irrelevant to a closed-form kernel.
- The LAWN 41 installation/testing guide was not fetched. All details above come from the source files, which are authoritative for current LAPACK.

---

## 2. How PyTorch, NumPy, SciPy and JAX test lstsq / pinv (drivers, devices, rank-deficient inputs, tolerances)

### Takeaway
None of the four libraries runs a systematic rank-deficient or near-threshold suite for lstsq. They mostly compare against another LAPACK-backed implementation on random well-conditioned matrices, with fixed and fairly loose tolerances (PyTorch uses 1e-5 even for float64). Ideas worth borrowing:
- the Penrose conditions for pinv (PyTorch)
- NumPy's exactly-rank-3 matrix whose computed rank flips with rcond
- SciPy's mixed-rank batch with NaN residuals for rank-deficient slices
- JAX's zero-matrix regression test
- PyTorch's gelsy "jpvt is reset" test for state leaking between batch elements
- PyTorch's small-scale (1e-8) scaling test

### Cited Findings
**PyTorch** — [test/test_linalg.py](https://github.com/pytorch/pytorch/blob/main/test/test_linalg.py)
- `TestLinalg.test_linalg_lstsq` (~l.372):
  - dtypes float, double, cfloat, cdouble.
  - Drivers: CPU `('gels','gelsy','gelsd','gelss',None)`, CUDA only `('gels', None)`.
  - Sizes m = 2^i for i < 5, as (m, m//2) and (m, m), plus (m//2, m) on CPU or when cusolver is present.
  - Batches `(), (2,), (2,2), (2,2,2)`.
  - Matrices come from `random_well_conditioned_matrix`, with b = `torch.rand`.
  - rconds `(None, True, -1)`. `True` maps to rcond = 1.0 for gelss/gelsd, "set to zero roughly half of all the singular values".
  - The solution, rank, singular values and residuals are compared against `scipy.linalg.lstsq(lapack_driver=driver, cond=cond)` and, for gelsd, `np.linalg.lstsq(rcond=rcond)`, at **atol = rtol = 1e-5 for all dtypes**. `gels` is compared against `a.pinverse() @ b` at 1e-5.
- Residuals are expected only when m > n and every batch element has rank n; otherwise they must be empty (issue #56483). — same file, `check_correctness_ref`
- `test_linalg_lstsq_gelsy_jpvt_is_reset` (~l.522, CPU, double): batch = [eye(3), a rank-2 matrix] with expected rank `[3, 2]`. It is a regression test for pivot state leaking from one batch element to the next. — same file
- `test_linalg_lstsq_batch_broadcasting` (~l.537): broadcasting shapes including the empty batch `(0,)`, checked against `a.pinverse() @ b` at rtol = atol = 1e-5. — same file
- `test_pinv` (~l.3469):
  - Checks the Penrose conditions: A·A⁺·A = A, A⁺·A·A⁺ = A⁺, and A·A⁺ and A⁺·A Hermitian.
  - Compares against `np.linalg.pinv` with per-matrix rcond tensors and broadcast rcond.
  - `@precisionOverride({float32: 1e-3, float64: 1e-7})`.
  - Sizes include fat, thin and zero-numel matrices: (0,0), (0,2), (2,0), (3,0,0), …
  - — same file
- `TestLinalgSVD.test_svd_small_scale` (~l.9842, float32/complex64, CPU and CUDA): A = eye·1e-8, and requires `lstsq(Am, B).solution * scale ≈ pinv(A)` at atol = rtol = 1e-5. This is a scaling metamorphic test inside PyTorch itself. — same file
- OpInfo `sample_inputs_linalg_lstsq` (~l.733):
  - shapes `batch + (3+delta, 3)`, with batch ∈ {(), (3,), (3,3)} and delta ∈ {−1, 0, +1}. These are 3-column problems like ours.
  - CUDA tests only `gels`; CPU tests gels/gelsy/gelss/gelsd.
  - Empty systems: `(0,0,2), (3,3,0), (0,3,2), (3,0,2), (4,3,0), (3,4,0)`.
  - The `linalg.lstsq` OpInfo skips `test_out`, `test_variant_consistency_eager` and the jit variant for **device_type="mps", float32**.
  - — [opinfo/definitions/linalg.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/opinfo/definitions/linalg.py)
- `sample_inputs_linalg_pinv_singular` builds rank-k inputs as `Q_a @ Q_b^T`, with Q from QR of random matrices, k < min(3, m, n), and sizes {0, 3, 50}: "by making the columns of a and b orthonormal we make sure that the product matrix a @ b.t() has condition number 1 when restricted to its image". — same file (~l.516)
- `torch.linalg.lstsq` documentation:
  - "For CUDA input, the only valid driver is 'gels', which assumes that A is full-rank".
  - σ_i is "rounded down to zero if σ_i ≤ rcond·σ_1".
  - With rcond = None, rcond is "the machine precision of the dtype of A times max(m, n)".
  - "The default value of rcond may change in a future PyTorch release".
  - — [torch/linalg/__init__.py](https://github.com/pytorch/pytorch/blob/main/torch/linalg/__init__.py)
- `linalg_lstsq.out` dispatches to `CPU, CUDA, MPS: linalg_lstsq_out`, so an MPS kernel exists. — [native_functions.yaml ~l.14383](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
- Default `torch.testing.assert_close` tolerances (rtol, atol): float16 (1e-3, 1e-5), bfloat16 (0.016, 1e-5), **float32 (1.3e-6, 1e-5)**, **float64 (1e-7, 1e-7)**. The legacy `assert_allclose` uses float32 (1e-4, 1e-5) and float64 (1e-5, 1e-8). — [torch/testing/_comparison.py ~l.96, ~l.1720](https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py)

**NumPy** — [numpy/linalg/tests/test_linalg.py](https://github.com/numpy/numpy/blob/main/numpy/linalg/tests/test_linalg.py)
- `LstsqCases.do` (~l.1065), using `lstsq(a, b, rcond=-1)`:
  - if m == 0, x must be all zeros
  - if m <= n, b ≈ a·x and rank == m; otherwise rank == n
  - sv ≈ svd(a).s
  - residuals == Σ|a·x − b|² only if rank == n and m > n, otherwise empty
- The local `assert_almost_equal` uses **6 decimals for single and 12 for double**. `PinvCases` checks `a @ pinv(a) @ a ≈ a` with 5/11 decimals. — same file
- `TestLstsq.test_rcond`: a 6x4 matrix with columns c1 = [0,1,0,1,2,0], c2 = [0,2,0,0,1,0], c3 = [1,0,1,0,0,4], c4 = [0,0,0,2,3,0]. The test asserts rank 4 with `rcond=-1` and rank 3 with the default rcond or `rcond=None`. — same file. **Own run:** c4 = 2·c1 − c2 exactly, so the matrix has exact rank 3; numpy 2.5.1 gives σ_min/σ_max ≈ 0.49·eps and ranks (4, 3). This is the only test found in any of the libraries that deliberately places a singular value between two rcond thresholds.
- `test_nonfinite`: inf, −inf or nan in `a` → `LinAlgError` for single, double, csingle, cdouble. `test_empty_a_b`: (4,2,2), (0,4,1), (0,4,2), (4,0,1), (4,0,2), (4,2,0), (0,0,0). `test_incompatible_dims`. — same file
- Case lists include `0x0, 1x1, 8x8, 8x11, 1x5, 5x1, 0x4, 4x0`, generalised (stacked) cases (`_make_generalized_cases`) and non-contiguous strided cases (`_make_strided_cases`). — same file
- ULP utilities: `assert_array_max_ulp(a, b, maxulp=1, dtype=None)` ("differ in at most N Units in the Last Place"), `assert_array_almost_equal_nulp` and `nulp_diff`. — [numpy/testing/_private/utils.py ~l.1798–1982](https://github.com/numpy/numpy/blob/main/numpy/testing/_private/utils.py)

**SciPy** — [scipy/linalg/tests/test_basic.py `TestLstsq` ~l.1978](https://github.com/scipy/scipy/blob/main/scipy/linalg/tests/test_basic.py)
- Drivers `('gelsd','gelss','gelsy',None)`.
- `test_random_exact`: square matrices with a boosted diagonal `a[i,i] = 20*(0.1 + a[i,i])`. It asserts the effective rank equals n and checks `dot(a,x) ≈ b` with rtol = atol = **500·eps (float32)** or **1000·eps (float64)**.
- `test_random_overdet`: sizes (20,15) and (200,2), compared against a normal-equations solve (`direct_lstsq`) at rtol = atol = **25·eps**.
- Also: `test_check_finite`, `test_empty`, `test_empty_dtype`, and m<n / m>n tests against hard-coded arrays at atol = 1e-15. Residual tests use atol = 1e-14.
- `test_rank_deficient_residuals` (gelsd/gelss/gelsy, shapes (3,2) and (4,3,2)) builds a batch with **one all-zero slice** and other slices whose **first column is all ones** plus random columns. It requires the residual Σ|a·x − b|² for full-rank slices and **NaN for rank-deficient slices**. — same file, ~l.2391

**JAX** — [tests/linalg_test.py](https://github.com/jax-ml/jax/blob/main/tests/linalg_test.py)
- `testLstsq` (~l.1262):
  - shape pairs ((1,1),(1,1)), ((4,6),(4,)), ((6,6),(6,1)), ((8,6),(8,4)), ((0,3),(0,)), ((3,0),(3,)), ((3,1),(3,0))
  - rcond ∈ {−1, None, 0.5}
  - tol float32 1e-4, float64 **1e-12**, complex64 1e-5, complex128 1e-12, against `np.linalg.lstsq`, plus a compile (jit) check
  - Gradient checks are disabled because "grad is flaky for low-rank inputs"
- `testLstsqZeroMatrix` uses shapes (2,1), (2,2), (1,2): an all-zero A against b = ones. It is a "Regression test for https://github.com/jax-ml/jax/issues/32666". — same file
- `testPinv` covers shapes including (2000,7), (7,1000), (400000,2), (2,400000), (2,0,0), (3,0,2), (1,0) at tol = 1e-4 against numpy. The gradient check uses rtol = 6e-2 (a TODO comment calls it "very loose"). — same file (~l.1117)

### Inferences
- **Adopt from the libraries** (mapped onto the 25x3 problem):
  1. Penrose conditions, if the pinv/SVD variant materialises X⁺ (3x25).
  2. A mixed-rank batch: full, degenerate_x, degenerate_y, point and all-zero-z boxes interleaved in one launch, checking each element independently (SciPy/PyTorch style). This also covers PyTorch's "jpvt is reset" class of bug, where state from element i leaks into i+1.
  3. A zero-data test: z ≡ 0 must give b = 0 exactly, with no NaN (JAX regression class). The analogue of an all-zero X is impossible here because of the column of ones; the closest case is the point box at the origin (lx = ux = ly = uy = 0), whose exact min-norm answer is b = (mean z, 0, 0).
  4. An rcond-straddling case modelled on NumPy's `test_rcond` (see §4: the boundary for the grid design is easy to hit).
  5. The 1e-8 scaling test from PyTorch's `test_svd_small_scale`.
- **The libraries' tolerances are too loose to serve as acceptance criteria for f64.** PyTorch uses 1e-5 for every dtype, so a planted bug changing results by 1e-6 would pass. Use JAX-like f64 1e-12 only for well-conditioned boxes, and the κ-scaled or ratio-based metrics in §1 and §4 elsewhere.
- **The CUDA reference is not usable as an oracle.** CUDA `lstsq` only supports `gels`, which assumes full rank, so it cannot be a min-norm reference for degenerate boxes. The reference must be CPU gelsd (float64) or the exact oracle.
- **Use the MPS float32 lstsq only as a cross-check, not as an oracle.** It exists, but its OpInfo consistency tests are skipped for MPS float32.

### Gaps
- I did not check the PyTorch MPS implementation of `linalg_lstsq_out` (which algorithm it uses, or whether it falls back to CPU). Check `aten/src/ATen/native/mps/operations/LinearAlgebra.mm` before using MPS lstsq as a cross-check.
- NumPy's `random`-based generalised cases were not inspected line by line.

---

## 3. Metamorphic and property-based tests for least squares (invariances, Hypothesis, special values)

### Takeaway
For a 25x3 grid design, metamorphic relations (MRs) are exact in real arithmetic and many of them hold bit-for-bit in floating point (scaling by 2^k, negation, transposing the grid). They kill sign, swap, layout and degenerate-branch bugs without any oracle. The important caveat is that SVD/rcond rank decisions are **not** invariant under column scaling or shifting. MRs that move the box (shift or scale x) must therefore exclude problems near the rcond boundary, or they will "fail" because of the spec rather than the code.

### Cited Findings
- Metamorphic relations are proposed as "automated partial test oracles" for "non-testable" scientific programs, where correct outputs are unknown (Kanewala & Bieman). — [Kanewala, "Testing scientific software: techniques for automatic detection of metamorphic relations"](https://mountainscholar.org/handle/10217/166910)
- Recent work (ICSE 2025 NIER) starts from domain-agnostic MRs and refines them with a model checker and domain expertise; it was applied to a high-performance chemistry library. — [Model Assisted Refinement of Metamorphic Relations for Scientific Software](https://conf.researchr.org/details/icse-2025/icse-2025-nier/22/Model-Assisted-Refinement-of-Metamorphic-Relations-for-Scientific-Software)
- PyTorch's own test suite contains a scaling MR: `lstsq(A·1e-8, I).solution · 1e-8 ≈ pinv(A)` (float32, 1e-5). — [test_linalg.py `test_svd_small_scale`](https://github.com/pytorch/pytorch/blob/main/test/test_linalg.py)
- PyTorch does not guarantee bitwise batch invariance: "`(A@B)[0]` ... is not guaranteed to be bitwise identical to `A[0]@B[0]`", and the same applies to reductions such as `A.sum(-1)[0]`. — [PyTorch notes: numerical_accuracy.md, "Batched computations or slice computations"](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.md)
- Hypothesis `floats(min_value=None, max_value=None, *, allow_nan=None, allow_infinity=None, allow_subnormal=None, width=64, exclude_min=False, exclude_max=False)`:
  - Without bounds, "subnormals, infinities, and NaNs are possible values".
  - `width=32` yields values "which can be exactly represented as a 32-bit float".
  - "Excluding either signed zero will also exclude the other".
  - The shrinker prefers finite values over infinity, and infinity over NaN.
  - `hypothesis.extra.numpy.arrays(dtype, shape, *, elements=None, fill=None, unique=False)`: inferred elements "may give any legal value (including eg NaN for floats)", and `fill` sets a single background value. Use `fill=nothing()` to draw every element independently.
  - — [Hypothesis strategies reference](https://hypothesis.readthedocs.io/en/latest/reference/strategies.html)
- `np.linalg.lstsq` raises `LinAlgError` on inf/−inf/NaN input. PyTorch documents no such guarantee: operations "may return a tensor with non-finite values, or raise an exception", and a computation may produce finite results on one backend and inf/NaN on another. — [NumPy test_nonfinite](https://github.com/numpy/numpy/blob/main/numpy/linalg/tests/test_linalg.py); [PyTorch numerical_accuracy.md "Extremal values" / "Non-finite values"](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.md)
- Own run: the exact rational check confirms the two properties DQRT17/DQRT14 test. For a degenerate box (lx = ux = c = 30.000000000001, y on a 5-point grid in [−1,1], random float z), the exact min-norm solution satisfies Xᵀ(z − Xb) = 0 **exactly** and (c, −1, 0)·b = 0 **exactly**, and X X⁺ X = X exactly.

### Inferences
**MR catalogue for the grid design.** Notation: z(i,j) is the value at x_i, y_j, with x_i = lx + i·(ux−lx)/4 and y_j likewise; full rank unless stated.

| # | Transformation (input) | Expected output relation | Exact in FP? | Kills |
|---|---|---|---|---|
| MR1 | z → 2^k·z (k ∈ {−60, −1, 1, 60}, no over/underflow) | b → 2^k·b | bitwise, if the solver is linear in z and avoids z-dependent branching | wrong scaling constants, z used in degeneracy test |
| MR2 | z → −z | b → −b | bitwise | sign/abs bugs, asymmetric clamps |
| MR3 | reverse the x-axis of the grid: z'(i,j) = z(4−i, j), box unchanged | B' = −B, C' = C, A' = A + B·(lx+ux) | tolerance (A' involves a product) | sign of B, wrong pivot (lx vs centre), off-by-one in i |
| MR4 | transpose: z'(i,j) = z(j,i), swap (lx,ux) ↔ (ly,uy) | B' = C, C' = B, A' = A | bitwise if the code is symmetric in x/y | row/column-major layout, swapped B/C, copy-paste of x-branch into y-branch |
| MR5 | z → z + p + q·x + r·y (dyadic p, q, r) | full rank: b → b + (p,q,r). Degenerate x (lx = ux = c): b → b + P_row(p,q,r), with P_row the projection onto span{(1,c,0), (0,0,1)} | tolerance | missing terms, wrong degenerate formula |
| MR6 | z = A0 + B0·x + C0·y exactly (dyadic coefficients and grid, so z is exact in FP) | b = (A0,B0,C0) to a few ulp·κ; R16 small | near-exact | everything in the formula |
| MR7 | shift box: lx,ux → lx+s, ux+s (same z) | full rank: B, C unchanged, A → A − B·s | tolerance; **exclude near the rcond boundary** | pivot-centring bugs |
| MR8 | scale box: x → 2^k·x (same z) | full rank: B → B/2^k, A and C unchanged | bitwise-ish; **rank decision is not column-scale invariant** | missing width normalisation |
| MR9 | idempotence: z → X·b (fitted values) | b unchanged | tolerance | projection errors |
| MR10 | linearity: solve(z1 + z2) = solve(z1) + solve(z2); three unit RHS → columns of X⁺ | linear | tolerance | non-linear branches on z |
| MR11 | batch invariance: same box at positions 0 and n−1, batch size 1 vs 65 537, shuffled order, NaN-padded neighbours | identical per-element result | bitwise expected for an element-wise closed form (no cross-batch reductions); otherwise ulp-level | indexing and broadcasting bugs, state leakage |
| MR12 | device/dtype: CPU f64 vs CUDA f64; MPS f32 vs CPU f64 | f64: R17 < 30 on both; f32 within κ-scaled f32 tolerance | no | dtype downcasts, TF32 |

**Exact closed forms for the degenerate cases** (derived here; checked with Fractions in the own run above). These make good oracles and also good mutation killers.
- *Degenerate x (lx = ux = c, ly < uy):* the fit is α + C·y with C the ordinary slope in y and α = mean(z) − C·mean(y). The min-norm split is **(A, B) = α·(1, c)/(1 + c²)**. Corollaries: B = c·A exactly, and A + c·B = α.
- *Point box (cx, cy):* the fit is the constant α = mean(z), and **b = α·(1, cx, cy)/(1 + cx² + cy²)**.
- *Degenerate y:* symmetric to degenerate x. **The project's kind list has `degenerate_x` and `point` but no `degenerate_y`. Add it**, because a copy-paste bug in the y branch would otherwise survive.

**Hypothesis / special-value strategy for box generation** (inference). Draw boxes as (centre, log10 width), not as (lx, ux):
- centre ∈ {0, ±tiny, ±5e-324 (subnormal), ±1, 25–35 (saturated), ±1e6, ±1e15}
- width ∈ {0, one ulp of the centre, 1e-16 … 1e2}
- the sign of zero (−0.0 vs 0.0) for lx/ux. lx = −0.0 and ux = +0.0 compare equal, so the `lx == ux` branch must treat them as degenerate.
- z from `floats(width=32)` for MPS runs, `allow_subnormal=True`, plus explicit cases with NaN or inf in one z entry or one bound.

Decide and document the contract for non-finite input. Possible contracts: NaN-propagate (mark the element), raise like NumPy, or match gelsd. Then test it per element, so that one NaN box does not poison its neighbours in the batch.

### Gaps
- I found no published metamorphic-testing study specific to least squares or linear-algebra solvers. The MR table above is derived here from the algebra of the design, not taken from a source.
- I did not verify whether `torch.linalg.lstsq` (gelsd) itself propagates NaN or raises for non-finite input on CPU. NumPy raises; PyTorch's notes disclaim any guarantee.

---

## 4. Exact / high-precision oracles, ULP vs relative error, condition-scaled tolerances

### Takeaway
An exact rational min-norm oracle on the float inputs is easy to build. It is **not the same spec as gelsd**, though: gelsd truncates singular values ≤ 25·eps·σ1. On near-degenerate boxes, which the project already generates, the exact pseudo-inverse is full rank while gelsd returns a rank-2 min-norm answer that differs by ~10 orders of magnitude. Separately, for ill-conditioned full-rank boxes, gelsd's own forward error reaches 1e-2 relative while a centred closed form is accurate to 1e-13–1e-15. The tests therefore need (a) an oracle that encodes gelsd's rcond truncation, (b) an "ambiguous band" around the threshold, and (c) primary metrics that are backward-error ratios (R17, R18), independent of κ, rather than forward comparison against gelsd.

### Cited Findings
- `mpmath.pinv(A, rtol=None)`: "In the overdetermined case, it provides the least squares solution. In the underdetermined case, it provides the minimum norm solution", with the default rtol documented as "ctx.eps * max(A.rows, A.cols)". **Implementation caveat found in the source:** it sets `rtol = max(A.rows, A.cols) * S[0] * ctx.eps` and then keeps singular values with `val > rtol * max(S)`. The default cut-off is therefore max(m,n)·eps·σ1², not max(m,n)·eps·σ1 as documented. `if not rtol` also silently turns rtol = 0 into the default. — [mpmath/matrices/linalg.py `pinv` ~l.313](https://github.com/mpmath/mpmath/blob/master/mpmath/matrices/linalg.py)
- `mpmath.qr_solve` raises `'cannot solve underdetermined system'` and does not handle rank deficiency; it works at `ctx.prec += 10` bits. — same file (~l.423)
- The LS forward-error bound has the form ERRBD = eps·(2/(RCOND·cosθ) + tanθ/RCOND²), with sinθ = ||r||/||b|| and RCOND = σ_min/σ_max. The error grows like κ when the residual is small and like κ²·tanθ when it is not. — [LAPACK Users' Guide node82](https://www.netlib.org/lapack/lug/node82.html)
- Structured backward error and condition numbers (including least-squares applications) are developed in Higham & Higham 1992, SIAM J. Matrix Anal. Appl. — [Higham & Higham, "Backward error and condition of structured linear systems"](https://webhomes.maths.ed.ac.uk/~dhigham/Publications/P12.pdf) (search-result summary only; not read in full).
- NumPy provides ULP tools (`assert_array_max_ulp`, `nulp_diff`). — [numpy/testing/_private/utils.py](https://github.com/numpy/numpy/blob/main/numpy/testing/_private/utils.py)
- **Own run (rank decision, gelsd with default rcond = 25·eps ≈ 5.55e-15):** X = [1, x, y] on a 5x5 grid, x = c + w·t with t ∈ {−.5, −.25, 0, .25, .5}, y ∈ [0,1] or [−1,1].

  | centre c | width w | computed σ3/σ1 | gelsd rank |
  |---|---|---|---|
  | 0 | 1e-13 | 3.1e-14 | 3 |
  | 0 | 1e-14 | 3.1e-15 | **2** |
  | 1 | 1e-13 | 1.7e-14 | 3 |
  | 1 | 1e-14 | 1.7e-15 | **2** |
  | 30 | 1e-10 | 3.9e-14 | 3 |
  | 30 | **1e-12** (the project's `near_degenerate` width) | 3.9e-16 | **2** |

  With mpmath at 50 digits the true σ3/σ1 for c=30, w=1e-12 is 3.93e-16 < 5.55e-15. gelsd's rank-2 decision is "correct per spec", while the exact rank is 3.
- **Own run (consequence):** for c=30, w=1e-12, gelsd returns b ≈ (0.00111, 0.0333, 0.3), the min-norm answer. The full-rank centred closed form returns (−6.68e9, 2.23e8, 0.3). For c=1, w=1e-14: gelsd gives (0.441, 0.441, 0.3) versus the full-rank (7.46e8, −7.46e8, 0.3).
- **Own run (accuracy against an exact Fraction oracle).** The oracle solves the normal equations exactly in rationals on the float inputs, valid because these boxes are exactly full rank.

  | box | cond(X) | gelsd relative error (A, B, C) | centred closed form relative error |
  |---|---|---|---|
  | c=30, w=1e-10 | 2.5e13 | 7.6e-5, 7.6e-5, 2.3e-7 | ≤ 1e-15 |
  | c=1, w=1e-12 | 5.7e12 | **1.1e-2**, 1.1e-2, 7.5e-9 | 5.4e-13 |
  | c=0.3, w=1e-3 | 3.1e3 | 3e-14, 1.4e-13, 2e-16 | ≤ 6e-16 |

- **Own run (exact rational min-norm for rank-deficient X).** With a full-rank factorisation X = F·G, where F = [1, y] (25x2) and G = [[1,c,0],[0,0,1]], the formula X⁺ = Gᵀ(GGᵀ)⁻¹(FᵀF)⁻¹Fᵀ evaluated in `fractions.Fraction` satisfies X X⁺ X = X exactly, Xᵀr = 0 exactly and (c,−1,0)·b = 0 exactly.

### Inferences
- **Two oracles, two questions.**
  - (O1) *Exact pseudo-inverse on the float inputs* (Fractions; the rank is decided exactly, i.e. lx == ux). This is the right oracle for "is the closed form correct as mathematics".
  - (O2) *gelsd-spec oracle*: high-precision SVD (mpmath, dps ≥ 50, or exact rational eigenvalues of XᵀX for a 3x3 matrix), apply the truncation σ_i ≤ 25·eps·σ1 exactly, then form the truncated-SVD solution in high precision. This is the right oracle for "is it a drop-in replacement for gelsd".
  - The project must decide which spec the GPU solver implements. The own runs show they disagree on realistic boxes (c=30, w=1e-12 is the project's own `saturated` + `near_degenerate` combination).
  - If the spec is "match gelsd", the closed form needs a rank test equivalent to σ3/σ1 ≤ 25·eps rather than `lx == ux`. For the grid design σ3/σ1 can be computed in closed form from the centred 3x3 Gram matrix.
  - Do not use `mpmath.pinv` with its default rtol because of the threshold caveat above. Pass an explicit absolute cut-off or write the truncated SVD yourself.
- **Ambiguous band.** Computed singular values carry error ~p(n)·eps·σ1, so for O2 mark any box with σ3/σ1 ∈ [rcond/8, 8·rcond] as "rank-ambiguous". On those boxes accept either answer, but require R17 < 30 and, if rank 2 is returned, R18 < 30. Generate boxes deliberately inside the band: for c = 0 the boundary width is ≈ 1.8e-14, for c = 1 ≈ 3.3e-14, for c = 30 ≈ 1.4e-11 (linear interpolation of the own-run σ3/σ1 values).
- **Metrics hierarchy.**
  1. **Backward-style ratios** (R17 normal equations, R18 row space, R16 for consistent data), with THRESH = 30. These are independent of κ, so the same threshold works for every box kind.
  2. **Forward error against O1/O2**, with tolerance `k · u · (κ + κ²·tanθ)`, where κ = σ1/σ3 from the oracle and tanθ = ||r||/||X b||. Use k ≈ 10–30 for f64 on CPU/CUDA and u = 2^-24 for f32 on MPS.
  3. **ULP distance** (`nulp_diff`) only where an exact or bitwise expectation exists (MR1, MR2, MR4, the consistent kind). Do not use ULPs for ill-conditioned forward errors.
- **Never compare the GPU solver to gelsd at a fixed tight tolerance.** On κ ~ 1e12–1e13 boxes gelsd is itself off by up to 1e-2, so a correct, more accurate solver would "fail". Compare both to the exact oracle, and report the GPU-vs-gelsd difference only as a diagnostic.
- **f32 on MPS.** κ·u_f32 exceeds 1 for κ ≳ 1.7e7, i.e. any box with relative x-width below ~1e-6 at centre ~30. For those boxes, either define the f32 contract as "R17 < 30 and finite" only, or route them to the degenerate branch with an f32-specific rcond (25·2^-23 ≈ 3e-6).

### Gaps
- I did not obtain a primary-source statement of the full Wedin/Higham least-squares perturbation bound (the Higham "Accuracy and Stability" chapter was not fetched). The LAPACK Users' Guide fragment above is the cited form.
- I did not verify how gelsd computes σ internally (bidiagonal divide and conquer) compared with numpy's gesdd `svd`. The ambiguous-band width (factor 8) is a heuristic, not a sourced constant.

---

## 5. Mutation testing for numerical code: meaningful mutants for the closed-form solver and how to kill them

### Takeaway
Generic operator mutation (mutmut) catches arithmetic-operator and constant slips. The bugs that matter for this solver are semantic: degenerate-branch formula, degeneracy predicate, pivot centring, grid layout, and output order. These need hand-planted mutants. Many mutants are "near-equivalent" on easy inputs (centre 0, well-conditioned boxes), so the suite must include c ≠ 0, large c, ill-conditioned and degenerate_y cases, and compare against the exact oracle with tight, κ-aware tolerances.

### Cited Findings
- Wu & Mitchell, "Mutant Accuracy Testing for Assessing the Implementation of Numerical Algorithms". Because oracles and outputs are both approximate, they "compare the error of target implementations to mutated versions of themselves". On MATLAB implementations "most mutants are worse while the few which are better show a distinctive pattern of mutation". — [Wu & Mitchell (PDF)](https://www.cs.toronto.edu/~rwu/papers/wu2019mutation.pdf)
- Magneto (ISSTA 2021) applies mutation testing to numerical specifications implemented in floating-point code, using interval constraints to check accuracy. — [Interval Constraint-Based Mutation Testing of Numerical Specifications](https://conf.researchr.org/details/issta-2021/issta-2021-technical-papers/18/Interval-Constraint-Based-Mutation-Testing-of-Numerical-Specifications)
- In a Java study, about 45% of undetected mutants were equivalent. Coverage-impact can flag non-equivalent mutants with 75% precision and 56% recall. — [Schuler & Zeller, STVR](https://www.st.cs.uni-saarland.de/publications/files/schuler-stvrbis-2013.pdf) (figures from the search-result summary; PDF not read in full)
- Hook's thesis on scientific code introduces "Mutation Sensitivity Testing", which measures a test set's sensitivity rather than binary kills. — [Using Code Mutation to Study Code Faults in Scientific Software (Queen's)](https://qspace.library.queensu.ca/items/80f7b8df-8ae7-489a-b5f1-a287c3821779) (search-result summary)
- mutmut (3.x):
  - mutates only code inside functions ("If you want to mutate code outside of functions, you can try using mutmut 2")
  - by default mutates only functions that are called, with optional line-level coverage filtering
  - supports `# pragma: no mutate`
  - can filter invalid mutants through mypy/pyrefly
  - — [mutmut README](https://github.com/boxed/mutmut/blob/main/README.rst)

### Inferences
**Planted-mutant catalogue.** Each mutant is listed with the test that must kill it and the "equivalence trap", meaning the inputs on which it survives.

| ID | Mutant | Killed by | Survives on (avoid relying only on these) |
|---|---|---|---|
| M1 | sign flip on B or C | MR6 consistent data, MR3 reflection | z constant in x (B = 0) |
| M2 | swapped output order (A↔B, B↔C) | MR6, MR4 transpose | symmetric boxes with symmetric z |
| M3 | degenerate-x branch returns (mean, 0, C) instead of α(1,c)/(1+c²) | R18 row-space test, O1 on lx = ux = c ≠ 0 | **c = 0** (both formulas coincide), so use c = 30 and c = 1e6 |
| M4 | degeneracy predicate on mean/relative width (`abs(ux−lx) < tol·abs(mean)`) or `std(x) == 0` instead of `lx == ux` (or the reverse, depending on the chosen spec) | rcond-band sweep (§4) against the chosen oracle, near_degenerate with c ∈ {0, 1, 30} | widths far from the threshold |
| M5 | missing pivot centring (raw normal equations on uncentred x) | O1 with tight tol on ill-conditioned boxes (c = 30, w ≤ 1e-8): error jumps from ~1e-15 to ≥ 1e-5 | well-conditioned boxes (c ≈ 0, w ~ 1); the mutant is near-equivalent there |
| M6 | grid layout off-by-one (spacing w/5 instead of w/4; index i·5+j vs j·5+i; `linspace(…, 4)`) | MR6, MR4, an asymmetric hand-computed 5x5 example | z constant or symmetric under transpose |
| M7 | wrong denominator (Σ(x−x̄)² with 24 or 5 instead of 25 terms; n−1 normalisation) | MR6 | z constant |
| M8 | y branch copy-pasted from x (uses lx/ux in the y-degenerate case) | **degenerate_y** kind (missing from the current kinds), MR4 | degenerate_x-only suites |
| M9 | point-box denominator 1+cx² instead of 1+cx²+cy² | point boxes with both cx, cy ≠ 0, R18 | points with cy = 0 |
| M10 | internal downcast to f32 or TF32 matmul in f64 mode | f64 O1 comparison at k·u·κ with k ≤ 30 | loose 1e-5 tolerances (PyTorch-style) |
| M11 | branch blending by arithmetic mask (`m*a + (1−m)*b`) instead of `torch.where`, so inf/NaN from the unused branch leaks (0·inf = NaN) | degenerate and point boxes: `isfinite` on every output | full-rank boxes |
| M12 | batch indexing (box[0] broadcast to all, wrong stride for n > 65 535) | MR11, heterogeneous batch, n = 65 537 | homogeneous batches |
| M13 | rcond off by a factor (eps instead of 25·eps, or σ instead of σ²) in an SVD/pinv variant | band sweep at w around the boundary | widths outside the band |

**Kill protocol** (inference):
1. Run each planted mutant (a separate function, or a flag in the solver) against the full suite and require 100% kills.
2. For a survivor, first check whether it is equivalent on the whole input domain (e.g. M3 when all generated c = 0). If it is not, add the minimal distinguishing case.
3. Run mutmut on the solver module for operator-level coverage. Mark the dtype or device plumbing lines with `# pragma: no mutate` to cut noise.
4. Optionally report Wu & Mitchell–style "accuracy degradation": the median error ratio of mutant vs original against O1, as a sensitivity measure for the tolerance choice.

### Gaps
- I found no published mutation-operator set specific to linear least-squares or closed-form regression solvers. The table is derived from the solver's structure.
- The cosmic-ray documentation was not fetched. I cannot state its operator list or its PyTorch compatibility.

---

## 6. GPU-specific concerns: TF32, FMA, reduction order, determinism, CPU vs CUDA vs MPS tolerances

### Takeaway
Whether TF32 is on depends on configuration, and PyTorch's own tests run float32 matmul-based checks with TF32 both on and off at a 1e-5 precision. FMA contraction and different reduction orders make CPU and GPU results differ at the ulp level, and batched results are not guaranteed bitwise equal to unbatched ones. Extremal inputs can be finite on one backend and inf/NaN on another. Tests should therefore:
- pin or force the TF32 and determinism settings
- use ulp-level or κ-scaled comparisons across devices, never bitwise ones (except where the kernel is designed to be bitwise)
- run the special-value suite on every backend

### Cited Findings
- TF32 on Ampere and later reads "only the first 10 bits of the input mantissa". It "may reduce accuracy and produce surprising results (e.g., multiplying a matrix by the identity matrix may produce results that are different from the input)". It is "disabled for matrix multiplications and enabled for convolutions" by default. It is enabled with `torch.backends.cuda.matmul.fp32_precision = "tf32"`; `allow_tf32 = True` "is going to be deprecated". — [PyTorch notes numerical_accuracy.md](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.md)
- PyTorch's test helper `tf32_on_and_off(tf32_precision=1e-5)` runs a test twice, with TF32 on and off. With TF32 on, the comparison uses the reduced precision, and only "on a CUDA >= 11 build on an >=Ampere architecture" for float32/complex64. Related helpers are `tf32_off`, `tf32_on` and `with_tf32_off`, and the code notes that the `allow_tf32` setter must be restored together with `fp32_precision`. — [torch/testing/_internal/common_cuda.py ~l.465–607](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_cuda.py)
- FMA "computes rn(X × Y + Z) with only one rounding step", whereas the unfused form is rn(rn(X×Y)+Z). The guide's example: a fused result off by 0.00004 versus 0.00064 unfused. — [NVIDIA, Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://docs.nvidia.com/cuda/floating-point/)
- Batched results are not bitwise equal to per-slice results, reductions included. "Extremal" inputs "may produce finite results on one backend and non-finite results, such as inf or NaN, on another", and linalg results for ill-conditioned inputs "may vary when using the same inputs on different devices or when using different backends via the keyword driver". — [PyTorch notes numerical_accuracy.md](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.md)
- `torch.use_deterministic_algorithms` makes PyTorch use deterministic algorithms where available and throw "if an operation is known to be nondeterministic (and without a deterministic alternative)". — [PyTorch notes randomness.md](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md)
- The PyTorch default comparison tolerances are float32 (1.3e-6, 1e-5) and float64 (1e-7, 1e-7). The lstsq tests use a fixed 1e-5 on both CPU and CUDA, and the OpInfo skips some MPS float32 lstsq consistency tests. — [_comparison.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py), [test_linalg.py](https://github.com/pytorch/pytorch/blob/main/test/test_linalg.py), [opinfo linalg.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/opinfo/definitions/linalg.py)

### Inferences
- **TF32.** Write the CUDA f32 path (if any) without `@`/`matmul`: use explicit element-wise sums over the 25 grid points. TF32 then cannot apply. Add a test that sets `torch.backends.cuda.matmul.fp32_precision = "tf32"` and asserts bitwise-identical output to the `"ieee"` setting, which kills M10. For f64, TF32 never applies, but assert the dtype of every intermediate tensor.
- **FMA and reduction order.** For a closed form built from five fixed sums of 25 terms, fix the summation order explicitly. Use a pairwise/tree order identical on all backends, or Kahan/compensated sums if R17 margins are tight. Expect CPU vs CUDA f64 differences of a few ulp per sum. The comparison metric across devices is |b_cuda − b_cpu| ≤ k·u·κ·|b|, or simply both having R17 < 30, never `torch.equal`. This matches the project's earlier observation that batched linspace on arm64 is not bit-exact because of FMA (memory note, 7 Sep 2026; not re-verified here).
- **Determinism.** Run the full suite twice on CUDA with `torch.use_deterministic_algorithms(True)` and require bitwise-identical outputs run-to-run. Same-device, same-batch bitwise reproducibility is achievable for a pure element-wise kernel and catches race conditions in custom kernels.
- **Batch-size invariance.** Assert that element i is bitwise identical for batch sizes 1, 2, 1024 and 65 537, and with the element at a different index. A closed form without cross-batch reductions should satisfy this; if it does not, an indexing or tiling bug is likely.
- **MPS (float32 only).** Run the f32 suite with eps_f32: R17/R18 thresholds of 30 with eps = 2^-23. Use rcond_f32 = 25·2^-23 ≈ 3e-6 if the solver matches gelsd semantics in f32, and remember that the near-degenerate band moves to widths ~1e-5–1e-4 at c ≈ 30 in f32. Compare MPS f32 to the CPU f64 exact oracle, not to CPU f32 gelsd.
- **Special values per backend.** Run the special-value Hypothesis suite on CPU, CUDA and MPS separately: subnormal widths, −0.0 bounds, NaN/inf z, and centres near 3.4e38 for f32. PyTorch explicitly says finiteness can differ across backends. Subnormal handling (flush-to-zero) on GPU and MPS is the most likely divergence point.
- **Tolerance table to adopt** (inference, combining all sections):

  | metric | f64 CPU/CUDA | f32 MPS |
  |---|---|---|
  | R16 / R17 / R18 | < 30 | < 30 (eps = 2^-23) |
  | forward error vs exact oracle | ≤ 30·u·(κ + κ²·tanθ) | ≤ 30·u·(κ + κ²·tanθ), u = 2^-24 |
  | CPU vs CUDA | same forward bound | n/a |
  | MR1, MR2, MR4, batch invariance, run-to-run | bitwise | bitwise |
  | rank decision vs gelsd (O2) | exact match outside the ambiguous band | exact match outside the band |

### Gaps
- No primary source fetched in this session confirms that MPS lacks float64 (the project's memory notes say so). I also did not verify MPS subnormal/flush-to-zero behaviour, or whether nvcc's default `--fmad=true` applies to PyTorch's own CUDA kernels. These should be checked empirically on the target hardware (Colab/Modal A100 for CUDA, M5 Pro for MPS).
- I did not check whether PyTorch's CPU gelsd path (MKL/OpenBLAS/Accelerate) gives bitwise-identical results across BLAS backends. If the CPU gelsd reference changes between the Mac (Accelerate) and Linux (MKL/OpenBLAS), any "match gelsd" tests near the rank boundary may flip between machines.
