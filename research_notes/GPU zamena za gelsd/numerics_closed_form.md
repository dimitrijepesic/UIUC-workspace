# Numerics of replacing LAPACK gelsd with a closed-form solver for the 25x3 Cartesian-grid plane fit

Scope: for each box [lx,ux] x [ly,uy], fit z ~ C + A*x + B*y on a 5x5 Cartesian grid (25 x 3 design X = [1, x, y]); reference = `torch.linalg.lstsq(driver='gelsd')` with torch default rcond. float64 is the main case, float32 is secondary.

Labels used below:
- **[SOURCED]**: in Cited Findings. Each item has a URL.
- **[DERIVED]**: my own algebra. Not taken from a source.
- **[EMPIRICAL]**: experiments I ran in this session with numpy 2.5.4 and scipy 1.18.1 on Apple Accelerate LAPACK (macOS arm64). They used `scipy.linalg.lstsq(..., cond=25*eps, lapack_driver='gelsd'|'gelss'|'gelsy')` and an exact rational (`fractions.Fraction`) normal-equation reference. Scripts are in `/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/exp/` (`gelsd_rank.py`, `gelsd_rank_d.py`, `fwd_err.py`). The x/y grids use `np.linspace`, which is not bit-identical to `torch.linspace`. Accelerate is not MKL or OpenBLAS, so torch on x86 may differ in the last bits near thresholds.

---

## Q1. How LAPACK xGELSD / xGELSS / xGELSY decide rank, and how torch picks rcond

### Takeaway
gelsd zeroes singular values with s_i <= RCND * s_max. In the code, RCND = RCOND only when 0 < RCOND < 1. Otherwise RCND = DLAMCH('E') = 2^-53, so the documented "RCOND < 0" case also covers RCOND = 0 and RCOND >= 1. The only scaling is a single scalar rescale of A (and of B) when max|a_ij| lies outside [~1e-292, ~1e292], plus an internal normalisation of the bidiagonal. There is no column equilibration, so the rank decision depends on column scaling, i.e. on |mean x| and |mean y|. torch's default is rcond = finfo(dtype).eps * max(m,n) = 25 * 2^-52 ≈ 5.55e-15 (float64) and 25 * 2^-23 ≈ 2.98e-6 (float32). That value lies in (0,1), so gelsd uses it directly.

### Cited Findings
- DGELSD purpose: "computes the minimum-norm solution to a real linear least squares problem: minimize 2-norm(|b - A*x|) using the singular value decomposition (SVD) of A. A is an M-by-N matrix which may be rank-deficient." It has three steps: (1) Householder bidiagonalisation, (2) solve the bidiagonal LS "using a divide and conquer approach", (3) apply the transformations back. The purpose text says: "The effective rank of A is determined by treating as zero those singular values which are less than RCOND times the largest singular value." — [dgelsd.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelsd.f)
- DGELSD argument doc: "RCOND is used to determine the effective rank of A. Singular values S(i) <= RCOND*S(1) are treated as zero. If RCOND < 0, machine precision is used instead." RANK is "the number of singular values which are greater than RCOND*S(1)". The purpose text says "less than" while the argument doc and the code use "<=". — [dgelsd.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelsd.f)
- DGELSD scaling code:
  - It sets EPS = DLAMCH('P'), SFMIN = DLAMCH('S'), SMLNUM = SFMIN/EPS and BIGNUM = 1/SMLNUM, then ANRM = DLANGE('M', ...) (max-abs entry).
  - If 0 < ANRM < SMLNUM it calls `DLASCL('G',0,0,ANRM,SMLNUM,M,N,A,...)`. If ANRM > BIGNUM it scales to BIGNUM. If ANRM = 0 it sets the solution and S to 0 and returns rank 0.
  - B gets the same treatment with BNRM. The solution and S are unscaled at the end.
  - This is one scalar factor for the whole matrix. It is not column equilibration. — [dgelsd.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelsd.f)
- DGELSD path for M >= N:
  - If M >= MNTHR it first does DGEQRF (QR) and applies Q^T to B. It then calls DGEBRD on the N x N R and DLALSD('U', SMLSIZ, N, ...).
  - MNTHR = ILAENV(6,'DGELSD',...) and SMLSIZ = ILAENV(9,'DGELSD',...).
  - ILAENV(6) returns INT(MIN(M,N)*1.6) and ILAENV(9) returns 25. — [dgelsd.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelsd.f), [ilaenv.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/ilaenv.f)
- DLALSD tolerance code (verbatim logic):
  - `EPS = DLAMCH('Epsilon')`, then `IF((RCOND.LE.ZERO).OR.(RCOND.GE.ONE)) THEN RCND = EPS ELSE RCND = RCOND`.
  - The bidiagonal (D,E) is scaled by ORGNRM = DLANST('M',N,D,E) (max-abs entry) to unit size.
  - If N <= SMLSIZ, DLASDQ computes the bidiagonal SVD, then `TOL = RCND*ABS(D(IDAMAX(N,D,1)))`. Each D(I) <= TOL gets its rows of B zeroed. Otherwise B is divided by D(I) and RANK = RANK+1.
  - Then it multiplies by the right singular vectors (DGEMM) and unscales.
  - The divide-and-conquer branch (N > SMLSIZ) also replaces |D(I)| < EPS by SIGN(EPS, D(I)) before D&C. That branch is never taken for N = 3. — [dlalsd.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dlalsd.f)
- DLALSD argument doc: "The singular values of A less than or equal to RCOND times the largest singular value are treated as zero in solving the least squares problem. If RCOND is negative, machine precision is used instead." — [dlalsd.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dlalsd.f)
- DLAMCH: with rounding (RND = 1), EPS = EPSILON(0d0)*0.5. 'E' returns EPS, which is 2^-53 for double. 'P' returns EPS*RADIX, which is 2^-52. 'S' returns the safe minimum. — [dlamch.f](https://github.com/Reference-LAPACK/lapack/blob/master/INSTALL/dlamch.f)
- DGELSS rank rule:
  - Uses DBDSQR, then `THR = MAX(RCOND*S(1), SFMIN)`, and `IF (RCOND.LT.ZERO) THR = MAX(EPS*S(1), SFMIN)` with EPS = DLAMCH('P') = 2^-52.
  - It keeps S(I) only if `S(I).GT.THR`.
  - With RCOND = 0, gelss uses THR = SFMIN (essentially no truncation), whereas gelsd with RCOND = 0 uses 2^-53. The same "rcond = 0" therefore means different things in gelss and gelsd.
  - It does the same SMLNUM/BIGNUM scalar scaling of A and B. — [dgelss.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelss.f)
- DGELSY rank rule:
  - RCOND doc: "RCOND is used to determine the effective rank of A, which is defined as the order of the largest leading triangular submatrix R11 in the QR factorization with pivoting of A, whose estimated condition number < 1/RCOND."
  - Code: DGEQP3 (column-pivoted QR). If |R(1,1)| = 0, rank = 0. Otherwise it grows the rank using incremental condition estimation (DLAIC1 for SMINPR and SMAXPR) and accepts column i `IF (SMAXPR*RCOND .LE. SMINPR)`.
  - The doc has no special case for RCOND < 0.
  - The rank-deficient solution comes from the complete orthogonal factorisation (DTZRZF/DORMRZ). It is the minimum-norm solution of the truncated problem. — [dgelsy.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelsy.f)
- torch.linalg.lstsq:
  - "rcond is used to determine the effective rank of the matrices in A when driver is one of ('gelsy', 'gelsd', 'gelss'). If rcond= None (default), rcond is set to the machine precision of the dtype of A times max(m, n)."
  - Warning: "The default value of rcond may change in a future PyTorch release. It is therefore recommended to use a fixed value."
  - Default driver: "'gelsy' is used for CPU inputs and 'gels' for CUDA inputs". "For CUDA input, the only valid driver is 'gels', which assumes that A is full-rank."
  - rank is returned for gelsy, gelsd and gelss. singular_values are returned for gelsd and gelss. — [torch.linalg.lstsq docs (2.14)](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.lstsq.html)
- torch.finfo.eps is "The difference between 1.0 and the next smallest representable float larger than 1.0", so it equals 2^-52 for float64 and 2^-23 for float32. This is DLAMCH('P'), not DLAMCH('E'). — [torch.finfo docs](https://docs.pytorch.org/docs/2.14/type_info.html)

### Inferences
- [DERIVED] Path for a 25x3 problem: MNTHR = INT(3*1.6) = 4 <= 25, so gelsd runs QR, then bidiagonalises the 3x3 R, then DLALSD with N = 3 <= SMLSIZ = 25, which uses DLASDQ (bidiagonal QR iteration, not divide and conquer). The rank test is exactly "computed s_i <= 5.55e-15 * computed s_1" (float64) on singular values of the scalar-rescaled matrix. The test is invariant to the global scalar rescale and **not** invariant to column scaling or affine shifts of x and y.
- [DERIVED] Mimicking torch's default exactly therefore needs rcond_eff = 25 * finfo.eps with the comparison `s3 <= rcond_eff * s1` (non-strict). If someone passes rcond = 0 or a negative value to gelsd, the effective rcond becomes 2^-53, not 0.
- [DERIVED] Because the comparison uses the *computed* s3, which has absolute error O(u * s1) (u = 2^-53), no external rule can reproduce gelsd bit-for-bit in a thin band around the cutoff. The relative half-width of that band in the width variable is roughly c*u/rcond = c/50, i.e. ~2% for c ~ 1. Outside the band, a rule based on exact singular values reproduces gelsd (see Q4).
- [EMPIRICAL] For exactly degenerate boxes (all y equal), gelsd's computed s3/s1 was at most 0.98 * 2^-52 over 4000 random boxes (float64) and at most 0.86 * 2^-23 over 2000 boxes (float32). That is ~25x below the cutoff, so exact-endpoint detection always agreed with gelsd on exact degeneracy. 199 of the 4000 y-degenerate float64 boxes came back with **rank 1**, because their x-width was also below the x threshold (see Q4). Exact-endpoint detection alone would call them rank 2.

### Gaps
- I did not read torch's C++ source (BatchLinearAlgebra.cpp) to confirm it passes rcond to LAPACK unchanged and loops per batch element. The docs imply this but I did not verify it.
- I did not verify whether MKL or OpenBLAS ship modified gelsd thresholds. The reference code was read, but vendor builds may differ.

---

## Q2. Minimum-norm least squares for a rank-deficient design, and the jump when a singular value crosses the cutoff

### Takeaway
Truncating the SVD solution at s_k <= tau gives the min-norm solution of a nearby rank-k problem. The pseudo-inverse is continuous only along perturbations that keep the rank. When s3 crosses the cutoff, the solution jumps by the full component of the coefficient vector along v3. For this problem the jump is O(1) (≈0.42 in the test case), not O(eps). Just above the cutoff, gelsd's own answer is dominated by the kappa^2 * tan(theta) error term and can be wrong by 10^6 or more.

### Cited Findings
- The pseudoinverse is continuous under perturbations dM with ||M^+||_2 ||dM||_2 < 1 **iff** rank(M + dM) = rank(M). "lim (M+dM)^+ = M^+ iff rank(M+dM) = rank(M)". For full-column-rank matrices the rank condition holds automatically. — [arXiv 2306.12177](https://arxiv.org/pdf/2306.12177)
- Classical perturbation theory of pseudo-inverses, projections and LS problems is surveyed in G.W. Stewart, SIAM Review 19 (1977), "On the Perturbation of Pseudo-Inverses, Projections and Linear Least Squares Problems". Only the metadata was accessible, not the text. — [SIAM Review](https://epubs.siam.org/doi/10.1137/1019104)
- LS sensitivity: relative sensitivity to perturbations in A is bounded by (kappa(A)^2 * tan(theta) + kappa(A)) times ||dA||/||A||, where theta is the angle between b and range(A). "the sensitivity of the least squares problem to perturbations in A can quickly be dominated by kappa(A)^2 tan(theta) if kappa(A) is at all large". — [Cornell CS 6210 notes, 2025-10-01](https://www.cs.cornell.edu/courses/cs6210/2025fa/lec/2025-10-01.html) (from search-result summary; the page itself failed TLS fetch)
- Tight version: the condition number of x with respect to A lies between kappa_2*sqrt((v*tan(theta))^2 + 1) and kappa_2*(v*tan(theta) + 1), where v is van der Sluis's ratio in [1, kappa_2]. — [Grcar, arXiv 1003.2733](https://arxiv.org/pdf/1003.2733)
- The kappa^2 factor drops out only when tan(theta) ≈ 0, i.e. when the data fit almost exactly. — [arXiv 2412.19960](https://arxiv.org/pdf/2412.19960)
- Golub & Van Loan "Matrix Computations" and Björck "Numerical Methods for Least Squares Problems" (1996) are the standard references for min-norm LS and truncated SVD. I could not access their text in this session. See Gaps.

### Inferences
- [DERIVED] Min-norm solutions:
  - **x constant (x_i = x0):** null(X) = span((-x0, 1, 0)). The LS solutions are theta_p + t*(-x0, 1, 0). Min-norm requires theta ⟂ null(X), i.e. A = x0*C. Combined with C + A*x0 = k (k = mean(z) - B*mean(y)), this gives C = k/(1+x0^2), A = k*x0/(1+x0^2). B is determined uniquely by the centred y-column.
  - **y constant:** the mirror case is (C, B) = k'(1, y0)/(1+y0^2), with k' = mean(z) - A*mean(x).
  - **Both constant:** null(X) = {theta : C + A*x0 + B*y0 = 0} is 2-D, so theta ∝ (1, x0, y0) and theta = mean(z)*(1, x0, y0)/(1+x0^2+y0^2).
  - These match the formulas in the brief.
- [DERIVED] Jump size, general form. If the computed s3 moves from just above tau to just below it, the solution changes from sum_{i<=3} (u_i^T z/s_i) v_i to sum_{i<=2}. The jump is (u3^T z/s3) * v3, i.e. exactly the v3-component of the full-rank solution. Its size is |u3^T z|/s3 ≥ |u3^T z|/(rcond*s1) just above the cutoff. It is bounded only if the data are nearly consistent along u3.
- [DERIVED] Jump size, this problem (y-degenerate side). Write a = LS slope of s'(x) on the x-grid and s̄ = mean_i s'(x_i).
  - Because z_ij = s'(x_i)*y_j is separable, the exact full-rank fit on any Cartesian grid is z ≈ s̄*y + a*ȳ*(x - x̄). So B = s̄, A = a*ȳ and C = -a*ȳ*x̄. The (x-x̄)(y-ȳ) part of z is orthogonal to span{1, x, y}.
  - As w_y → 0, v3 → (-y0, 0, 1)/sqrt(1+y0^2). The gelsd-truncated solution minus the full-rank solution tends to [(s̄ + a*x̄*y0^2)/(1+y0^2)] * (y0, 0, -1).
- [EMPIRICAL] For x in [1.75, 2.25] and y0 = 3, the formula above predicted a jump of (-0.399, 0, +0.133). gelsd(rank 2) minus the exact full-rank solution was (-0.386 … -0.400, 0, +0.129 … +0.133) for w_y between 2e-15 and 1e-13. They agree.
- [EMPIRICAL] Same case at w_y = 1e-12, where gelsd returns rank 3: gelsd gave (3.38e6, -0.239, -1.13e6) against the exact (0.479, -0.239, 0.106). Relative error 7e6. Relative error against the exact rational solution of the same rounded data, as w_y grows:

  | w_y | gelsd | gelss | gelsy | pivot-centred closed form |
  |---|---|---|---|---|
  | 1e-12 | 7e6 | 7e6 | 4e6 | 1.2e-5 |
  | 1e-10 | 1.5e3 | | | 8e-8 |
  | 1e-8 | 5.8e-2 | | | 9.5e-10 |
  | 1e-6 | 1.3e-5 | | | 1.2e-11 |
  | 1e-4 | 3.4e-10 | | | 1.2e-13 |
  | 1e-3 | 1.6e-11 | | | 1.4e-14 |
  | 1e-1 | 5.7e-15 | | | 1.2e-16 |

  - ||r|| ≈ 7.5e-3 throughout, from the curvature of s' over wx = 0.5.
  - gelsd error scales like 1/w^2. That is the kappa^2 * tan(theta) term, with kappa ∝ 1/w (Q4) and a fixed residual.
  - The closed form scales like 1/w. That is the irreducible sensitivity of B to O(u*|z|) rounding in z and z - z̄.
  - Narrow-x case (x̄ = 10, wx from 1e-11 to 1e-3): gelsd 1.9e-4 … 9.5e-13; closed form 1.9e-6 … 2.6e-14.
  - This is consistent with the user's earlier "9e-15 vs 7.6e-12" observation.
- [DERIVED] Consequence: in a band of widths above the cutoff, gelsd is not a trustworthy reference. With the residual above, its float64 error exceeds 1e-8 for w_y ≲ 2e-5. Below the cutoff it returns a well-defined but O(1)-different truncated answer. "Agreement with gelsd" should be tested only for (a) exactly degenerate boxes and (b) boxes wide enough that kappa^2 * u * tan(theta) << tolerance. In between, compare against an exact or extended-precision reference instead.

### Gaps
- I found no source that states the "jump" magnitude for truncated SVD in this exact form. The general term (u_k^T b/s_k)*v_k is standard truncated-SVD algebra, but I did not obtain a citable page (GVL §5.5 / Björck §2.7 would be the natural citations, but I could not fetch them).
- I could not access Stewart (1977) full text for an explicit discontinuity bound such as ||B^+|| >= 1/||E|| under a rank increase.

---

## Q3. Centring, conditioning with an intercept, and accurate variances/covariances (why a data-point shift is stable)

### Takeaway
Uncentred regression with an intercept has "non-essential" ill-conditioning that centring removes. For variances, Chan, Golub & LeVeque show that shifting by any data point gives condition number kappa^2 <= 1 + N. Shifting by the computed floating-point mean gives only kappa^2 <= 1 + N^2 * kappa^2 * u^2, and the computed mean can even fall outside [min x, max x]. With a data-point pivot, x_i - x_0 is exact (Sterbenz), which explains the 8e-8 vs 9e-15 behaviour seen.

### Cited Findings
- Chan, Golub & LeVeque (1983), The American Statistician 37(3):242–247, DOI 10.1080/00031305.1983.10483115 — [PDF](https://www.math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf):
  - **Condition number** (2.1), from Chan & Lewis 1978: kappa = ||x||_2/sqrt(S) = sqrt(1 + N*x̄^2/S), with S = sum (x_i - x̄)^2. For small S, kappa ≈ x̄*sqrt(N/S), i.e. "the mean divided by the standard deviation" (2.2). Relative errors delta in the data give relative change <= kappa*delta in S.
  - **Table 1** (dominant relative-error bounds for S, constants dropped):

    | Algorithm | Bound |
    |---|---|
    | textbook one-pass | N*kappa^2*u |
    | textbook, pairwise summation | kappa^2*u*log N |
    | two-pass | N*u + N^2*kappa^2*u^2 |
    | two-pass, pairwise summation | u*log N + (kappa*u*log N)^2 |
    | corrected two-pass | N*u + N^3*kappa^2*u^3 |
    | updating (Welford/West) | N*kappa*u |
    | pairwise (conjectured) | kappa*u*log N |

  - **Shifted data**, x̃_i = x_i - d: kappã^2 = 1 + N*(x̄ - d)^2/S (3.2). Taking d = x̄ gives kappã = 1. If d is within one standard deviation of the mean, kappã < sqrt(2).
  - **Shift by a data point:** "When min x_i <= d <= max x_i we have (x̄ - d)^2 <= S and so kappã^2 <= 1 + N" (3.4), "easily guaranteed, for example by choosing one of the data points as the shift". For a uniformly random data point, E[kappã^2] = 2 (3.5).
  - **Shift by the computed mean:** fl(x̄) = (1/N) sum x_i(1+eta_i) with |eta_i| <= N*u (3.8–3.9), which gives kappã^2 <= 1 + N^2*kappa^2*u^2 (3.11). "One can easily construct examples where the computed mean does not even lie between min x_i and max x_i … In this case one is better off shifting by any single data point than by the computed mean."
  - **Corrected two-pass** (1.7): S = sum (x_i - x̄)^2 - (1/N)(sum (x_i - x̄))^2. The second term corrects for the rounding error in x̄. It is the recommended method when all data are in memory.
- Welford (1962), "Note on a Method for Calculating Corrected Sums of Squares and Products", Technometrics 4:419–420, is the classic updating method for sums of squares **and products** (cited in the CGL reference list). — [CGL 1983 PDF](https://www.math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf)
- Sterbenz lemma: "In a floating-point number system with subnormal numbers, if x and y are floating-point numbers such that y/2 <= x <= 2y, then x - y is also a floating-point number", so the correctly rounded subtraction is exact. It does **not** hold with flush-to-zero. — [Wikipedia: Sterbenz lemma](https://en.wikipedia.org/wiki/Sterbenz_lemma)
- Marquardt & Snee (1975), quoted: "failure to remove nonessential ill conditioning through the use of standardized predictor variables". The report adds "it is clearly safest to center the independent variables at their means". — [DTIC ADA039547](https://apps.dtic.mil/sti/tr/pdf/ADA039547.pdf) (via search summary)
- Marquardt (1980), JASA 75(369):87–91, "You should standardize the predictor variables in your regression models". Later work describes "non-essential" multicollinearity as the kind caused by the relation with the intercept, removed by centring. — [Salmerón et al., "Diagnosis and quantification of the non-essential collinearity"](https://digibug.ugr.es/handle/10481/89481); [arXiv 1905.12293](https://arxiv.org/pdf/1905.12293)
- Belsley (1984), "Demeaning conditioning diagnostics through centering", The American Statistician 38(2):73–77, with comments and reply. It argues **against** centring for *collinearity diagnostics*, because centring hides intercept-related ill-conditioning that is real for the uncentred parametrisation. Velilla (2018) argues centring may fail to detect sources of ill-conditioning. — [Salmerón et al. 2020 (TAS accepted version)](https://digibug.ugr.es/bitstream/handle/10481/86536/2020_TAS_version_aceptada.pdf?sequence=1); secondary: Wicklin's SAS blog (blogs.sas.com/content/iml/2020/01/29/collinearity-diagnostics-centering.html), which now redirects and was not readable.

### Inferences
- [DERIVED] Why the pivot shift works here:
  - For a narrow box with lx > 0 and ux <= 2*lx, which covers every narrow box away from 0, Sterbenz makes d_i = fl(x_i - x_0) = x_i - x_0 **exactly**.
  - In every other case a single subtraction of two exact floats still has relative error <= u in d_i. Cancellation only hurts when an operand is already inexact, like fl(mean(x)).
  - mean(d) is then computed from numbers of size <= w, so its absolute error is <= ~N*u*w instead of ~N*u*|x̄|.
  - For the 5-point endpoint pivot on an equispaced grid, (x̄ - x_0)^2 = w^2/4 and S = 5w^2/8 per axis, so kappã^2 = 1 + 5*(w^2/4)/(5w^2/8) = 3, i.e. kappã = sqrt(3), whatever |x̄|/w is.
  - With the computed mean as pivot, kappa ≈ |x̄|/sigma_x = 2*sqrt(2)*|x̄|/w enters via (3.11).
- [DERIVED] Leakage mechanism behind the earlier 8e-8 errors:
  - With dx_i = x_i - c, A = sum dx_i*z_ij / sum dx_i^2 is correct only if sum dx_i = 0. In general the numerator is sum (x_i - x̄)*z_ij + n*(x̄ - c)*z̄, so the error in A ≈ (x̄ - c)*z̄/sigma_x^2 with sigma_x^2 = w^2/8.
  - With c = fl(mean), |x̄ - c| ≲ N*u*|x̄|, so |dA| ≲ 8*N*u*|x̄|*|z̄|/w^2. For example x̄ = 2, z̄ = 0.3, w = 1e-4, N = 25 gives ~2.6e-7. That is the same order as the observed 8e-8; the user's actual box parameters were not available to me.
- [DERIVED] Make the cross-product shift-invariant:
  - Use centred z in the numerator: A = sum_i,j (x_i - c)(z_ij - z̄) / [5*(sum_i (x_i - c)^2 - 5*m_c^2)], with m_c = mean(x_i - c). This is the "corrected two-pass" denominator (CGL 1.7).
  - sum (x_i - c)(z_ij - z̄) is identical for every c because sum (z_ij - z̄) = 0, so a wrong pivot only enters at second order. This is the corrected sums-of-products idea in Welford's title.
  - Recommended combination: pivot shift, then subtract mean(d), then use (z - z̄).
- [DERIVED] Recover x̄ as x_0 + mean(d), not as fl(mean(x)). Then C = z̄ - A*x̄ - B*ȳ has the inherent conditioning of extrapolating the intercept to x = 0: dC ≈ x̄*dA + ȳ*dB. gelsd has the same issue. Errors in C and A are correlated and largely cancel in C + A*x within the box. **Test fitted values at box corners, or the centred parameters (z̄, A, B), as well as raw C.**
- [DERIVED] Floor on achievable accuracy. z_ij = fl(s'(x_i)*y_j) carries relative rounding ~u. For the narrow coordinate, B is sensitive to that noise at the level ~u*|z|/(sigma_y*|B|). That is the 1/w law of the closed form in the Q2 table. No algorithm that starts from rounded z beats it, so the closed form is essentially at the data-conditioning floor.
- [DERIVED] Using the separable structure gives B = mean_i s'(x_i) and A = ȳ * slope_x(s') directly from 5 values instead of 25. This avoids that noise floor. It changes results at the eps level relative to a z-based reference, so it is not a bit-exact gelsd replacement.

### Gaps
- I did not obtain Higham's "Accuracy and Stability of Numerical Algorithms" text (sample variance §1.9; least squares ch. 20) to cite exact theorem numbers.
- I did not read Belsley (1984) directly. The characterisation above comes from secondary sources.

---

## Q4. Singular values of X = [1, x, y] on a Cartesian grid; analytic 3x3 eigenvalues; an rcond-equivalent rule; cutoff vs box width

### Takeaway
[DERIVED] X = Q·(5K), where Q has orthonormal columns and K = [[1, x̄, ȳ], [0, σx, 0], [0, 0, σy]] (σ² = (1/5)·Σ_i (x_i − x̄)²). So sing(X) = 5·sing(K) exactly. This holds for any Cartesian product grid; equal spacing is not needed. gelsd drops s3 roughly when w_y ≤ 2√2·rcond·√(1+ȳ²)·√(1+x̄²+ȳ²+σx²), and symmetrically for x. The cutoff grows like |x̄|²: in float64 it is ≈1.6e-12 at x̄ = 10 and ≈1.6e-8 at x̄ = 1e3; in float32 it is ≈8e-4 at x̄ = 10 and ≈8e-2 at x̄ = 100. Cardano applied to an explicitly formed XᵀX can't resolve this cutoff because it squares the condition number. Use the 3x3 K, or symmetric functions computed from K, instead.

### Cited Findings
- Smith (1961), "Eigenvalues of a symmetric 3 × 3 matrix", Communications of the ACM 4(4):168, DOI 10.1145/355578.366316. It derived an explicit (trigonometric) solution for computing many 3x3 symmetric eigenproblems. — [CACM](https://cacm.acm.org/research/eigenvalues-of-a-symmetric-3-x-3-matrix)
- Kopp (2008), "Efficient numerical diagonalization of hermitian 3x3 matrices", Int. J. Mod. Phys. C 19:523–548, arXiv physics/0610206:
  - "The analytical algorithm outperforms the others by more than a factor of 2" but "becomes inaccurate or may even fail completely if the matrix entries differ greatly in magnitude".
  - "Jacobi is the most accurate, but also the slowest method, while QL and Cuppen are good general purpose algorithms".
  - A hybrid "falls back to QL" when the analytic result may be inaccurate. — [arXiv physics/0610206](https://arxiv.org/abs/physics/0610206)
- Habera & Zilian (2025/2026, Numerical Algorithms), arXiv 2511.00292:
  - The trigonometric (Cardano/Viète) formulas "are numerically unstable for matrices with repeated eigenvalues".
  - They propose a stable closed form based on invariants (trace, deviatoric invariants, discriminant) with forward error bounds, ~10x faster than LAPACK "while maintaining comparable accuracy". — [arXiv 2511.00292](https://arxiv.org/abs/2511.00292)
- Forming normal equations squares the condition number. LS sensitivity carries a kappa^2 * tan(theta) term even for stable methods. — [Cornell CS 6210 notes](https://www.cs.cornell.edu/courses/cs6210/2025fa/lec/2025-10-01.html) (search summary), [Grcar arXiv 1003.2733](https://arxiv.org/pdf/1003.2733)

### Inferences
- [DERIVED] Exact factorisation and singular values:
  - x = x̄·1 + dx and y = ȳ·1 + dy. On a Cartesian grid the columns 1, dx⊗1 and 1⊗dy are mutually orthogonal. (dx⊗1)ᵀ(1⊗dy) = (Σ dx_i)(Σ dy_j) = 0 needs only Σ dx_i = Σ dy_j = 0, not equal spacing. So `torch.linspace` rounding does not break the decoupling, provided x̄ and ȳ are the exact means of the rounded points.
  - Therefore X = [1, dx, dy]·T with T = [[1, x̄, ȳ], [0,1,0], [0,0,1]], and [1, dx, dy] = Q·diag(5, 5σx, 5σy). Hence sing(X) = 5·sing([[1, x̄, ȳ], [0, σx, 0], [0, 0, σy]]).
  - For 5 equispaced points across width w, σ² = w²/8, i.e. σ = w/(2√2).
- [DERIVED] Symmetric functions of λ_i = s_i², all sums of non-negative terms and so computable to high relative accuracy:
  - e1 = Σλ = 25(1 + x̄² + ȳ² + σx² + σy²)
  - e2 = 625[σx²(1+ȳ²) + σy²(1+x̄²) + σx²σy²] (Cauchy–Binet: sum of squared 2x2 minors of 5K)
  - e3 = Πλ = 15625·σx²σy², i.e. s1·s2·s3 = 125·σx·σy
- [DERIVED] Stable route to s3 without an SVD:
  1. λ1 from the trigonometric formula. The largest root is well-conditioned, with relative error O(u).
  2. q = λ2·λ3 = e3/λ1 and p = λ2 + λ3 = (e2 − q)/λ1. Since q ≤ e2/2 this loses at most about 1 bit.
  3. λ2 = (p + √(p² − 4q))/2 and λ3 = q/λ2.
  - Relative accuracy degrades only when λ2 ≈ λ3 (both coordinates nearly degenerate by similar amounts), consistent with the repeated-eigenvalue instability in Habera & Zilian.
  - The simpler, robust option is batched `torch.linalg.svdvals` on the (n,3,3) tensor 5K. It runs on CUDA, and its absolute error is O(u·s1), which is enough outside the ~2% fuzzy band.
- [DERIVED] Why Cardano on a formed XᵀX fails. Absolute errors in its entries are ~u·λ1 = u·s1². So λ3 = s3² is resolved only down to ~u·s1², i.e. s3 only down to ~√u·s1 ≈ 1e-8·s1. gelsd's cutoff is 5.55e-15·s1, about six orders of magnitude below what this route can resolve.
- [DERIVED] Approximate cutoff for a nearly degenerate y-axis (σy small): s3 ≈ 5σy/√(1+ȳ²) and s1² ≤ 25·T with T = 1 + x̄² + ȳ² + σx². So gelsd treats the box as rank 2 when
  w_y ≲ 2√2 · rcond · √(1+ȳ²) · s1/5 ≈ 2√2 · rcond · √(1+ȳ²) · √T.
  The x-axis case swaps x and y. For x̄ ≫ 1 and ȳ = O(1) this is ≈ 2√2·rcond·(1+x̄²). The rank decision therefore depends on the box's location, not only its relative width. It also changes under a shift of origin, because gelsd does not centre.
- [EMPIRICAL] gelsd's rank transition (log sweep, 2000 points) against the exact-K prediction and the crude formula:

  | dtype | axis swept | centre x̄ | other width | y0 | gelsd: largest rank-def. w | K-SVD prediction | crude formula |
  |---|---|---|---|---|---|---|---|
  | f64 | y | 0.3 | 1 | 0.5 | 2.02e-14 | 2.02e-14 | 2.13e-14 |
  | f64 | y | 2 | 0.5 | 3 | 1.84e-13 | 1.84e-13 | 1.86e-13 |
  | f64 | x | 10 | 1 | 1 | 1.58e-12 | 1.58e-12 | 1.60e-12 |
  | f64 | x | 1e3 | 1 | 1 | 1.56e-8 | 1.56e-8 | 1.57e-8 |
  | f64 | x | 1e4 | 1 | 0.2 | 1.54e-6 | 1.54e-6 | 1.57e-6 |
  | f32 | y | 0.3 | 1 | 0.5 | 1.08e-5 | 1.08e-5 | 1.14e-5 |
  | f32 | x | 10 | 1 | 1 | 8.44e-4 | 8.44e-4 | 8.56e-4 |
  | f32 | x | 100 | 1 | 1 | 8.29e-2 | 8.29e-2 | 8.43e-2 |

  The rule "rank-deficient iff s3(5K) ≤ rcond·s1(5K)" matched gelsd's transition to the grid resolution (~2.5%) in all 8 cases. The crude formula overestimates by 1–6%.
- [DERIVED] Recommended decision logic for a gelsd-faithful closed form:
  1. Exact-endpoint degeneracy (lx == ux, ly == uy) gives σ = 0 exactly. gelsd always agrees, with a 25x margin per [EMPIRICAL].
  2. Otherwise compute K from pivot-centred quantities and apply s3 ≤ 25·eps·s1. If it fires, use the rank-2 truncated solution. Use the rank-1 solution if s2 also falls below the cutoff.
  - In the truncated, non-exact case the truncated solution is no longer the simple x0/y0 formula: v3 is not exactly (−ȳ, 0, 1)/√(1+ȳ²) when σy > 0. Either project out v3 from the 3x3 SVD of 5K (θ_trunc = θ_full − (v3ᵀθ_full)·v3, valid because the X = Q·5K factorisation is exact) or accept an O(σy) mismatch.
  - Whether to mimic gelsd at all is a design decision. Inside the band gelsd's answer is a regularised one, and just above it gelsd's answer is numerically meaningless (Q2 table).

### Gaps
- I did not obtain the full Kopp paper's quantitative accuracy plots for small eigenvalues. The abstract talks about entries differing in magnitude, not small eigenvalues specifically.
- I found no published source giving singular values of a Cartesian-grid regression design in this closed form. The factorisation above is my own derivation, verified numerically.

---

## Q5. Floating-point hazards of the unified closed-form formula (float64 first, float32 second)

### Takeaway
1 + x0² losing the 1 is harmless, because the relative error stays ≤ u. The real hazards are:
- overflow of x0² (|x0| > ~1.34e154 in f64, > ~1.84e19 in f32), which silently returns C = A = 0;
- k·x0 overflow;
- NaN/Inf from the unused branch, which is safe in a `torch.where` forward pass but poisons arithmetic masking and autograd;
- subnormal results flushed to zero under FTZ;
- the large float32 rank band of Q4;
- cancellation in 1 − σ(x) when forming the targets.

### Cited Findings
- `torch.where` still backpropagates through the branch it did not select: "torch.where() backpropagates a gradient of 0 * nan = nan (or maybe 0 * inf = nan) through the 'branch not taken'". The workaround is to make the unselected branch finite in the forward pass (for example, a safe denominator). — [PyTorch forums: incorrect gradient with torch.where and NaNs](https://discuss.pytorch.org/t/incorrect-gradient-calculation-with-torch-where-and-nans/185367); [MaskedTensor NaN-grad note](https://docs.pytorch.org/maskedtensor/main/notebooks/nan_grad.html)
- torch.finfo: `smallest_normal` is "The smallest positive normal number"; "there are smaller subnormal numbers"; `max` is "The largest representable number". — [torch.finfo docs](https://docs.pytorch.org/docs/2.14/type_info.html)
- Sterbenz exactness, which the pivot shift relies on, fails under flush-to-zero. — [Wikipedia: Sterbenz lemma](https://en.wikipedia.org/wiki/Sterbenz_lemma)
- DGELSD and DGELSS rescale A and B only when max|a_ij| is outside [SFMIN/EPS, EPS/SFMIN] (~1e-292 … 1e292 in f64). LAPACK is therefore safe across almost the whole range, and a closed form needs its own scaling to match it at extremes. — [dgelsd.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelsd.f)

### Inferences
- [DERIVED + EMPIRICAL] 1 + x0²:
  - In f64 the 1 is lost when x0² > 2^53 (|x0| ≳ 9.5e7). In f32 it is lost when |x0| ≥ 4096; I checked fl32(1 + 4096²) == 4096².
  - Both are harmless: C = k/(1+x0²) and A = k·x0/(1+x0²) keep relative error O(u). The test showed x0 = 1e8 → C = 1e-16, A = 1e-8, which is correct.
- [EMPIRICAL] Overflow. Naive f64 at x0 = 1e155 gave C = A = 0, but the correct values are C ≈ 1e-310 (subnormal) and A ≈ 1e-155.
  - Fix [DERIVED]: for |x0| > 1, set t = 1/x0, then A = k·t/(1+t²) and C = A·t. This gave A = 1e-155, C = 1e-310, and at x0 = 1e200 it gave A = 1e-200, C = 0 (correct underflow).
  - Two-variable case: scale by m = max(1, |x0|, |y0|) before squaring. θ = mean(z)·(1, x0, y0)/(1+x0²+y0²) = (mean(z)/m)·(1/m, x0/m, y0/m)/((1/m)²+(x0/m)²+(y0/m)²).
  - Also guard k·x0 overflow (|k·x0| > ~1.8e308) by dividing before multiplying.
  - Thresholds: √DBL_MAX ≈ 1.34e154 and √FLT_MAX ≈ 1.84e19.
- [DERIVED] Underflow. x0² underflows for |x0| < ~1.5e-154 (f64) and ~1.1e-19 (f32), which is harmless because 1 + x0² = 1. Results such as C ≈ 1e-310 are subnormal. Under FTZ/DAZ (CUDA `--use_fast_math`, `torch.set_flush_denormal(True)` on CPU) they become 0. That is negligible in absolute terms but breaks exact equality tests against LAPACK.
- [DERIVED] Cancellation in C = z̄ − A·x̄ − B·ȳ. This is inherent conditioning (|x̄|·|dA|), not an algorithm defect; see Q3. Evaluate errors on fitted values, or reformulate downstream to use (z̄, A, B) about the box centre.
- [DERIVED] Division by zero and NaN propagation:
  - In the non-degenerate branch, Σdx² = 0 for degenerate boxes, so A = 0/0 = NaN there.
  - `torch.where(deg, a, b)` returns the right value in the forward pass.
  - Arithmetic blending (`mask*a + (1-mask)*b`) yields NaN, because 0·NaN = NaN in IEEE arithmetic.
  - Any autograd through `where` yields NaN gradients (per the forum thread above).
  - Use a safe denominator: `den = torch.where(deg, 1, den)`.
  - Also handle NaN/Inf in the inputs. gelsd's DLANGE scaling sees NaN as the max norm, and LAPACK behaviour on NaN input is unspecified; see Gaps.
- [DERIVED] Degeneracy that is not exact. Endpoints a few ulps apart give rounded linspace points with σ > 0 but tiny. The exact-endpoint test calls these full-rank, while gelsd calls them rank-deficient whenever w < the Q4 cutoff (f64 cutoff ≈ 1.6e-14·(1+x̄²)). Results then differ by O(1) (Q2). The f32 band is much wider: about 8.4e-6·(1+x̄²) in absolute width, which is ~8e-4 at x̄ = 10. If f32 gelsd is ever the reference, many real boxes fall into it.
- [DERIVED] Forming the targets. s'(x) = σ(x)·(1 − σ(x)) loses all relative accuracy once σ(x) rounds to 1 (x ≳ 37 in f64, ≳ 17 in f32), returning 0 instead of ≈ e^(−x). σ(x)·σ(−x) avoids this. Both paths share z, so this does not cause gelsd-vs-closed-form disagreement, but it matters for absolute correctness. `exp(−t)` overflows for t < −709 (f64), which gives σ = 0, s' = 0, still finite, plus a numpy RuntimeWarning.
- [DERIVED] Hazard tests to run, as a checklist:
  1. Exact-degenerate x, y, and both, including a degenerate axis at x0 or y0 = 0, ±1e8, ±1e154, ±1e155, 1e-160.
  2. Widths swept log-uniformly from 1 ulp to 1 around |centre| ∈ {0, 1, 10, 1e3, 1e4}, comparing gelsd rank against s3(5K) ≤ 25·eps·s1(5K).
  3. Boxes straddling 0, where Sterbenz does not apply, and boxes with ux > 2·lx.
  4. Identical-after-rounding linspace points (endpoints 1–3 ulps apart).
  5. Signed zeros, Inf/NaN inputs.
  6. float32 versions of all of the above, with rcond = 25·2^-23.
  7. NaN-free forward and backward passes under `torch.where`.
  8. FTZ on vs off.
  9. Fitted-value error at the 4 box corners in addition to coefficient error.
  10. An exact rational or extended-precision reference for boxes in the gelsd "garbage band" (Q2), where gelsd itself is not a valid oracle.

### Gaps
- I did not check what torch or LAPACK actually return for NaN or Inf inputs to gelsd (error, NaN output, or a hang). This needs a direct test.
- I did not confirm whether PyTorch's CUDA kernels flush subnormals by default without fast-math.

---

## Q6. Orthogonal / factorial designs: standard results that make the decoupled closed form exact

### Takeaway
A full factorial (Cartesian-product) design with centred coding has a diagonal XᵀX. The least-squares estimates then decouple: each coefficient is its own one-dimensional projection, uncorrelated with the others and unchanged by adding or removing the other orthogonal terms. That is exactly the closed form A = Σdx·z/Σdx², B = Σdy·z/Σdy², C' = z̄.

### Cited Findings
- In factorial designs, when XᵀX is diagonal, "the least squares estimates of the factorial effects are uncorrelated and can be calculated easily". — [Berkeley handout (Cheng)](https://www.stat.berkeley.edu/~cheng/Handout13.pdf) (via search summary)
- Coefficients are estimated with minimum variance iff they are uncorrelated, which requires (X′X)⁻¹, equivalently X′X, to be diagonal. Orthogonal designs exist for first-degree models but not when the model degree is greater than one. — [Krennrich, DoE & Optimization, ch. 4](https://bookdown.org/gerhard_krennrich/doe_and_optimization/doe2.html) (via search summary)
- When design blocks are mutually orthogonal, each LS estimate does not depend on whether the other terms are in the model. — [JHSPH 140.752 notes, ch. 8](https://www.biostat.jhsph.edu/~iruczins/teaching/140.752/notes/ch8.pdf) (via search summary)

### Inferences
- [DERIVED] For the first-degree model on the 5x5 product grid, with centred columns:
  - 1ᵀ(dx⊗1) = 5·Σdx_i = 0, 1ᵀ(1⊗dy) = 0, and (dx⊗1)ᵀ(1⊗dy) = Σdx_i·Σdy_j = 0. So XcᵀXc = diag(25, 5Σdx_i², 5Σdy_j²).
  - The closed form is the exact LS solution, not an approximation, for any product grid, including non-uniform `torch.linspace` rounding.
  - This holds provided the centring makes Σdx = 0 and Σdy = 0 exactly enough (Q3).
- [DERIVED] The "orthogonal designs don't exist for degree > 1" remark doesn't matter here (the model is planar). It does warn that a future switch to a quadratic or bilinear surrogate would lose exact decoupling: the x², y² columns are not orthogonal to 1. The xy column, however, is orthogonal to 1, x and y on a centred product grid, so a bilinear term would still decouple.
- [DERIVED] Because z = s'(x)·y is separable, the plane fit equals s̄·y + a·ȳ·(x − x̄). The neglected part is (s'(x) − s̄ − a(x−x̄))·y + a(x−x̄)(y−ȳ). This gives the exact residual structure, useful for error bounds on the linearisation.

### Gaps
- I did not obtain a textbook page (Box, Hunter & Hunter; Montgomery) with the formal statement. The three sources above are lecture notes and handouts, read via search summaries rather than full fetches.
