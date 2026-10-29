# Plane choice and error-term soundness in NN verification / abstract interpretation (Pasado's σ'(x)·y transformer and related work)

Scope: how Pasado and related work pick a linear (planar) approximation of a nonlinear term such as σ'(x)·y over a box, and how they argue that the error term is sound. Status as of 25 Oct 2026. "Cited" means the source says it. "Inference" means I concluded it. Numbers marked **[own experiment]** come from a small NumPy/SciPy script I ran this session: `/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/plane_check.py`. They are not from any paper.

Sources I read in full text: the Pasado OOPSLA'23 paper (local `papers/OOPSLA2023Final.pdf`) and its appendix, the POPL'22 DeepJ paper, Laurel's 2024 PhD thesis, Phocus (OOPSLA'25), LinSyn, Prover (Ryou et al. CAV'21), Shi et al. NeurIPS'22, auto_LiRPA, Shi et al. ICLR'20, CROWN, DeepZ (NeurIPS'18), Jia & Rinard, Rump & Kashiwagi, Zhang et al. ASE'22, König et al. 2024 and ProveSound (OOPSLA'25). I also read the upstream Pasado code (local `Pasado-original/forward_mode_tensorized_src/precise_transformer.py`, branch `fix-zonotope-neg-and-boundary`; line numbers below refer to that copy).

---

## Q1. What Pasado (OOPSLA'23 + appendix), its POPL'22 companion and the follow-ups say about the planar approximation, the error term, the soundness theorem and floating point

### Takeaway
Pasado fits the plane by "linear regression" on a uniform grid only as a precision heuristic. The paper gives no reason for choosing least squares over other fits and never mentions 5×5; the 5×5 grid exists only in the code. Soundness comes entirely from the separately computed max-deviation D. For the chain-rule transformer this is Theorem 4.1, proved with a Hessian/boundary argument that needs A ≠ 0. The paper says the δ perturbation is only applied when regression returns A = 0, and that δ < 1e-9. The code instead always adds N(0, 1e-4) noise. Pasado states outright that its implementation "assumes ideal real arithmetic and is thus not floating-point sound". The OOPSLA'25 follow-up (Phocus) repeats the same regression-plus-max-deviation recipe and the same floating-point disclaimer.

### Cited Findings
- **How the plane is chosen.** "Pasado's abstract transformer synthesis involves a combination of linear regression at uniformly spaced points and solving a nonlinear optimization problem to ensure soundness. For tractability, we limit the number of sampled grid points used in the linear regression (which trades off precision)." (§4, p. 291:11). The paper gives no other justification for least squares (versus minimax, LP or tangent) and does not state a grid size. — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Plane versus soundness.** "because the hyperplane is only a linear approximation it is unsound, hence we must also account for the deviation between this hyperplane and the original function to maintain soundness. Thus we solve for the maximum deviation …" (§1). Algorithm 1 is: GridSample → pts = {f'(x)·y} → (A,B,C) ← LinearRegression → "if A = 0 then A ← A + δ" → D ← max |f'(x)·y − (Ax+By+C)| over the box → return Ax̂ + Bŷ + C + Dε_new, together with optimal interval bounds [l,u] (p. 291:13). — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Fig. 4 caption** (σ(x)(1−σ(x))·y on [−1,1]²): the yellow plane comes from regression on grid points, and D is the maximum difference. "The enclosing bounds are parallel for the Zonotope domain and here the maximum difference D occurs at a corner point." — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Why A ≠ 0.** "For the correctness of our proof which is subsequently shown in Theorem 4.1, it is a technical requirement that A ≠ 0. If linear regression obtains A = 0, we perturb A by a small quantity, δ < 10⁻⁹." The stationary points are all x* with f''(x*) = A/l_y and all x** with f''(x**) = A/u_y. D is the max over {l_x,u_x}×{l_y,u_y} ∪ {(x*,l_y),(x**,u_y)}. — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Root-solver requirement (§3.2).** "we will also require a guaranteed root solver for the second derivative of each nonlinear function f … one may use a verified root finding technique like Bisection or a verified Newton's method, however for many of the functions such that f''⁻¹ has an analytical formula, we can use the analytical formula." — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Theorem 4.1 (chain rule sound).** The proof uses the Hessian of f'(x)y − (Ax+By+C), [[f'''(x)y, f''(x)],[f''(x), 0]], whose determinant is −f''(x)² ≤ 0. Where f''(x) ≠ 0 the Hessian is indefinite, so there is no interior extremum. Where f''(x) = 0 the gradient's x-component is −A ≠ 0. So extrema lie on the four boundary lines. On x = l_x or x = u_x the function is linear in y, so the corners suffice. On y = l_y or y = u_y the problem is 1-D and is solved with the root solver. The absolute value is handled because the Hessian of −E is also indefinite (Lemma A.1). — [Pasado OOPSLA'23 §4.4](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf); [Appendix A.1](https://jsl1994.github.io/papers/OOPSLA2023_appendix.pdf)
- **Origin of the proof idea.** "While inspired by Ryou et al. [2021], our proof technique is more general as we can handle any f satisfying the properties of Section 3.2." — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **What the appendix covers.** Lemma A.1; chain rule for intervals (the edge case f'(x*) = f''(x*) = 0, which "will never happen" for exp, log, σ, tanh); product rule (check the 2⁴ corners); and the long quotient-rule case analysis. It contains nothing on regression, grid size, noise on A or floating point. (I grepped for regression, grid, float and round and found no hits.) — [Pasado appendix](https://jsl1994.github.io/papers/OOPSLA2023_appendix.pdf)
- **Precision (Theorem 4.4).** The interval bounds from Eqs. 4–5 are optimal for the interval domain, "Since Pasado solves these optimization problems exactly". Theorem 4.5 says the same transformers carry over to DeepPoly by using Ax+By+(C∓D) as the lower and upper planes. — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Floating point.** "Pasado's implementation assumes ideal real arithmetic and is thus not floating-point sound (though floating-point sound versions of the operations exist [Miné 2004])." (§5.1). — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Cost argument.** The composite transformer adds one noise symbol instead of three. "It is known that having fewer noise symbols can lead to computational savings [Turner 2020], and this reduction helps us offset some of the cost of synthesizing transformers." — [Pasado OOPSLA'23 §2](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Code: grid and solver.** `get_linspace(..., STEPS=5)` builds the 5×5 grid (l. 7), and `torch.linalg.lstsq(xplusone, zs, driver='gelsd')` solves it (l. 67). — [uiuc-arc/Pasado](https://github.com/uiuc-arc/Pasado) (local copy `Pasado-original/forward_mode_tensorized_src/precise_transformer.py`)
- **Code: noise on A.** `ABCs = [(x[0] + np.random.normal(0, .0001), x[1], x[2]) ...]` adds noise to A unconditionally, followed by `assert A>0 or A<0` (l. 426–427). The softplus path (l. 467) asserts A ≠ 0 but adds no noise. — [uiuc-arc/Pasado](https://github.com/uiuc-arc/Pasado)
- **Code: root path.** The cubic s(1−s)(1−2s) = A/y is solved with Cardano in `torch.complex64`, using `cuberoot(x) = x ** 0.33333`, not the exact 1/3 (l. 82–104). Non-real roots are dropped with `torch.isreal`. A/y outside [−1/(6√3), 1/(6√3)] is discarded, and y = 0 gives inf, which is mapped to NaN, so no stationary point is used (l. 235–255). — [uiuc-arc/Pasado](https://github.com/uiuc-arc/Pasado)
- **POPL'22 companion (DeepJ).** This is an interval-domain dual-number abstraction of the Clarke Jacobian. It has no planar or regression transformer. It "optionally offers floating-point soundness [Miné 2004], i.e., its result can capture all possible outputs under different rounding modes and under different orders of computations". The sound-rounding version changes the Lipschitz constants by less than 1e-10, but "ensuring floating-point soundness adds up to 4x runtime overhead" (2.8–4.1× in the intro). — [Laurel et al. POPL'22](https://jsl1994.github.io/papers/POPL2022Final.pdf)
- **Follow-ups 2024–2026 by the same author.**
  - Laurel's PhD thesis (UIUC, Dec 2024) reproduces the Pasado chapter verbatim. It again says "δ < 10⁻⁹", and for the product and quotient rules an "Adjust" step that adds "small perturbations (δ < 10⁻⁹)" until the conditions hold. — [thesis](https://jsl1994.github.io/papers/jacob-laurel-phd-thesis.pdf)
  - Phocus, "Synthesizing Sound and Precise Abstract Transformers for Nonlinear Hyperbolic PDE Solvers" (Laurel, Laguna, Hückelheim, OOPSLA'25, Distinguished Artifact). It reuses "Grid Sampling and Linear Regression" and then an "Optimization Problem" for the max deviation D, and says Pasado "is closest in spirit … uses different proof techniques". Its implementation "assumes ideal real arithmetic and thus is not floating-point sound though there exist several techniques to add floating-point soundness at the cost of additional runtime [Miné 2004]". — [Phocus OOPSLA'25](https://jsl1994.github.io/papers/OOPSLA2025.pdf)
  - Other items listed: a LAFI'24 position talk "Abstract Interpretation for AD"; a 2025 Foundations and Trends monograph "Safety and Trust in AI with Abstract Interpretation"; AURA (SAS'25). Laurel is now an Assistant Professor at Georgia Tech. — [Laurel homepage](https://jsl1994.github.io/)

### Inferences
- **The paper does not explain least squares or 5×5.** Both are engineering choices. The paper only says the number of grid points trades precision for cost. 5×5 is visible only in the code (`STEPS=5`).
- **The δ in the paper and the noise in the code do not match.**
  - The paper says δ < 1e-9 is applied only when A = 0.
  - The code always adds N(0, 1e-4) to A. A Gaussian is not bounded by 1e-9, and the noise is applied even when A ≠ 0.
  - Soundness is unaffected, because D is computed with the noisy A that is also used in the zonotope.
  - Precision is affected (see Q5 [own experiment]).
  - Bit-reproducibility is affected too, because the noise comes from an unseeded global NumPy RNG.
- **The code does not meet the paper's root-solver requirement.** The paper asks for a "guaranteed/verified root solver". The code uses a complex64 Cardano formula with the exponent 0.33333, which does not meet that bar. In a NumPy emulation of that path, the stationary points in s = σ(x) were off by up to about 9e-6. Exact 1/3 in complex128 gives errors around 3e-14. **[own experiment]**
- **Not floating-point sound, by the authors' own statement.** "Theorem 4.1 is sound" is a real-arithmetic statement. That is accepted practice at OOPSLA: both Pasado and Phocus carry the same disclaimer.

### Gaps
- I found no arXiv or extended version of Pasado beyond the camera-ready plus the appendix and the thesis chapter. The thesis text I grepped adds nothing about grid size, least-squares rationale or floating point.
- I did not read the LAFI'24 abstract or the 2025 FnT monograph. I don't know whether either discusses plane choice.
- I found no Laurel-group publication from 2024–2026 that revisits Pasado's σ'·y transformer specifically: no GPU version, no minimax or LP plane, no floating-point-sound version.

---

## Q2. Is "E is affine in y, so max |E| lies on the y-edges, so any plane is sound" standard and airtight?

### Takeaway
Yes, in real arithmetic. For fixed x, E(x,·) is affine in y, so |E(x,·)| is convex and attains its maximum at y ∈ {l_y, u_y}. That argument is simpler than Pasado's Hessian proof and does not need A ≠ 0. Splitting the work into "the plane controls precision, D controls soundness" is the standard pattern in this literature (Prover, LinSyn, DeepG, affine arithmetic). In practice it is airtight only if four conditions hold:
1. All stationary points on both y-edges are found.
2. |E| is maximized, not E. This is exactly the upstream bug that was fixed.
3. The (A,B,C) used for D is bit-identical to the one written into the zonotope.
4. Rounding in evaluating E, and error in the roots, are either accounted for or explicitly disclaimed.

### Cited Findings
- **Affine arithmetic: any approximation with a bounded error is sound.** If [[p,q,Δ]] "represents" f on X, meaning |px+q−f(x)| ≤ Δ for all x ∈ X, then F(C) = ⟨pc+q; pγ, Δ⟩ is a sound affine form for f(C). "This general principle applies to Min-Range and Chebyshev approximations and to general continuous or differentiable nonlinear functions." The choice of (p,q) is about tightness: Chebyshev minimizes the max error, while Min-Range minimizes the range. — [Rump & Kashiwagi, NOLTA 2015](https://www.tuhh.de/ti3/paper/rump/RuKas14.pdf)
- **Prover (Ryou et al. CAV'21), the same pattern for σ(x)·y.** An LP on sampled points gives a plane that "may not be sound as the sampled points do not fully cover the continuous input domain". "To address this, we shift the plane … by an offset … equal to the maximum violation … based on Fermat's theorem."
  - For G(x,y) = σ(x)·y − (Ax+By+C), they check three cases:
    - x fixed: G is monotone in y.
    - y ∈ {l_y, u_y}: solve s(1−s) = A/y.
    - interior: the Hessian determinant is −(σ(x)(1−σ(x)))² < 0, so there is no interior extremum.
  - This is the same structure Pasado uses for σ'(x)·y. — [Ryou et al. 2021](https://arxiv.org/abs/2005.13300)
- **Prover's comparison with DeepG.** "While the method of [Balunović et al. 2019] also performs offset calculation for obtaining sound bounds, they … [use] expensive branch and bound for calculating the offset. In contrast, we … [obtain] a closed-form formula for the offset yielding an exact solution." — [Ryou et al. 2021](https://arxiv.org/abs/2005.13300)
- **LinSyn uses the same split.** Its LP stage aims for tightness "while making a best effort for soundness". It then "verifies the soundness (and adjusts the bounds if necessary)" with dReal. Correctness "is guaranteed if the v_l returned by Algorithm 1 is a true lower bound". — [LinSyn](https://arxiv.org/abs/2201.13351)
- **Pasado's own Theorem 4.1** relies on A ≠ 0 only for the Hessian route. The edges x = const are handled by "the optimal value of a linear function will occur at the end points". — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Unsound transformers do occur and are worth checking.** ProveSound (OOPSLA'25) automatically verifies DNN-certifier transformers written in the ConstraintFlow DSL and detects bugs injected into them. Its motivation: "Unsoundness in the mathematical logic of these certifiers can lead to incorrect results." — [Singh et al. 2025](https://arxiv.org/abs/2504.04542)

### Inferences
- **Proof sketch without A ≠ 0.**
  1. For every x, y ↦ E(x,y) = σ'(x)y − Ax − By − C is affine, so max over y of |E| is reached at an endpoint. Hence max over the box of |E| = max over e ∈ {l_y, u_y} of max over x ∈ [l_x, u_x] of |E(x, e)|.
  2. Each 1-D function φ_e(x) = eσ'(x) − Ax − (Be + C) is C^∞, so max |φ_e| is reached at x ∈ {l_x, u_x} or at a root of φ_e'(x) = eσ''(x) − A = 0.
  3. If e = 0, φ_e is affine and the endpoints suffice.
  4. If e ≠ 0, solve σ''(x) = A/e. With s = σ(x) ∈ (0,1) this is 2s³ − 3s² + s − A/e = 0. There are at most 2 roots in (0,1), and none when |A/e| > 1/(6√3).
  5. Nothing in steps 1–4 uses any property of (A,B,C), so every plane is sound. So the A-noise (or δ) has no soundness role under this argument. It could be dropped, but doing so changes results relative to upstream.
- **Tangency is harmless.** At |A/e| = 1/(6√3) the cubic has a double root. φ_e' does not change sign there, so it is not an extremum. Missing it, or filtering it out by rounding, costs nothing.
- **Where the argument can break in an implementation.**
  - (i) A missed root: for example a numerically non-real root dropped by `isreal`, or a root just outside [l_x, u_x] because of rounding. In my NumPy emulation no roots were dropped over 2001 values of c.
  - (ii) Inaccurate roots: E is evaluated near, not at, the stationary point. The underestimate is second order in the root error, but it is not zero.
  - (iii) D computed with different coefficients than those emitted, for example D from float64 coefficients and the zonotope from float32-cast coefficients.
  - (iv) The box used for D does not enclose the concretization. Pasado uses Range(), the intersection of the zonotope box and the interval, and that must remain sound.
  - (v) Plain floating-point evaluation of g and of the plane (see Q6).
- **Effect of the root accuracy [own experiment]**, using a NumPy emulation of the upstream complex64 / 0.33333 path against exact float64 roots:
  - LS planes: D was underestimated by at most 1.1e-8 relative over 300 random boxes.
  - Non-LS planes (random, tangent, minimax, noisy-A): up to 3.7% relative.
  - So if the GPU port changes the plane rule, root accuracy matters more than before.
  - Caveat: this is NumPy, not torch's complex pow. It should be re-checked against the real torch code.
- **Real-arithmetic check [own experiment].** I tested 300 random boxes × 5 plane rules (LS, LS + N(0,1e-4), minimax, center tangent, random). Corners plus exact cubic roots (np.roots, float64) never gave a smaller D than a dense scan of 41 y-lines × 4001 x-points. The largest relative shortfall was 0.0. This is consistent with the y-edge argument, but it is evidence, not a proof.

### Gaps
- I found no paper that states the "|E| is convex in y, so only the y-edges matter, so A ≠ 0 is unnecessary" argument for Pasado specifically. Prover uses the closely related "monotone in y on the x-edges" argument. The A ≠ 0-free version is my own observation.
- I did not run the real upstream torch code. The root-accuracy numbers above come from an emulation.

---

## Q3. LinSyn (Paulsen & Wang, TACAS'22): how it picks the bound, how it makes it sound, and what it costs

### Takeaway
LinSyn chooses each linear bound by solving an LP over uniformly sampled points. The LP minimizes the volume between the bound and the function, subject to the bound lying on the correct side at every sample. It then makes the bound sound by shifting it by a certified max violation: L-BFGS-B estimates the violation, and the dReal SMT solver, which uses interval constraint propagation with branch-and-bound, verifies it. The shift picks up small fixed margins (2δ with δ = 1e-7, plus 1e-6). The authors report under 1/100 s per pair of bounds, and whole networks at 796–1163 s against 15–1217 s for auto_LiRPA.

### Cited Findings
- **Candidate bounds.** "we draw inspiration from prior work [Ryou et al. 2021; Balunović et al. 2019] that leverages sampling to estimate the curvature of a particular function, and then uses a linear programming (LP) solver to compute a plane". "We generate S by sampling uniformly-spaced points over the input intervals". The objective (their Eq. 3) is the integral (volume) under the upper bound. — [LinSyn](https://arxiv.org/abs/2201.13351)
- **Making the bound sound.** The violation v(·) = bound − σ is estimated with L-BFGS-B, starting from the best sample. The bound is then shifted by the violation plus 1e-6 plus 2δ. If dReal cannot verify it, the shift is increased in steps of 1e-6. dReal is δ-complete, so "we must shift our plane upward by δ in addition to the true v_l"; they use δ = 1e-7. — [LinSyn](https://arxiv.org/abs/2201.13351)
- **Correctness argument.** "The correctness (i.e. soundness) of the synthesized bounds is guaranteed if the v_l returned by Algorithm 1 is a true lower bound on v(·). Since Algorithm 1 does not return until dReal verifies v_l … the correctness is guaranteed." — [LinSyn](https://arxiv.org/abs/2201.13351)
- **Cost.**
  - "the entire process typically takes less than 1/100th of a second per pair of bounds".
  - Table 1, certified fraction and time, LinSyn vs auto_LiRPA:

    | Network | LinSyn | auto_LiRPA |
    |---|---|---|
    | MNIST 4-layer CNN, Swish | 0.76 in 796 s | 0.34 in 15 s |
    | MNIST 4-layer CNN, GELU | 0.72 in 814 s | 0.01 in 359 s |
    | CIFAR 5-layer CNN, GELU | 0.31 in 1163 s | 0.00 in 1217 s |

  - The comparison ran on an 8-core i7 CPU with 6 threads. — [LinSyn](https://arxiv.org/abs/2201.13351)
- **Positioning.** LinSyn handles composite activations "as a whole", for example swish(x) = x·σ(x) and GELU. It compares against POPQORN, which "tackles the challenge of computing tight linear bounds for sigmoid(x)×tanh(y) and x×sigmoid(y) using an expensive gradient descent based approach". — [LinSyn](https://arxiv.org/abs/2201.13351)
- **Pasado's contrast.** "unlike [Paulsen and Wang 2022a,b] we do not use an SMT solver, as we can solve our optimization problems directly." — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)

### Inferences
- **LinSyn is the generic, expensive end** of the "heuristic plane + certified offset" spectrum. Pasado and Prover are the specialized, closed-form end: the structure of f'(x)·y reduces the certified max to corners plus cubic roots.
- **For a GPU closed form**, the LinSyn and Prover pattern confirms that the plane can come from any cheap heuristic: LS, LP or a closed form. The cost and the soundness burden sit in the max-violation step. Pasado's version of that step is already closed form and vectorizable.
- **LinSyn's margins** (δ = 1e-7 from dReal, plus 1e-6 for optimizer slack) show what a careful tool adds to absorb numerical slack. Pasado adds no margin.

### Gaps
- I did not check whether LinSyn's dReal step is itself sound with respect to floating point. dReal uses interval arithmetic, which suggests it is, but LinSyn does not claim this in the text I grepped.
- I did not find a public LinSyn repository during this session.

---

## Q4. auto_LiRPA / CROWN / α-CROWN for products and sigmoid; Shi et al. NeurIPS'22 for derivative products

### Takeaway
LiRPA-style tools don't fit planes by regression. They use closed-form families of sound linear bounds and choose within the family for tightness:
- **Bilinear x·y:** planes from Shi et al. ICLR'20 with α_L = l_y, β_L = l_x, γ_L = −l_x l_y (upper: α_U = u_y, β_U = l_x, γ_U = −l_x u_y). These coincide with McCormick facets (my identification; the paper doesn't use the term) and are chosen to minimize the integral gap.
- **Sigmoid:** tangent and secant lines, with tangent points found by binary search.
- **ReLU lower bound:** any slope in [0,1] is sound. α-CROWN optimizes that slope.

Shi et al. NeurIPS'22 bound the Jacobian chain product J·Δ, where Δ is the activation derivative, with a provably optimal closed-form relaxation for ReLU. For non-ReLU activations they keep only an interval [l,u] for σ'(z), so the correlation between σ'(z) and z is lost. That correlation is exactly what Pasado's joint plane in (x,y) keeps.

### Cited Findings
- **Shi et al. ICLR'20 (transformers), z = xy.** "we derive optimal parameters: α_L = l_y, α_U = u_y, β_L = β_U = l_x, γ_L = −l_x l_y, γ_U = −l_x u_y". The goal is to minimize "the integral of the difference function F_L(x,y) = xy − (α_L x + β_L y + γ_L)" (Appendix C). — [Shi et al. 2020](https://arxiv.org/abs/2002.06622)
- **auto_LiRPA.** It treats "multiplication and division … [as] binary nonlinear functions, and matrix multiplication or dot product of two variable matrices … as multiplications with an affine transformation", with relaxations taken from prior work. For the ReLU lower bound, "it can be any line with 0 ≤ α ≤ 1 and β = 0. To minimize the relaxation error, Zhang et al. proposed to adaptively choose α". — [Xu et al. 2020 (auto_LiRPA)](https://arxiv.org/abs/2002.12920)
- **CROWN, general activations.** For concave σ on [l,u] "we can let h_U be any tangent line of σ(y) at point d ∈ [l, u]". For mixed-curvature segments, tangent lines pass through an end-point, and "the value of d for transcendental functions can be found using a binary search". — [Zhang et al. 2018 (CROWN)](https://arxiv.org/abs/1811.00866)
- **Shi et al. NeurIPS'22, Lipschitz via bound propagation.**
  - They model the Clarke Jacobian as a backward graph and relax the product [J_{i+1}(x)]_j·[Δ_i(x)]_jj, with Δ ∈ [0,1] for ReLU, using a closed-form relaxation (Prop. 3).
  - Theorem 4 proves this relaxation "is the tightest", and it is "closed-form with a provable optimality, without gradient-based optimization".
  - Using the generic multiplication relaxation of Shi et al. 2020 instead "will produce a similarly loose relaxation as [RecurJac]". — [Shi et al. 2022](https://arxiv.org/abs/2210.07394)
- **Shi et al. 2022 for non-ReLU activations (App. A.3).** "we can still obtain the range of [Δ_i(x)]_jj … Suppose the Clarke gradient satisfies [Δ_i(x)]_jj ⊆ [l,u]. Then [J·Δ] ≤ max{J·l, J·u}". This is a Leaky-ReLU-like function of J only. The paper also notes "we mainly focus on feedforward ReLU networks". — [Shi et al. 2022](https://arxiv.org/abs/2210.07394)
- **Sigmoid tightness work.**
  - Zhang et al. (ASE'22) show that neuron-wise "tightness definitions are heuristic". They propose "network-wise tightness" and two "provably tightest" approximations. — [Zhang et al. 2022](https://arxiv.org/abs/2208.09872)
  - König et al. (2024) propose "a novel parameter search method to improve the quality of these linear approximations" for sigmoidal activations, adapted per verification problem. — [König et al. 2024](https://arxiv.org/abs/2406.10154)

### Inferences
- **No LiRPA-family work fits σ'(z)·y jointly.** For smooth activations, Shi 2022 uses σ'(z) ∈ [l,u] times J. Pasado's plane Ax+By+C keeps first-order dependence on both x (= z) and y (= J). That is the precision advantage Pasado claims over "standard" transformers.
- **The bilinear-plane relaxations offer a cheap alternative rule for products.** Within the zonotope domain, though, the standard zonotope/affine-arithmetic product already uses the minimax plane of xy (see Q5).
- **One-sided versus symmetric.** LiRPA selects separate lower and upper planes (one-sided, non-parallel). A zonotope transformer needs a single plane with a symmetric ±D. So the LiRPA choice rules (minimize the integral gap of each side) don't transfer directly. The zonotope analogue is DeepZ's minimal-area parallelogram, or minimax.

### Gaps
- I did not fetch the α-CROWN paper (Xu et al. ICLR'21). The statement that α-CROWN optimizes ReLU and sigmoid slopes by gradient, with any α in the valid range sound, is from memory plus the auto_LiRPA "any 0 ≤ α ≤ 1" quote. Verify before citing.
- I did not check the current auto_LiRPA source (`BoundMul`, `BoundSigmoid`) for exact implementation details, such as precomputed tangent tables.

---

## Q5. DeepZ/DeepPoly sigmoid transformers, zonotope multiplication, and minimax (Chebyshev) versus least-squares plane selection

### Takeaway
- **DeepZ** picks the sigmoid zonotope transformer as the minimum-area parallelogram among sound pointwise options: slope λ = min(f'(l), f'(u)). It implements this soundly in floating point with interval coefficients and directed rounding.
- **Affine arithmetic** distinguishes Chebyshev (minimax error) from Min-Range approximations. Any Δ-bounded approximation is sound.
- **Minimum-error multiplication** of affine forms is studied explicitly (Skalna & Hladík 2017).
- **No surveyed work** fits the zonotope plane of a bivariate composite by least squares except Pasado and Phocus.
- **[own experiment]** The minimax plane lowers D (median 0.93× of LS) but barely changes the total radius proxy (median 0.999×, 10th percentile 0.972×). The N(0,1e-4) noise on A costs up to about 1.9× radius in tiny boxes (90th percentile). A tensor-grid least-squares fit has an exact closed form with no gelsd needed.

### Cited Findings
- **DeepZ sigmoid/tanh (Theorem 3.2).** Among pointwise zonotope approximations with l_y = σ(l_x) and u_y = σ(u_x), "The new affine form ŷ for the output with the minimum area in the xy-plane" is ŷ = [λ_l, λ_u]·x̂ + [μ1_l, μ1_u] + [μ2_l, μ2_u]·ε_new.
  - λ_opt = min(f'(l_x), f'(u_x)).
  - λ, μ1 and μ2 are computed "using rounding towards −∞ and +∞ and adding the error due to the non-associativity of floating point addition". — [Singh et al. NeurIPS'18 (DeepZ)](https://proceedings.neurips.cc/paper_files/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf)
- **DeepZ floating-point model.** It uses interval affine forms whose coefficients are intervals. "To ensure soundness with respect to different rounding modes and to account for the lack algebraic properties such as associativity and distributivity in the floating point world, the lower bounds and the upper bounds are rounded towards −∞ and +∞ respectively and suitable error values are added as explained in [Miné 2004]". It is implemented in ELINA. — [DeepZ](https://proceedings.neurips.cc/paper_files/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf)
- **Chebyshev versus Min-Range in affine arithmetic.** "the Chebyshev approximation … is minimizing the maximum error between f(x) and px + q on X, where the Min-Range approximation minimizes the range". For multiplication C·D = ⟨cd; cδ + γd, e⟩ "any e … satisfying (7) is a suitable choice". The optimal e is max |(Σγ_i ε_i)(Σδ_i ε_i)|, which takes O(n²) to compute. — [Rump & Kashiwagi](https://www.tuhh.de/ti3/paper/rump/RuKas14.pdf)
- **Minimum-error multiplication.** Skalna & Hladík (2017) give "a new algorithm for Chebyshev minimum-error multiplication of reduced affine forms" that runs in linear time. (Source: bibliographic and abstract entry only; I did not read the paper.) — [Hladík publication page](https://kam.mff.cuni.cz/~hladik/publ/b2hd-SkaHla2017a.html)
- **DeepT (Bonaert et al. PLDI'21).** It introduces Multi-norm Zonotopes and "all Multi-norm Zonotope abstract transformers necessary", "including the challenging softmax function and dot product", "carefully constructed to strike a good balance between precision and performance". — [PLDI'21 page](https://pldi21.sigplan.org/details/pldi-2021-papers/31/Fast-and-Precise-Certification-of-Transformers); [eth-sri/DeepT](https://github.com/eth-sri/DeepT)
- **Prover on sample count.** The precision of its LP-based planes "increases with the number of samples". It also splits the box into sub-regions to get extra candidate planes. — [Ryou et al. 2021](https://arxiv.org/abs/2005.13300)

### Inferences
- **Standard zonotope multiplication is already minimax.** For xy on a box, xy − (y₀x + x₀y − x₀y₀) = (x−x₀)(y−y₀) ∈ [−r_x r_y, r_x r_y]. So the standard affine-arithmetic product (plane through the center, new symbol r_x r_y) is the Chebyshev plane of xy. Pasado's regression plane departs from this only because σ'(x) is nonlinear in x.
- **Closed form for LS on a full tensor grid** (verified numerically, see below).
  - The centered x and y columns of a full tensor grid are orthogonal, so the 3×3 normal equations decouple.
  - For the separable g = h(x)·y with h = σ':
    - B = mean_i h(x_i)
    - A = ȳ · Σ_i (x_i − x̄)(h(x_i) − h̄) / Σ_i (x_i − x̄)²
    - C = h̄·ȳ − A·x̄ − B·ȳ
  - This needs only 5 σ' evaluations per neuron, not 25, and no gelsd.
  - **[own experiment]** The largest coefficient difference from `np.linalg.lstsq` over 300 non-degenerate boxes was 2.1e-11.
- **Degenerate boxes.** When l_y = u_y or l_x = u_x the design is rank-deficient. gelsd then returns the minimum-norm solution: for constant y = y₀ it splits the intercept between B and C. The closed form above gives B = h̄ instead. Both planes are sound, because any plane is, but they differ numerically. Special-case this if bit-level parity with upstream matters.
- **[own experiment] Plane comparison.** Setup: 300 random boxes, x-center in [−6,6], x-radius 1e-3 to 5, y-center in [−3,3], y-radius 1e-3 to 3.2, float64. Ratios are relative to the 5×5 LS plane. W = |A|r_x + |B|r_y + D is an interval-radius proxy that ignores zonotope correlations.

  | Plane rule | D ratio: median [10th–90th pct] | W ratio: median [10th–90th pct] |
  |---|---|---|
  | Minimax (LP on 81×81 grid, then exact D) | 0.93 [0.76–0.997] | 0.999 [0.972–1.000] |
  | Center tangent plane | 1.19 [1.003–2.27] | 1.003 [1.000–1.27] |
  | LS + N(0,1e-4) on A, as upstream | 1.34 [0.998–127] | 1.006 [0.999–1.88] |
  | Random (σ = 0.05 noise on A, B, C) | ≈451 | ≈9.7 |

  - The noise on A hurts mainly in tiny boxes, where 1e-4·r_x dominates D.
  - **Conclusion:** 5×5 LS is close to the best cheap rule for this proxy. Minimax buys at most a few percent in W. The 1e-4 A-noise is the largest avoidable precision loss among the rules tested. The paper's own δ is below 1e-9.

### Gaps
- I could not get the DeepT PDF (no arXiv version; ETH links returned 404 or HTML). The exact form of its multiplication and dot-product transformers (fast vs precise variants, handling of ε_i·ε_j terms) is not confirmed here.
- I did not read DeepPoly's sigmoid transformer or PRIMA directly. PRIMA is about multi-neuron relaxations, not planar products.
- Kochdumper et al. 2022 (polynomial zonotopes) and Ko et al. 2019 are cited by Pasado as composite-abstraction work. I did not check how they choose their approximating polynomial.
- I found no paper that states the L∞ suboptimality of discrete least squares versus minimax for this class of functions. The [own experiment] numbers are the only quantification.

---

## Q6. Floating-point soundness in zonotope domains: what a reviewer would expect for the error term

### Takeaway
There are two accepted positions:
1. **Disclaim floating point.** Claim soundness over the reals and state explicitly that floating point is not covered. Pasado (OOPSLA'23) and Phocus (OOPSLA'25) both do this.
2. **Implement a floating-point-sound mode** following Miné 2004: interval-valued coefficients, directed rounding, and explicit error terms for non-associativity. DeepZ/ELINA do this, and so does Laurel's own DeepJ (POPL'22), at 2.8–4.1× runtime overhead and with negligible change (< 1e-10) in results.

Jia & Rinard (2021) show that ignoring floating point can be exploited in practice. So a reviewer will at minimum expect the disclaimer. For the σ'·y error term, a sound mode would also need sound enclosures of σ and σ', and a margin for root inaccuracy.

### Cited Findings
- **Pasado:** "assumes ideal real arithmetic and is thus not floating-point sound (though floating-point sound versions of the operations exist [Miné 2004])". — [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- **Phocus:** "assumes ideal real arithmetic and thus is not floating-point sound though there exist several techniques to add floating-point soundness at the cost of additional runtime [Miné 2004]". — [Phocus OOPSLA'25](https://jsl1994.github.io/papers/OOPSLA2025.pdf)
- **DeepJ (POPL'22)** "optionally offers floating-point soundness [Miné 2004], i.e., its result can capture all possible outputs under different rounding modes and under different orders of computations". The differences are "negligible (< 10⁻¹⁰)", but there is "up to 4x runtime overhead" (2.8–4.1×). — [Laurel et al. POPL'22](https://jsl1994.github.io/papers/POPL2022Final.pdf)
- **DeepZ:** interval affine forms, with bounds "rounded towards −∞ and +∞ respectively and suitable error values are added as explained in [Miné 2004]". Transformers are "sound w.r.t. floating point arithmetic". — [DeepZ NeurIPS'18](https://proceedings.neurips.cc/paper_files/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf)
- **INTLAB affine arithmetic.** "all operations in the definition of a new affine quantity are executed in interval arithmetic, then the midpoint is stored and the radius added to the error term". Rounding errors go into a separate `.rnderr` term computed with upward rounding. "Rounding errors in the computation p, q, Δ for an approximation are treated appropriately." — [Rump & Kashiwagi](https://www.tuhh.de/ti3/paper/rump/RuKas14.pdf)
- **Jia & Rinard 2021:** "many verifiers inaccurately model floating point arithmetic", and "the negligence of floating point error leads to unsound verification that can be systematically exploited in practice". "Any verification system must accurately (or conservatively) model the effects of any floating point computations." — [Jia & Rinard (SAS'21)](https://arxiv.org/abs/2003.03021)
- **Miné 2004** ("Relational abstract domains for the detection of floating-point run-time errors", ESOP) is the standard citation used by DeepZ, DeepJ, Pasado and Phocus for floating-point-sound linear and relational domains. — cited in [DeepZ](https://proceedings.neurips.cc/paper_files/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf) and [Pasado](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)

### Inferences
- **Minimal reviewer-proof position for a GPU port of the planar transformer:**
  1. Keep the real-arithmetic soundness claim (Theorem 4.1, or the simpler y-convexity argument) and the explicit floating-point disclaimer, as Pasado and Phocus do.
  2. Guarantee that D is computed from exactly the (A,B,C) that are emitted, after any dtype casts or noise.
  3. Optionally add a cheap a-posteriori margin: an ulp-scaled bound on the evaluation error of |g| + |Ax| + |By| + |C| at the candidate points, plus a curvature-based bound for root error (½·max|φ''|·δx²).
- **A fully floating-point-sound GPU version** would need directed rounding or interval arithmetic for the zonotope update (A·x̂ + B·ŷ + C) and for D. It would also need a verified enclosure of the cubic roots, for example interval Newton or bisection on the bracket, and sound σ/σ' enclosures. DeepJ suggests a 3–4× overhead is the realistic cost.
- **Low-cost hardening of the root path.** Replace Cardano in complex64 with exponent 0.33333 by either the trigonometric three-real-root formula in float64, or a bracketed Newton/bisection on the two monotone branches of σ'' (x < x_peak and x > x_peak, where σ''' = 0). Then evaluate E on a small bracket around each root, not at a single point.

### Gaps
- I did not verify which directed-rounding facilities are available in PyTorch or CUDA. From general knowledge, CUDA has `_rd`/`_ru` intrinsics and PyTorch does not expose rounding modes. Confirm before relying on this.
- I did not read Miné 2004 directly. Its content is summarized here only as cited by others.
- I found no published floating-point-sound treatment of a regression-plane-plus-max-deviation transformer like Pasado's or Phocus's.
