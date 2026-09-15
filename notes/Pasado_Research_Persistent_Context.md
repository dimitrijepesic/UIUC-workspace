# Pasado Research Project — Persistent Technical Context

> **Purpose of this document**  
> This file is intended to be added to a Claude Project / ChatGPT Project as persistent context. It records the work completed so far on the UIUC Pasado research project, the measured results, implementation details, experiments, current repository state, and the exact point where work stopped.
>
> **Important status convention used below**
>
> - **[IMPLEMENTED]** — code change exists in the working tree.
> - **[MEASURED]** — result was actually run/measured and reported in this project work.
> - **[VERIFIED]** — correctness or behavior was explicitly tested.
> - **[ANALYZED]** — investigated/profiling result exists, but not necessarily optimized.
> - **[PREPARED]** — helper/script/experiment exists but has not yet been fully run or integrated.
> - **[PLANNED]** — explicitly discussed next direction, not yet completed.
> - **[REJECTED]** — investigated and intentionally not pursued based on evidence.
>
> Do not silently promote a **[PREPARED]** or **[PLANNED]** item into a completed result.

---

## 1. Project Context

### 1.1 Research setting

The work is part of a remote research internship in the UIUC SRSE program, advised by Prof. Saša Misailović, with a focus on compilers / programming languages / ML systems.

The immediate codebase is **Pasado**, a research artifact for precise static analysis of automatic-differentiation / differentiable programs using abstract interpretation.

The project is currently being treated as a **systems / compiler optimization problem**:

1. reproduce and understand Pasado;
2. profile real workloads;
3. identify where the precise analysis spends time;
4. remove obvious implementation overhead first;
5. only after the tensor path is cleaner, evaluate compiler and accelerator techniques such as `torch.compile` / TorchInductor, GPU execution, JAX/XLA, and possibly fusion/locality ideas inspired by Neptune.

The research is **not** currently “apply Triton/Neptune/JAX blindly.” The working principle established in this project is:

> First identify what is actually expensive in Pasado, then choose the compiler/runtime technique that matches the bottleneck.

### 1.2 Current research framing

Pasado is not a normal neural-network optimizer. It computes abstract transformers / bounds for differentiable programs.

The current workload under detailed investigation is the **Section 5.4 local Lipschitz benchmark**. The precise path computes much tighter bounds than interval or ordinary zonotope analysis, but is significantly more expensive.

The main empirical question so far has been:

> How much of the cost of Pasado’s precise transformer comes from the actual numerical analysis, and how much comes from Python orchestration, many small PyTorch operations, and repeated small linear algebra problems?

A broader possible research direction has been discussed: domain-aware batching/fusion for abstract transformers, with PyTorch/JAX/GPU backends. This is only **research framing / future direction**, not a completed contribution.

---

## 2. Repository / Environment

### 2.1 Local repository

Observed local paths:

```text
C:\everything\prakse\UIUC\Pasado
```

Main working directory for the current benchmark:

```text
C:\everything\prakse\UIUC\Pasado\Section_5_4
```

Shell/environment seen during runs:

```text
(.venv) (base) PS C:\everything\prakse\UIUC\Pasado\Section_5_4>
```

Platform:

- Windows
- PowerShell
- local CPU execution for the current runs
- PyTorch reported no accelerator available during these runs

Recurring PyTorch warning:

```text
UserWarning: 'pin_memory' argument is set as true but no accelerator is found,
then device pinned memory won't be used.
```

This warning is informational and did not stop the benchmark.

### 2.2 PyTorch / numerical environment

During the later `lstsq` investigation, the reported environment used:

```text
PyTorch 2.12.1
Windows CPU
```

The exact final dependency lockfile has not yet been established in this chat.

### 2.3 Windows-specific noise observed

The benchmark originally used shell commands such as:

```python
os.system('mkdir -p results')
```

On Windows this produced benign messages such as:

```text
A subdirectory or file -p already exists.
Error occurred while processing: -p.
```

This was not a benchmark failure.

The later `--no-save` path skips the final result-saving block and therefore also avoids this noise for no-save profiling runs.

---

## 3. Earlier Reproducibility / Compatibility Work

Before the current optimization sprint, Pasado was reproduced across several artifact sections.

### 3.1 Tests / benchmarks reported as passing

**[MEASURED]**

- `forward_mode_non_tensorized_src/tests/*.py` — 4 tests
- `reverse_mode_non_tensorized_src/tests/*.py` — 3 tests
- `Section_5_2/climate.sh`
- `Section_5_2/chemical.sh`
- `Section_5_3/black_scholes.sh`
- `Section_5_4/lipschitz.sh` — fully connected networks
- `Section_5_5/adult_script.py`

### 3.2 Compatibility issues found

#### NumPy 2.0: `np.product` removal

**[ANALYZED] / compatibility fix identified**

CNN Lipschitz path failed at:

```text
forward_mode_tensorized_src/conv.py:38
```

because `np.product` was removed in NumPy 2.0.

Required compatibility change:

```python
np.product(...)
```

to:

```python
np.prod(...)
```

This affects `Section_5_4/lipschitz_cnn.sh`.

#### NumPy repr / parsing change in Adult plot

**[ANALYZED] / compatibility fix identified**

`Section_5_5/adult_plot.py` failed around line 21 because code depended on a previous NumPy string representation and attempted to parse an integer from it.

The discussed repair was to use robust numeric extraction / explicit integer conversion rather than depending on NumPy repr formatting.

#### `PYTHONPATH` for Section 5.2 shell scripts

**[ANALYZED] / compatibility fix identified**

For:

```text
Section_5_2/climate.sh
Section_5_2/chemical.sh
```

the proposed reproducibility fix was:

```sh
export PYTHONPATH="$(cd "$(dirname "$0")/../forward_mode_non_tensorized_src" && pwd):$PYTHONPATH"
```

#### scikit-learn pickle version warning

Pretrained models were reported as serialized with approximately:

```text
scikit-learn 1.3.0
```

while a newer environment around:

```text
scikit-learn 1.9.0
```

was loading them.

This gives `InconsistentVersionWarning`. It did not crash the run, but it is a reproducibility risk because serialized sklearn objects are not guaranteed to behave identically across versions.

### 3.3 Planned separate compatibility PR

**[PLANNED, deferred]**

A separate small PR for outdated dependencies / artifact reproducibility was discussed. It should remain separate from the performance research changes.

Possible scope discussed:

- `np.product -> np.prod`
- robust `adult_plot.py` parsing
- `PYTHONPATH` fixes
- dependency/version documentation
- sklearn pickle warning documentation
- possibly Windows-safe directory creation

This PR has **not** been completed in the current optimization sprint.

---

# 4. Section 5.4 Lipschitz Benchmark

## 4.1 Entry point

Main shell script:

```text
Section_5_4/lipschitz.sh
```

Observed contents:

```sh
#!/bin/sh

python3 get_lipschitz.py --network 3layer
python3 get_lipschitz.py --network 4layer
python3 get_lipschitz.py --network 5layer
python3 get_lipschitz.py --network big
```

Main Python benchmark:

```text
Section_5_4/get_lipschitz.py
```

Model definitions:

```text
Section_5_4/model.py
```

## 4.2 Networks

### `3layer`

`FCN(3)`.

Architecture:

```text
784 -> 100 -> 100 -> 10
```

with sigmoid after the hidden layers.

Conceptually:

```text
fc1:      Linear(784 -> 100) + sigmoid
fc2:      Linear(100 -> 100) + sigmoid
fc_final: Linear(100 -> 10)
```

### `4layer`

`FCN(4)`.

Adds one hidden layer:

```text
784 -> 100 -> 100 -> 100 -> 10
```

### `5layer`

`FCN(5)`.

Adds another hidden layer:

```text
784 -> 100 -> 100 -> 100 -> 100 -> 10
```

### `big`

Separate `FCNBig` implementation rather than another width-100 `FCN`.

Architecture:

```text
784 -> 1024 -> 1024 -> 1024 -> 1024 -> 10
```

with sigmoid after each hidden layer.

This network is much wider than `3layer/4layer/5layer` and is the dominant Section 5.4 runtime.

## 4.3 Benchmark workload

The benchmark evaluates:

- 30 correctly classified MNIST images
- 16 epsilon radii
- three abstract-analysis modes:
  - interval
  - zonotope
  - precise zonotope

The 16 epsilon values observed in the run were:

```text
0.6324555320336759
0.3556558820077846
0.2
0.11246826503806982
0.06324555320336758
0.03556558820077846
0.02
0.011246826503806982
0.006324555320336759
0.0035565588200778455
0.002
0.0011246826503806982
0.0006324555320336759
0.00035565588200778454
0.0002
0.00011246826503806981
```

The benchmark therefore repeatedly runs analysis over:

```text
16 epsilons x 30 images
```

for each network.

## 4.4 Result files in the original benchmark

`get_lipschitz.py` originally saves results under:

```text
Section_5_4/results/
```

with names such as:

```text
lc_zonos_{network}.pth
lc_intervals_{network}.pth
lc_precise_{network}.pth

time_zonos_{network}.pth
time_intervals_{network}.pth
time_precise_{network}.pth
```

---

# 5. Initial Runtime Measurements

## 5.1 First full `lipschitz.sh` run

**[MEASURED]**

Early wall-clock results:

| Network | Observed runtime |
|---|---:|
| `3layer` | ~43.8 s |
| `4layer` | ~52.7 s |
| `5layer` | ~1m06s |
| `big` | ~21m40s |

The initial conclusion was that `big` dominates the total benchmark runtime.

These early times were gathered under a different run configuration than the later controlled old-vs-new benchmark harness. Absolute wall-clock times across these different experiments should **not** be mixed as if they were directly comparable.

## 5.2 Qualitative bound behavior observed

The benchmark-level results also showed the expected cost/precision tradeoff:

```text
precise <= zono <= interval
```

for the bound values.

At larger epsilon values the gap was very large. One reported example on `big`, epsilon approximately `0.632`:

```text
interval ~ 16.11M
zono     ~ 2.02M
precise  ~ 13,029
```

This is important context: the precise transformer is expensive, but it is also the reason the analysis can produce substantially tighter results.

---

# 6. Profiling Chronology

## 6.1 First cProfile attempt was polluted by serialization

The first `3layer` cProfile included the final `torch.save` operations.

Reported first-profile characteristics:

```text
~12.58M function calls
~210.7s under cProfile
```

Major entry:

```text
torch.save / serialization._save ~93s
```

Compute path was still visible:

```text
forward_zono_precise
-> PreciseSigmoidDualZonotope
-> sigmoid_prime_product_tensor
-> compute_max_error
```

but serialization was large enough to make the profile unsuitable for compute optimization.

## 6.2 `--no-save` flag

**[IMPLEMENTED]**

A `--no-save` option was added to:

```text
Section_5_4/get_lipschitz.py
```

Conceptually:

```python
parser.add_argument("--no-save", action="store_true")
```

and the final save block became conditional:

```python
if not args.no_save:
    os.system('mkdir -p results')
    torch.save(...)
    ...
```

Important behavior:

- default benchmark behavior remains unchanged;
- `--no-save` skips only final result serialization;
- benchmark computation remains the same.

Example run:

```powershell
python .\get_lipschitz.py --network 3layer --no-save
```

## 6.3 Clean compute-only `3layer` cProfile

Command:

```powershell
python -m cProfile -o ..\profiles\lipschitz_3layer_nosave.prof `
    .\get_lipschitz.py --network 3layer --no-save
```

Profile artifact:

```text
profiles/lipschitz_3layer_nosave.prof
```

pstats artifact:

```text
logs/lipschitz_3layer_nosave_pstats.txt
```

Reported profile:

```text
13,758,372 function calls
13,646,859 primitive calls
45.546 s total under cProfile
```

### Top cumulative compute path

| Function | Cumulative time |
|---|---:|
| `get_lipschitz.py:92(forward_zono_precise)` | 33.151 s |
| `precise_transformer.py:493(PreciseSigmoidDualZonotope)` | 32.912 s |
| `precise_transformer.py:411(sigmoid_prime_product_tensor)` | 32.353 s |
| `precise_transformer.py:379(compute_max_error)` | 19.738 s |
| `objective_fn` | 10.896 s |
| `check_nonlinear_boundary` | 10.801 s |
| `check_corners` | 8.403 s |
| `sigmoid_prime_times_y` | 7.488 s |
| `lin_reg_tensor` | 5.003 s |
| `torch.linalg.lstsq` | 3.793 s |

Comparison of abstract-analysis paths:

```text
forward_zono_precise ~33.15s
forward_interval     ~1.56s
forward_zono         ~1.11s
```

This established that the main bottleneck was specifically the **precise sigmoid zonotope transformer**, not the normal neural network forward pass or the basic interval / zonotope paths.

### High-call-count operations in the baseline profile

Examples:

```text
objective_fn                    288,000 calls
sigmoid_prime_times_y           384,000 calls
torch.tensor                    396,528 calls
torch.sigmoid                   785,280 calls
lin_reg_tensor                   96,000 calls
torch.linalg.lstsq               96,000 calls
torch.cartesian_prod             96,000 calls
torch.linspace                  192,000 calls
```

This supported the hypothesis that a significant amount of time was being spent on:

- Python-level iteration;
- many small tensor operations;
- repeated tensor construction;
- repeated small linear algebra calls.

---

# 7. Hotspot Code Understanding

Main file:

```text
forward_mode_tensorized_src/precise_transformer.py
```

Key functions studied:

```text
lin_reg_tensor
objective_fn
check_corners
check_nonlinear_boundary
compute_max_error
sigmoid_prime_product_tensor
PreciseSigmoidDualZonotope
```

## 7.1 `objective_fn`

Conceptually computes the difference between the actual nonlinear expression and a planar approximation:

```text
F(x, y) = sigmoid'(x) * y - (A*x + B*y + C)
```

The code is approximately:

```python
def objective_fn(A, B, C, xy_tensor):
    return sigmoid_prime_times_y(xy_tensor) - PlanarApproximation(A, B, C, xy_tensor)
```

## 7.2 `lin_reg_tensor`

Original code:

```python
def lin_reg_tensor(x, zs):
    xplusone = torch.cat((torch.ones(x.size(0), 1), x), 1)
    R = torch.linalg.lstsq(
        xplusone,
        zs,
        driver='gelsd'
    ).solution
    R = R[0:xplusone.size(1)]
    return R
```

Important convention:

`torch.linalg.lstsq` returns coefficients in intercept-first form for this design matrix:

```text
[C, A, B]
```

Downstream code reorders these to:

```text
(A, B, C)
```

## 7.3 `check_corners`

Original implementation loops over neurons/components.

For each `i` it constructs the four corners:

```text
(lx[i], ly[i])
(lx[i], uy[i])
(ux[i], ly[i])
(ux[i], uy[i])
```

then evaluates `objective_fn` and returns the maximum absolute error at the box corners.

Original pattern:

```python
maxs = []
for i in range(lx.size()[0]):
    corner1 = torch.tensor([lx[i], ly[i]])
    corner2 = torch.tensor([lx[i], uy[i]])
    corner3 = torch.tensor([ux[i], ly[i]])
    corner4 = torch.tensor([ux[i], uy[i]])

    all_corners = torch.stack((corner1, corner2, corner3, corner4))

    evaluation = objective_fn(
        ABCs[i][0], ABCs[i][1], ABCs[i][2], all_corners
    )

    maxs.append(
        torch.max(torch.max(evaluation), -torch.min(evaluation))
    )
```

This was a clear Python-loop / tiny-op hotspot.

## 7.4 `check_nonlinear_boundary`

This checks extrema on nonlinear boundaries corresponding to fixed:

```text
y = ly
y = uy
```

Key structure:

1. compute `A / ly` and `A / uy`;
2. filter invalid values;
3. invert the sigmoid second derivative to obtain candidate roots;
4. filter roots to `[lx, ux]`;
5. pair roots with `ly` or `uy`;
6. evaluate the objective;
7. take max values.

The root inversion machinery was already mostly tensorized.

However, the final evaluation had per-neuron loops:

```python
for i in range(lx.size()[0]):
    evaluation = objective_fn(
        ABCs[i][0],
        ABCs[i][1],
        ABCs[i][2],
        roots_paired_stacked[i]
    )
    evaluation = torch.nan_to_num(evaluation, -float("Inf"))
    maxs.append(torch.max(evaluation))
```

and the same structure again for the `uy` branch.

## 7.5 `compute_max_error`

Original structure:

```python
errors = check_corners(...)
errors2_ly, errors2_uy = check_nonlinear_boundary(...)

max_errors = []
for i in range(len(errors)):
    max_errors.append(
        torch.max(
            torch.max(errors[i], errors2_ly[i]),
            errors2_uy[i]
        )
    )

return max_errors
```

## 7.6 `sigmoid_prime_product_tensor`

This is the main precise transformer subroutine identified by profiling.

Conceptual flow:

```text
x, y zonotopes
  -> lower/upper bounds
  -> sample 5x5 grids
  -> evaluate sigmoid'(x) * y
  -> planar regression Ax + By + C
  -> compute max approximation error
  -> use A/B/C to transform centers and generators
  -> add fresh error symbols
  -> return new Zonotope
```

Relevant code pattern:

```python
lx = x.get_lb()
ux = x.get_ub()
ly = y.get_lb()
uy = y.get_ub()

xys = get_linspace(lx, ux, ly, uy)

zs = [sigmoid_prime_times_y(xy) for xy in xys]

lineqs = [
    lin_reg_tensor(xys[i], zs[i])
    for i in range(len(zs))
]

ABCs = [
    (lineqs[i][1], lineqs[i][2], lineqs[i][0])
    for i in range(len(lineqs))
]
```

The original code also applies a small perturbation to `A`:

```python
ABCs = [
    (x[0] + np.random.normal(0, .0001), x[1], x[2])
    for x in ABCs
]
```

then constructs:

```python
As = torch.tensor([x[0] for x in ABCs])
Bs = torch.tensor([x[1] for x in ABCs])
Cs = torch.tensor([x[2] for x in ABCs])
```

and calls:

```python
errors = compute_max_error(...)
```

Finally:

```python
new_centers = As * x_centers + Bs * y_centers + Cs
new_generators = As * x_generators + Bs * y_generators

new_generators = torch.cat((new_generators, errors), 0)

return Zonotope(new_centers, new_generators)
```

---

# 8. Optimization 1 — Vectorized `check_corners`

## 8.1 Motivation

**[ANALYZED]**

The original `check_corners` was expensive because each neuron caused:

- one Python loop iteration;
- 4 new small `torch.tensor` constructions;
- a `torch.stack`;
- indexing into Python list/tuple `ABCs`;
- tiny `objective_fn` operations;
- several tiny reductions.

Microbenchmark evidence reported before integration:

| n | old | vectorized | speedup |
|---:|---:|---:|---:|
| 100 | ~8.07 ms | ~0.06 ms | ~138x |
| 500 | ~32.30 ms | ~0.15 ms | ~210x |
| 2000 | ~127.24 ms | ~0.36 ms | ~357x |

## 8.2 Implementation

**[IMPLEMENTED]**

New function:

```text
check_corners_tensor(ABCs_tensor, lx, ux, ly, uy)
```

added in:

```text
forward_mode_tensorized_src/precise_transformer.py
```

Input:

```text
ABCs_tensor shape = [n, 3]
```

with columns corresponding to:

```text
A, B, C
```

The implementation constructs all corners in batched tensor form, evaluates the objective for all neurons/components at once, and returns:

```text
[n]
```

instead of a Python list of scalar tensors.

The original `check_corners` remains in the file as a fallback.

## 8.3 Correctness test

**[VERIFIED]**

Test file:

```text
forward_mode_tensorized_src/test_check_corners_tensor.py
```

Reported cases included:

- random `n = 1 ... 1000`
- degenerate zero-width box

Observed differences were within float32 reordering tolerance in the initial test, approximately:

```text
<= 4.8e-7
```

Later dtype work made the end-to-end old/new paths substantially closer; see the dtype section below.

## 8.4 First integration result

`compute_max_error` was extended to optionally accept `ABCs_tensor`.

Conceptually:

```python
def compute_max_error(lx, ux, ly, uy, ABCs, ABCs_tensor=None):
    if ABCs_tensor is not None:
        errors = list(
            check_corners_tensor(
                ABCs_tensor, lx, ux, ly, uy
            ).unbind(0)
        )
    else:
        errors = check_corners(ABCs, lx, ux, ly, uy)
```

`sigmoid_prime_product_tensor` already has `As`, `Bs`, `Cs`, so:

```python
ABCs_tensor = torch.stack((As, Bs, Cs), dim=1)
```

was used to feed the tensorized path.

### Profile after only vectorized corners

**[MEASURED]**

| Metric | Baseline | Corners vectorized |
|---|---:|---:|
| Total cProfile time | 45.55 s | 33.997 s |
| `sigmoid_prime_product_tensor` | 32.35 s | 23.02 s |
| `compute_max_error` | 19.74 s | 11.09 s |
| corner check | 8.40 s | 0.155 s |
| `check_nonlinear_boundary` | 10.80 s | 10.33 s |

Real benchmark corner-check reduction:

```text
8.40s -> 0.155s
~54x
```

The full profiled benchmark dropped by about 25% at this intermediate stage.

---

# 9. Optimization 2 — Vectorized `check_nonlinear_boundary`

## 9.1 Motivation

After vectorizing `check_corners`, the remaining `compute_max_error` cost was almost entirely:

```text
check_nonlinear_boundary ~10.33s
```

This made it the next clear local target.

## 9.2 Implementation

**[IMPLEMENTED]**

New function:

```text
check_nonlinear_boundary_tensor(
    lx, ux, ly, uy, ABCs_tensor
)
```

was added to:

```text
forward_mode_tensorized_src/precise_transformer.py
```

A small private helper was also introduced:

```text
_max_objective_over_x_candidates
```

The new path:

- takes `ABCs_tensor [n,3]`;
- replaces loop-built `A_by_ly` / `A_by_uy` values with tensor division;
- retains the original tensorized cubic/root inversion machinery;
- replaces the two per-neuron `objective_fn + nan_to_num + max + append` loops with batched `[n,3]` operations;
- returns:
  - `maxs [n]`
  - `maxs_uy [n]`

The original `check_nonlinear_boundary` remains available.

## 9.3 Correctness test

**[VERIFIED]**

Test:

```text
forward_mode_tensorized_src/test_check_nonlinear_boundary_tensor.py
```

Reported coverage:

- random `n = 1 ... 500`
- `ly` / `uy` crossing zero
- roots forced outside `[lx, ux]`
- degenerate `lx == ux`
- degenerate `lx == ux` and `ly == uy`
- mixed positive/negative `A`
- `A` sweeping through zero

Initial differences:

```text
<= 4.8e-7
```

Later dtype fixes tightened the end-to-end equivalence further.

## 9.4 Integration into `compute_max_error`

**[IMPLEMENTED]**

When `ABCs_tensor` is present:

```python
errors2_ly_t, errors2_uy_t = check_nonlinear_boundary_tensor(
    lx, ux, ly, uy, ABCs_tensor
)

errors2_ly = list(errors2_ly_t.unbind(0))
errors2_uy = list(errors2_uy_t.unbind(0))
```

The list conversion was initially kept so downstream `max_errors` / `lst_as_tensor` behavior stayed compatible with the old code.

Fallback remains:

```python
errors2_ly, errors2_uy = check_nonlinear_boundary(
    lx, ux, ly, uy, ABCs
)
```

---

# 10. `3layer` Result After Both Vectorizations

## 10.1 cProfile result

**[MEASURED]**

| Metric | Baseline | Corners only | Both vectorized |
|---|---:|---:|---:|
| Total | 45.55 s | 33.997 s | 24.968 s |
| `sigmoid_prime_product_tensor` | 32.35 s | 23.02 s | 14.03 s |
| `compute_max_error` | 19.74 s | 11.09 s | 1.97 s |
| corner check | 8.40 s | 0.155 s | 0.156 s |
| nonlinear-boundary check | 10.80 s | 10.33 s | 1.070 s |

Key result:

```text
45.55s -> 24.97s
```

which is approximately:

```text
45% runtime reduction under cProfile
~1.82x cProfile speedup
```

The important interpretation is:

> `compute_max_error` stopped being the main bottleneck after the two vectorizations.

## 10.2 Normal wall-clock benchmark after later correctness fixes

A later benchmark harness ran old/new variants in a paired setup.

**[MEASURED]**

Reported `3layer` medians:

```text
original median: 140.3s
vec_both median: 75.0s
n = 3 runs per variant
```

Result:

```text
speedup factor ~1.87x
runtime reduction ~46.5%
```

This ratio is consistent with the earlier cProfile ratio.

### Important timing caveat

Absolute Section 5.4 times varied substantially across different sessions:

- early simple run: `3layer ~43.8s`
- later controlled benchmark harness: `original median ~140.3s`
- other no-save/smoke runs also differed

Therefore:

- compare variants only within the same controlled benchmark experiment;
- do not treat unrelated absolute times from different sessions as directly comparable;
- the `1.87x / 46.5%` paired result is the strongest old-vs-new `3layer` result reported so far.

---

# 11. Selector Infrastructure

## 11.1 Vectorization selector

**[IMPLEMENTED] / [VERIFIED]**

A runtime selector was added so old/new paths can be compared without editing source between runs.

Reported selector:

```text
PASADO_VECTORIZED_PRECISE=0/1
```

Default reported as:

```text
1
```

meaning vectorized precise checks are the default in the current working tree.

There is also:

```text
PASADO_VEC_BOUNDARY
```

which can independently revert the nonlinear boundary part, enabling the intermediate “corners-only” variant.

The selector is read at call time, allowing tests/harness code to switch paths in-process.

## 11.2 Selector-path test

**[VERIFIED]**

Test:

```text
forward_mode_tensorized_src/test_selector_paths.py
```

The test uses call counters to verify that:

- old mode actually executes the old functions;
- fully vectorized mode executes both new functions;
- corners-only mode mixes new corner checking with old boundary checking.

It also compares output zonotopes.

---

# 12. Important Dtype / Numerical Equivalence Bug

## 12.1 Problem discovered

**[ANALYZED] / [FIXED]**

The selector test exposed a subtle dtype mismatch.

Root cause:

```text
inverse_poly_tensor
```

uses:

```text
complex64
```

internally.

Therefore the roots naturally emerge as float32 after taking `.real`.

In the original path, these float32 roots were stacked together with:

```text
ly / uy
```

which were float64 in the tested benchmark.

That stack implicitly promoted the roots to float64 **before** the sigmoid/objective computation.

The new vectorized implementation initially kept the roots in float32 longer, so the sigmoid was effectively evaluated in float32.

This caused old/new differences around:

```text
~1e-6 in earlier correctness CSV rows
```

and an observed selector difference around:

```text
2.2e-8 in float64
```

which was too large to dismiss as normal operation reordering.

## 12.2 Fix

**[IMPLEMENTED]**

The vectorized roots are explicitly promoted to the `ly` / `uy` dtype, matching the implicit promotion that occurred in the old stack-based path.

The operation association was also aligned with the original expression, specifically preserving a form equivalent to:

```text
(1 - s) * (s * ys)
```

rather than relying on an algebraically equivalent but differently associated expression.

The mathematical algorithm was not changed.

## 12.3 Correctness after the fix

**[VERIFIED]**

Reported results:

Selector test:

```text
max difference ~2.2e-16
```

Existing vectorization unit tests:

```text
0.000e+00 difference
```

Reduced end-to-end harness on:

```text
3 images x 3 epsilon values x 3layer
```

Final Lipschitz bound:

```text
abs difference <= 5.7e-14
```

Centers / generators / lb / ub:

```text
<= 4.5e-14
```

Reported output shape:

```text
[402, 10]
```

Dtype:

```text
float64
```

NaN / Inf:

```text
no mismatched NaN/Inf patterns
```

Correctness CSV:

```text
logs/correctness_results.csv
```

Reported current state:

```text
45 rows
all PASS
```

This dtype discovery is important because it invalidated earlier looser “~1e-6 is fine” assumptions. The final vectorized path was adjusted to reproduce the original precision behavior much more closely.

---

# 13. `big` Network Runs

## 13.1 Optimized `big --no-save` normal run

Command:

```powershell
python .\get_lipschitz.py --network big --no-save
```

**[MEASURED]**

Observed runtime:

```text
18m15s
```

This run used the vectorized corner + boundary path.

Important caveat:

- the earlier original `big ~21m40s` run included result saving;
- therefore `21m40s -> 18m15s` is **not** a formal fair old-vs-new comparison;
- the later dedicated old-vs-new benchmark harness was started specifically to obtain a fair comparison;
- its final `big` numbers have **not been provided yet in this chat**.

Also, the 18m15s run predates or may predate the later dtype-equivalence cleanup; it is useful as an observed runtime, not necessarily the final post-fix benchmark number.

## 13.2 Optimized big cProfile

Command:

```powershell
python -m cProfile `
    -o ..\profiles\lipschitz_big_nosave_both_vectorized.prof `
    .\get_lipschitz.py --network big --no-save
```

Terminal progress completed around:

```text
20m37s
```

Profile artifact:

```text
profiles/lipschitz_big_nosave_both_vectorized.prof
```

pstats artifact:

```text
logs/lipschitz_big_nosave_both_vectorized_pstats.txt
```

Later pstats analysis reported approximately:

```text
1249s total profiled time
~61M calls
```

Use the profile mainly for **relative attribution**, not as a wall-clock benchmark.

---

# 14. New `big` Bottleneck Analysis

Analysis document:

```text
results/optimization_section_5_4/BIG_PROFILE_ANALYSIS.md
```

**[ANALYZED]**

## 14.1 `AffineZonotope` became the largest single hotspot

Reported:

```text
SimpleZono.AffineZonotope
~416s self time
~9600 calls
~33% of total profile
```

Interpretation:

- this is real dense generator-matrix propagation;
- it is primarily BLAS / large tensor work;
- unlike the original corner/boundary issue, it is not simply a Python-loop problem.

Reported attribution suggested approximately half of this work appears in the ordinary zonotope pass and half in the precise path, with roughly:

```text
~200s in the precise side
```

The exact split should be treated as profile attribution, not a separately timed benchmark.

## 14.2 Lower/upper-bound reductions

`get_coeff_abs` / related large reductions were also significant.

Reported components:

```text
abs ~73s
sum ~60s
```

Total roughly:

```text
~133s
```

Other large-tensor work reported:

```text
__mul__              ~51s
imatmul / matmul      ~37s
```

Approximate combined large matrix/reduction block:

```text
~637s
~51% of the big cProfile
```

This is mostly genuine numerical work rather than millions of trivial Python dispatches.

## 14.3 `compute_max_error` is no longer a main bottleneck

After the two vectorizations, on `big`:

```text
compute_max_error ~40.8s cumulative
~3.3% total profile
```

Inside it:

```text
check_corners_tensor             ~1.1s
check_nonlinear_boundary_tensor  ~5.5s
```

Remaining overhead included:

```text
~14.9s self time in the final Python combine loop
~5.6s from unbind/list conversion
```

This means the core optimization goal for `compute_max_error` was achieved.

A possible cleanup was identified:

```python
torch.maximum(...)
```

could replace the final Python combine loop and avoid `unbind`, but this cleanup has not been the main focus yet.

---

# 15. `lin_reg_tensor` / `torch.linalg.lstsq` Analysis

Analysis document:

```text
results/optimization_section_5_4/LSTSQ_ANALYSIS.md
```

**[ANALYZED]**

This became the next major **addressable** optimization block after the corner/boundary work.

## 15.1 Exact call counts

Reported formula:

```text
# sigmoid layers
x width
x 30 images
x 16 epsilons
```

Observed exact counts:

```text
3layer: 96,000 lin_reg_tensor / lstsq calls
big:   1,966,080 lin_reg_tensor / lstsq calls
```

The profile count matched the expected formula exactly.

## 15.2 Shapes

A reduced instrumented run used:

```text
1 image x 2 epsilon values
400 regression calls
```

All 400/400 calls had:

```text
x  = [25, 2]
zs = [25]
dtype = float64
```

This corresponds to a hardcoded/default:

```text
STEPS = 5
```

so the sampled grid is:

```text
5 x 5 = 25 points
```

Regression design matrix:

```text
[25, 3]
```

because of:

```text
[1, x, y]
```

columns.

This regular shape is ideal for batching.

## 15.3 Does the exact design matrix repeat?

**[REJECTED: value-cache idea]**

A SHA-1 check over 400 observed design matrices found:

```text
0 exact duplicates
```

Bounds depend on:

- neuron/component;
- epsilon;
- image;
- layer.

So a value-keyed cache would have essentially zero hit rate.

The repeated structure is only:

- shape;
- a column of ones;
- canonical 5x5 grid layout.

## 15.4 Batched `torch.linalg.lstsq`

**[VERIFIED experimentally]**

Target batching layout:

```text
A: [n, 25, 3]
B: [n, 25, 1]
```

or compatible `[n,25]` B representation.

Reported tests showed:

- batched `torch.linalg.lstsq` works on the current Windows CPU PyTorch environment;
- `driver='gelsd'` supports batched input in this setup;
- batched result was bit-identical to repeated individual `gelsd` calls in the performed tests.

## 15.5 Why keep `driver='gelsd'`

The default CPU driver is different (reported as `gelsy` / QR-based behavior).

Differences on ordinary cases were tiny, around:

```text
~3.3e-16
```

but rank-deficient cases are semantically more important: the default driver does not guarantee the same minimum-norm behavior as the SVD-based `gelsd` path.

Therefore the analysis explicitly concluded:

> Keep `driver='gelsd'`. Batching does not require changing the mathematical solver.

## 15.6 Where regression time goes on `big`

Reported approximate `big` cProfile block:

```text
torch.linalg.lstsq           ~119.6s
torch.cat for xplusone        ~89.4s
grid generation              ~94s
lin_reg_tensor self           ~18.4s
sigmoid_prime_times_y cum     ~60.7s
.item()                        ~8.3s
```

Grid block detail:

```text
torch.cartesian_prod ~39s
torch.linspace       ~21s
get_linspace self    ~33s
```

The aggregate regression/grid region was estimated around:

```text
~300s
```

and was entirely on the precise path.

## 15.7 Dispatch vs actual mathematics

The analysis concluded that the small `25x3` SVD itself is not the main reason each call is expensive.

Reported rough per-call cost:

```text
~37-61 microseconds per small lstsq call
```

while the mathematical work for such a tiny system is far smaller.

The dominant cost is repeated:

- Python function entry;
- argument parsing;
- allocation;
- LAPACK setup;
- `torch.cat`;
- per-neuron dispatch.

This is why batching was judged promising.

---

# 16. Experimental `lin_reg_tensor_batched`

## 16.1 Implementation location

**[IMPLEMENTED EXPERIMENTALLY, NOT YET IN MAIN PASADO PATH]**

File:

```text
experiments/batched_lin_reg.py
```

Function:

```text
lin_reg_tensor_batched(x_batch, zs_batch)
```

Expected inputs:

```text
x_batch:  [n, m, 2]
zs_batch: [n, m]
          or [n, m, 1]
```

Output:

```text
[n, 3]
```

with intercept-first convention:

```text
[C, A, B]
```

The implementation:

- creates the ones column in the dtype/device of the input;
- concatenates once in batch;
- calls one batched:

```python
torch.linalg.lstsq(..., driver='gelsd')
```

for the batch;
- contains no Python loop over `n`;
- does not use `torch.tensor(list_of_tensors)`.

The old:

```text
lin_reg_tensor
```

remains unchanged.

## 16.2 Tests

**[VERIFIED]**

Test file:

```text
experiments/test_batched_lin_reg.py
```

Reported coverage:

| Case | Result |
|---|---|
| `n = 1, 10, 100, 1024`, realistic grids | bit-identical |
| random points `n = 1,10,100` | bit-identical |
| degenerate x / rank 2 | bit-identical |
| point box / rank 1 | bit-identical |
| near-degenerate width `1e-12` | bit-identical |
| saturated sigmoid | bit-identical |
| mixed-scale | bit-identical |
| mixed healthy + degenerate batch | bit-identical |
| float64 | bit-identical |
| float32 under its default dtype test setup | bit-identical |
| downstream planar error | `0.000e+00` difference |
| `[n,m,1]` target | passed |
| shape/dtype/device preservation | passed |

Hard assertion tolerances used:

```text
float64: rtol=1e-12, atol=1e-14
float32: rtol=1e-5,  atol=1e-6
```

but actual observed results across the reported configurations were:

```text
torch.equal == True
```

i.e. zero coefficient difference.

## 16.3 Extra implementation observation

The old `lin_reg_tensor` builds:

```python
torch.ones(x.size(0), 1)
```

without explicitly inheriting `x.dtype/device`.

The experimental batched implementation uses the input dtype/device and is therefore more robust in that respect.

The old behavior was intentionally not changed during the analysis phase.

## 16.4 Current integration status

This point is critical:

> `lin_reg_tensor_batched` has been implemented and validated in `experiments/`, but at the last confirmed state in this chat it had **not yet been wired into the production precise-transformer path**.

A selector:

```text
PASADO_BATCHED_LSTSQ
```

has been prepared/added in the working tree, with default intended to remain OFF until integration validation.

However, the latest reported state says the batched implementation was waiting for:

1. the active big benchmark to finish;
2. a clean-machine microbenchmark;
3. integration into `precise_transformer.py`;
4. full old/new correctness and 3layer re-profiling.

Do not claim production batched `lstsq` speedup yet.

---

# 17. LSTSQ Microbenchmark Status

Script:

```text
experiments/benchmark_lstsq.py
```

**[PREPARED]**

It was reported as containing:

- warm-up;
- ~15 repetitions;
- median timing;
- CSV output.

At the last confirmed state, it had **not yet been run on an otherwise idle machine**, because a long `big` benchmark was running in the background and would contaminate timing.

The intended acceptance gate discussed was:

- correctness remains valid;
- stable run;
- more than ~5% useful improvement before wiring/keeping the optimization.

No final microbenchmark numbers were provided yet.

---

# 18. Grid Generation Investigation

## 18.1 `cartesian_prod`

It was experimentally observed that a replacement using:

```text
repeat_interleave / repeat
```

could reproduce `torch.cartesian_prod` exactly for the tested grid construction.

Reported:

```text
0 mismatches / 2000 trials
```

This suggests grid assembly itself can potentially be batched exactly.

## 18.2 `torch.linspace`

A direct affine-formula replacement for `torch.linspace` was **not** bit-identical.

Reported:

```text
280 mismatches / 2000 trials
~1 ULP
```

Therefore replacing `torch.linspace` with a hand-written affine expression is not considered a free exact transformation.

It may be a separate optional optimization with explicitly documented numerical differences, but it has not been adopted.

---

# 19. Caching Investigation

Analysis document:

```text
results/optimization_section_5_4/CACHING_ANALYSIS.md
```

**[REJECTED]**

Caching was explicitly investigated and rejected.

### Data-dependent candidates

Examples:

- generated linspace/grid values;
- `xplusone` design matrices.

Observed exact-repeat rate:

```text
0%
```

for the instrumented matrices.

A value-keyed cache would:

- have effectively no hits;
- require hashing/comparing tensor data;
- add overhead.

### Repeated constants

Examples:

- `ones(25,1)` column;
- grid structure;
- constants inside polynomial inversion.

Conclusion:

- batching removes repeated allocation more directly;
- tiny scalar constants contribute negligible runtime;
- caching is not justified by the current profile.

This is a completed negative result and should remain documented rather than repeatedly reconsidered without new evidence.

---

# 20. GPU Readiness Analysis

Document:

```text
results/optimization_section_5_4/GPU_READINESS.md
```

**[ANALYZED]**

Reported classification included approximately 15 items.

## 20.1 Already GPU-friendly

The new vectorized checking kernels are mostly pure tensor operations:

```text
check_corners_tensor
check_nonlinear_boundary_tensor
_max_objective_over_x_candidates
```

These are substantially more suitable for GPU execution / graph compilation than the original per-neuron Python loops.

## 20.2 Small device/dtype fixes required

Examples found:

- constants created inside `inverse_poly_tensor`;
- `torch.ones` in old `lin_reg_tensor`;
- `torch.eye` in `traceify`;
- several `torch.zeros` sites;
- entry-point model/data/device plumbing.

Preferred future pattern discussed:

```python
x.new_tensor(...)
x.new_ones(...)
torch.zeros(..., device=x.device, dtype=x.dtype)
```

No broad device rewrite has been claimed as completed.

## 20.3 Synchronization hazards

Examples:

```text
.item()
```

inside `get_linspace`.

Reported magnitude on `big`:

```text
~7.87 million .item() operations
```

Such operations are especially problematic for GPU execution because they can force device/host synchronization.

The original old paths also contain:

```python
torch.tensor([tensor_scalar, tensor_scalar])
```

style constructions, which are poor for GPU execution.

## 20.4 `gelsd` is a GPU blocker

Important finding:

```text
torch.linalg.lstsq(..., driver='gelsd')
```

is a CPU/LAPACK choice.

The reported CUDA support is different; only `gels` is available in the considered path, and `gels` assumes full rank.

Therefore:

> simply moving the current `gelsd` regression to GPU is not a semantics-preserving mechanical change.

Changing the solver is a research/numerical decision and has not been made.

## 20.5 FP64 concern

The current benchmark uses float64 in important paths.

Consumer GPUs, especially devices such as a T4, can have poor FP64 throughput relative to FP32.

Therefore a GPU port is not automatically expected to win.

The plan is to measure real FP64 throughput rather than assume GPU = faster.

---

# 21. Prepared Compiler / Profiler / GPU Experiments

These are **prepared**, not completed results.

## 21.1 `torch.compile`

Script:

```text
experiments/compile_experiment.py
```

**[PREPARED]**

Planned candidates include pure tensor functions such as:

- `check_corners_tensor`
- `check_nonlinear_boundary_tensor`
- `_max_objective_over_x_candidates`
- possibly batched regression code

The prepared experiment is intended to compare:

```text
eager
torch.compile default
torch.compile(mode="reduce-overhead")
```

and separate:

- first-call compilation latency;
- steady-state runtime;
- graph-break reasons via `torch._dynamo.explain`.

No final `torch.compile` performance result has been reported yet.

## 21.2 `torch.profiler`

Script:

```text
experiments/torch_profiler_compare.py
```

**[PREPARED]**

Intended reduced-workload comparison:

- old precise path;
- vectorized path;
- batched path once integrated.

Goal:

- operator call counts;
- self CPU time;
- total CPU time;
- confirm that vectorization actually removes tiny dispatches.

No final torch.profiler table has been reported yet.

## 21.3 Colab / GPU

Script:

```text
experiments/colab_gpu_profile.py
```

**[PREPARED]**

Reported intended behavior:

- check CUDA availability;
- print GPU model;
- measure float64 capability/performance;
- run reduced 3layer test;
- compare CPU/GPU numerical behavior;
- synchronize CUDA properly for timing;
- skip GPU result honestly if CUDA is unavailable.

Local no-CUDA behavior was reportedly checked so the script does not invent results.

No actual GPU benchmark result has been reported yet.

---

# 22. JAX / XLA Status

**[PLANNED / research context only]**

JAX was discussed as a future direction for two reasons:

1. Pasado analyzes automatic differentiation, so a JAX/JAXPR frontend is conceptually natural.
2. JAX `vmap` / `jit` / XLA may be useful as another backend or comparison point for the cleaned tensor computations.

A possible longer-term idea discussed with the professor was conceptually:

```text
Pasado for JAX
```

where the input could be a JAX program / JAX IR (JAXPR).

No JAX implementation, JAX benchmark, JAXPR frontend, or JAX/XLA speedup has been completed in this work so far.

---

# 23. Neptune Status

This must remain unambiguous:

> **Neptune has not been directly used in the current Pasado optimization work.**

No current result includes:

- running Pasado through Neptune;
- modifying Neptune code;
- applying a Neptune implementation directly to Pasado;
- measuring Pasado with/without Neptune;
- reproducing a Neptune performance result as part of this optimization sprint.

Neptune is currently **research context / inspiration** for:

- operator fusion;
- locality;
- avoiding intermediate materialization;
- GPU execution strategy.

A possible future connection is to inspect sequences such as:

```text
AffineZonotope
-> generator matmul
-> abs
-> sum/reduction
-> lower/upper bounds
```

and ask whether domain-aware fusion/locality ideas could reduce materialization or memory traffic.

That is not yet an implemented or measured result.

---

# 24. Big-vs-Small Scaling Conclusion

The most important profiling conclusion after the vectorizations is that the bottleneck changes with network width.

## `3layer` / width 100

The original implementation strongly exposed:

- Python loops;
- many tiny tensor operations;
- repeated per-neuron `objective_fn`;
- repeated small tensor creation.

The precise path was dominated by this orchestration.

## `big` / width 1024

After removing the obvious Python-loop overhead:

- dense `AffineZonotope` generator propagation becomes the largest single block;
- large `abs/sum` reductions become significant;
- the ~1.97M small regression calls are still the largest clearly batchable block.

Interpretation:

> `3layer` is useful for quickly finding Python orchestration overhead, but it underestimates the importance of dense matrix/reduction work that appears at width 1024.

The `big` model is therefore necessary for evaluating whether an optimization scales to the benchmark that dominates total runtime.

---

# 25. Current Files / Artifacts Reported in the Working Tree

The following paths have been explicitly mentioned as existing or created during this work.

## Main modified benchmark / implementation files

```text
Section_5_4/get_lipschitz.py
Section_5_4/lipschitz.sh
forward_mode_tensorized_src/precise_transformer.py
```

`get_lipschitz.py` includes `--no-save`.

`precise_transformer.py` includes:

- vectorized corner checking;
- vectorized nonlinear-boundary checking;
- vectorization selectors;
- dtype-equivalence fix;
- prepared `PASADO_BATCHED_LSTSQ` selector infrastructure.

## Unit / correctness tests

```text
forward_mode_tensorized_src/test_check_corners_tensor.py
forward_mode_tensorized_src/test_check_nonlinear_boundary_tensor.py
forward_mode_tensorized_src/test_compute_max_error_tensor.py
forward_mode_tensorized_src/test_selector_paths.py
```

## Experimental batched regression

```text
experiments/batched_lin_reg.py
experiments/test_batched_lin_reg.py
experiments/benchmark_lstsq.py
```

## Prepared compiler / profiler / GPU scripts

```text
experiments/compile_experiment.py
experiments/torch_profiler_compare.py
experiments/colab_gpu_profile.py
```

## Profiles / logs

```text
profiles/lipschitz_3layer_nosave.prof
logs/lipschitz_3layer_nosave_pstats.txt

profiles/lipschitz_big_nosave_both_vectorized.prof
logs/lipschitz_big_nosave_both_vectorized_pstats.txt

logs/correctness_results.csv
logs/benchmark_results.csv

logs/vectorization_current_state.diff
logs/vectorization_pre_selector_state.diff
```

Additional profile text files mentioned:

```text
Section_5_4/cprofile_3layer_vectorized.txt
Section_5_4/cprofile_3layer_both_vectorized.txt
```

Exact current presence of every historical text artifact should be confirmed from the repository before relying on it, but they were reported during this chat.

## Analysis documents

```text
results/optimization_section_5_4/BIG_PROFILE_ANALYSIS.md
results/optimization_section_5_4/LSTSQ_ANALYSIS.md
results/optimization_section_5_4/CACHING_ANALYSIS.md
results/optimization_section_5_4/GPU_READINESS.md
```

---

# 26. Git / Diff State

A repository snapshot step was performed.

Reported behavior:

- `git status` and `git diff` were inspected;
- current sprint modifications were preserved;
- unrelated earlier Section 5.2 / 5.3 / 5.5 result changes were not touched.

An older diff snapshot was found to be stale.

It was archived as:

```text
logs/vectorization_pre_selector_state.diff
```

A newer current snapshot was written to:

```text
logs/vectorization_current_state.diff
```

At the reported point, the main sprint diff involved approximately:

```text
Section_5_4/get_lipschitz.py
Section_5_4/lipschitz.sh
forward_mode_tensorized_src/precise_transformer.py
```

plus untracked tests / experiment artifacts.

Do not assume this remains the exact Git state in a future session; run `git status` before changing anything.

---

# 27. Key Commands Used

## Run Section 5.4 network

```powershell
python .\get_lipschitz.py --network 3layer --no-save
```

```powershell
python .\get_lipschitz.py --network big --no-save
```

## Profile 3layer compute-only

```powershell
python -m cProfile `
    -o ..\profiles\lipschitz_3layer_nosave.prof `
    .\get_lipschitz.py --network 3layer --no-save
```

## Profile optimized big

```powershell
python -m cProfile `
    -o ..\profiles\lipschitz_big_nosave_both_vectorized.prof `
    .\get_lipschitz.py --network big --no-save
```

## Create profile/log directories

From `Section_5_4`:

```powershell
New-Item -ItemType Directory -Force ..\profiles
New-Item -ItemType Directory -Force ..\logs
```

Important path note:

```text
..\profiles
..\logs
```

are inside `Pasado`.

Using:

```text
../../profiles
```

from `Section_5_4` would go outside the Pasado repo.

## Extract hotspot source definitions

Command used:

```powershell
Select-String `
  -Path ..\forward_mode_tensorized_src\precise_transformer.py `
  -Pattern "def sigmoid_prime_product_tensor|def compute_max_error|def objective_fn|def check_nonlinear_boundary|def check_corners|def lin_reg_tensor|def PreciseSigmoidDualZonotope" `
  -Context 0,90
```

## pstats pattern

A helper script used this structure:

```python
import pstats

prof = "../profiles/<profile>.prof"
out = "../logs/<profile>_pstats.txt"

with open(out, "w") as f:
    stats = pstats.Stats(prof, stream=f)
    stats.strip_dirs().sort_stats("cumulative").print_stats(150)
    f.write("\n\n--- SORTED BY SELF TIME ---\n\n")
    stats.strip_dirs().sort_stats("time").print_stats(150)
```

---

# 28. What Has Been Proven vs What Has Not

## Proven / measured

### **[VERIFIED]**
- `check_corners_tensor` reproduces the old corner logic.
- `check_nonlinear_boundary_tensor` reproduces the old boundary logic.
- dtype promotion bug was found and fixed.
- reduced end-to-end old/new correctness is extremely close:
  - Lipschitz bound `<= 5.7e-14`
  - centers/generators/lb/ub `<= 4.5e-14`
- selector paths call the intended implementations.
- batched `gelsd` can reproduce repeated individual `gelsd` regression results in the tested cases.
- caching exact design matrices is not useful in the measured sample.

### **[MEASURED]**
- vectorizing the two error-check paths reduces `3layer` cProfile time:
  - `45.55s -> 24.97s`
- paired later `3layer` benchmark:
  - `140.3s -> 75.0s`
  - `1.87x`
  - `46.5% runtime reduction`
- `compute_max_error`:
  - `19.74s -> 1.97s` in cProfile
- optimized `big --no-save` observed:
  - `18m15s`
- optimized big cProfile:
  - ~20m37 terminal elapsed
  - pstats ~1249s / ~61M calls
- `AffineZonotope` is the largest single big-profile hotspot after vectorization.
- there are approximately:
  - `96,000` small regressions for 3layer
  - `1,966,080` for big.

## Implemented but not yet fully production-benchmarked

### **[IMPLEMENTED EXPERIMENTALLY]**
- `lin_reg_tensor_batched` in `experiments/batched_lin_reg.py`
- tests for batched regression
- `PASADO_BATCHED_LSTSQ` selector infrastructure

The batched implementation had not yet been fully wired/benchmarked in the main precise path at the last confirmed state.

## Prepared only

### **[PREPARED]**
- `benchmark_lstsq.py`
- `compile_experiment.py`
- `torch_profiler_compare.py`
- `colab_gpu_profile.py`

No final results from these prepared scripts should be invented.

## Not done

### **[PLANNED]**
- direct JAX implementation / JAXPR frontend
- direct Neptune integration
- actual GPU speedup result
- final `torch.compile` result
- final batched-lstsq end-to-end speedup
- final fair old-vs-new `big` result after the benchmark harness finishes
- compatibility/outdated-versions PR
- complete codebase walkthrough for personal understanding

---

# 29. Exact Point Where Work Stopped

This is the current continuation point.

A long benchmark harness was running in the background to obtain fair old-vs-new timings.

The `3layer` portion had finished:

```text
original median = 140.3s
vec_both median = 75.0s
n = 3
speedup ~1.87x
runtime reduction ~46.5%
```

The `big` old-vs-new single-run comparison was still running / had not yet produced final numbers in the chat.

While that run was active, work intentionally avoided starting CPU-heavy microbenchmarks that could contaminate measurements.

Meanwhile, all non-heavy analysis work was completed:

- big profile analysis;
- selector verification;
- dtype bug fix;
- end-to-end correctness;
- LSTSQ shape/call-count analysis;
- experimental batched regression implementation;
- batched-regression correctness tests;
- caching analysis;
- GPU readiness audit;
- preparation of compiler/profiler/GPU scripts.

The explicitly agreed continuation sequence was:

1. let the active `big` old-vs-new benchmark finish;
2. record the fair big numbers in `logs/benchmark_results.csv`;
3. run `experiments/benchmark_lstsq.py` on an otherwise idle machine;
4. if the batched solver passes the performance gate, wire `lin_reg_tensor_batched` into `forward_mode_tensorized_src/precise_transformer.py` behind `PASADO_BATCHED_LSTSQ`;
5. rerun relevant unit tests and reduced end-to-end correctness;
6. benchmark and cProfile `3layer` with batched LSTSQ;
7. only if that result is clean and useful, run one `big` batched-LSTSQ wall-clock test;
8. only after finishing this manual batching stage, run the already-prepared `torch.compile` / `torch.profiler` experiments and later GPU/JAX work.

This sequence was already agreed in the chat; it is not a new roadmap invented by this document.

---

# 30. High-Level Interpretation for Future Sessions

The most important conceptual result so far is:

> Pasado’s precise transformer was not slow only because the mathematical analysis is expensive. A substantial fraction of its cost came from executing structurally identical, tiny tensor problems one neuron/component at a time through Python.

The first optimizations converted two major per-neuron error-checking loops into batched PyTorch tensor code.

That change:

- preserved the analysis behavior;
- exposed a subtle dtype-equivalence issue and fixed it;
- reduced `compute_max_error` from a major hotspot to a minor one;
- produced a reproducible paired `3layer` speedup of about `1.87x`.

After that optimization, the profile changed:

```text
before:
precise error checking / Python orchestration dominates

after:
large zonotope matmul/reductions + millions of small lstsq calls dominate
```

On `big`, the largest single block is now real dense zonotope linear algebra (`AffineZonotope`), while the largest clearly batchable block is the ~1.97M small regression calls.

The next batching experiment has already shown that the regression systems have uniform shapes and that batched `gelsd` can be bit-identical in the tested cases. The missing piece is end-to-end integration/performance measurement.

---

# 31. Research Directions Discussed but Not Yet Results

For future context only, the following directions were explicitly discussed:

### PyTorch Compile / TorchInductor

The original code was a poor compile target because it had:

- Python lists;
- custom zonotope objects;
- per-neuron loops;
- many small tensor constructions;
- NumPy calls;
- `.item()`.

The manual tensorization/batching work is intended to produce cleaner pure-tensor kernels that can be meaningfully tested with `torch.compile`.

No compile speedup has been established yet.

### GPU

The `big` profile suggests GPU-relevant work:

- dense generator matrix multiplies;
- abs/sum reductions;
- wide 1024-dimensional layers.

But:

- float64 performance may be poor on consumer GPUs;
- `.item()` creates synchronization hazards;
- `gelsd` does not mechanically map to the CUDA LSTSQ path.

No GPU speedup has been established yet.

### JAX / XLA

JAX is a possible future backend/frontend direction because of:

- `vmap` / batching;
- `jit` / XLA;
- natural relationship to automatic differentiation;
- possible future JAXPR input to Pasado.

No JAX implementation exists yet.

### Neptune

Neptune has only been used conceptually as background for:

- fusion;
- locality;
- minimizing intermediate materialization.

No Neptune code/result has been used directly in this Pasado optimization work.

---

# 32. Resume-Level Result Already Extracted from This Work

For context, the project result was summarized in the CV as:

```text
Optimized Pasado's precise zonotope analysis, achieving a 1.87x speedup
while preserving numerical equivalence; extending the work toward
compiler-assisted batching, fusion, and GPU execution.
```

This is a high-level summary only. The technical details in this document are the source of truth for future project conversations.

---

# 33. Future-AI Continuation Instructions

When this document is loaded into a future Claude/ChatGPT Project and the user says:

> “Nastavljamo odavde.”

do **not** restart from basic Pasado explanations or rerun already completed experiments without reason.

First establish the current repo/runtime state with lightweight checks such as:

```text
git status
git diff
whether the active big benchmark completed
latest rows in logs/benchmark_results.csv
whether PASADO_BATCHED_LSTSQ is actually wired
```

Then continue from the “Exact Point Where Work Stopped” section.

Do not assume that prepared experiments have run simply because their scripts exist.

Do not claim Neptune, JAX, GPU, or `torch.compile` results until they are actually measured.

Do not mix early uncontrolled wall-clock times with later paired benchmark-harness times.

Preserve the distinction between:

```text
manual tensorization/batching results
compiler experiments
GPU experiments
future research framing
```

The current strongest confirmed performance result is the paired `3layer` vectorization result:

```text
140.3s -> 75.0s median
~1.87x speedup
~46.5% runtime reduction
```

with end-to-end numerical equivalence validated to approximately `1e-14` scale in the reduced correctness harness.
