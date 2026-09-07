# UIUC Research Status — Week 1

## Artifact Reproducibility and Modernization Pass

### Goal

Before diving into a deeper exploration of optimization opportunities and performance bottlenecks in **Pasado** and **Neptune**, I first assess the current state of both research artifacts. The goal is to determine whether the projects still build and run successfully, whether their documented dependencies remain compatible with modern Python/CUDA/toolchain versions, and whether small modernization changes are needed before meaningful profiling or compiler optimization experiments can begin.

This first pass focuses on:

- reproducing the existing tests and experiment scripts;
- identifying scripts that fail under newer dependency versions;
- distinguishing local setup issues from genuine artifact compatibility issues;
- collecting small modernization opportunities that could become GitHub issues or pull requests;
- preparing the next step: profiling Pasado and evaluating whether compiler-based optimizations such as `torch.compile`, TorchInductor, TVM, Triton, or Neptune-style fusion are relevant.

---

# Pasado

## Current Status

I performed an initial reproducibility pass over the Pasado artifact by running the available unit tests and Section 5 experiment scripts. Most of the core tests and experiments completed successfully. The failures I observed appear to be small compatibility issues caused by newer dependency versions, especially NumPy 2.0 behavior changes, rather than fundamental problems in the artifact.

## Tests and Scripts Run

### Passing Tests / Scripts

The following tests and experiment scripts completed successfully with exit code `0`:

| Area | Command / Script | Result | Notes |
|---|---|---:|---|
| Forward-mode non-tensorized tests | `forward_mode_non_tensorized_src/tests/*.py` | Pass | 4 tests passed |
| Reverse-mode non-tensorized tests | `reverse_mode_non_tensorized_src/tests/*.py` | Pass | 3 tests passed |
| Section 5.2 | `climate.sh` | Pass | Completed successfully |
| Section 5.2 | `chemical.sh` | Pass | Completed successfully |
| Section 5.3 | `black_scholes.sh` | Pass | Completed successfully |
| Section 5.4 | `lipschitz.sh` | Pass | FFNN Lipschitz experiment completed successfully |
| Section 5.5 | `adult_script.py` | Pass | Completed successfully |

### Failing Tests / Scripts

The following scripts failed with exit code `1`:

| Area | Command / Script | Result | Likely Cause |
|---|---|---:|---|
| Section 5.4 | `lipschitz_cnn.sh` | Fail | `np.product` was removed in NumPy 2.0 |
| Section 5.5 | `adult_plot.py` | Fail | NumPy 2.0 representation change breaks brittle integer parsing |

---

## Observed Issues and Possible Fixes

### 1. NumPy 2.0 compatibility issue in CNN Lipschitz experiment

**Failing script:**

```bash
Section_5_4/lipschitz_cnn.sh
```

**Failure cause:**

The script fails because `np.product` was removed in NumPy 2.0. The failing usage appears in:

```text
forward_mode_tensorized_src/conv.py:38
```

**Suggested fix:**

Replace:

```python
np.product(...)
```

with:

```python
np.prod(...)
```

**Why this matters:**

This is a small compatibility issue with modern NumPy versions. The fix is local, low-risk, and does not appear to change the intended semantics.

**Potential GitHub issue title:**

```text
NumPy 2.0 compatibility: np.product removed in forward_mode_tensorized_src/conv.py
```

---

### 2. Brittle integer parsing in `adult_plot.py`

**Failing script:**

```bash
Section_5_5/adult_plot.py
```

**Failure location:**

```text
Section_5_5/adult_plot.py:21
```

**Failure cause:**

The script appears to rely on a brittle string representation of numeric values. Newer NumPy versions changed the representation, which breaks the parsing logic.

**Suggested fix:**

Use a more robust parsing strategy, such as:

- regex-based parsing of the numeric component;
- explicit integer casting before string conversion;
- avoiding reliance on NumPy-specific `repr` output.

**Potential GitHub issue title:**

```text
adult_plot.py fails with newer NumPy repr due to brittle integer parsing
```

---

### 3. Missing machine-readable dependency file

Pasado currently documents dependencies in a README table. This is useful for understanding the original artifact environment, but a machine-readable dependency file would make setup easier and more reproducible.

**Suggested improvement:**

Add one of the following:

```text
requirements.txt
```

or:

```text
pyproject.toml
```

Neptune already uses `pyproject.toml`, so Pasado could potentially follow a similar approach.

**Recommended approach:**

The README should distinguish between:

1. the original artifact environment;
2. a tested modern environment;
3. known incompatible versions.

This avoids replacing the original reproduction environment while still documenting which newer dependency versions are known to work.

**Potential GitHub issue title:**

```text
Add machine-readable dependency file for reproducible setup
```

---

### 4. Shell scripts require robust `PYTHONPATH` setup

Some Section 5.2 scripts require the correct source directory to be visible through `PYTHONPATH`. A robust fix for scripts such as `climate.sh` and `chemical.sh` would be to add:

```bash
export PYTHONPATH="$(cd "$(dirname "$0")/../forward_mode_non_tensorized_src" && pwd):$PYTHONPATH"
```

**Why this matters:**

This makes the scripts less dependent on the current working directory and local shell environment.

**Potential GitHub issue title:**

```text
Section 5.2 scripts require explicit PYTHONPATH for forward_mode_non_tensorized_src
```

---

### 5. scikit-learn pickle version warning

Some pretrained models appear to have been pickled with scikit-learn `1.3.0`. Loading them with a newer scikit-learn version produces an `InconsistentVersionWarning`.

**Observed behavior:**

- The warning does not crash the script.
- However, it may risk silently different numerical behavior when using newer scikit-learn versions.

**Suggested improvement:**

Document this explicitly in the README or dependency notes.

**Potential GitHub issue title:**

```text
Document scikit-learn version warning when loading pretrained models
```

---

## Suggested Pasado Modernization Summary

| Issue | Type | Severity | Suggested Action |
|---|---|---:|---|
| `np.product` removed in NumPy 2.0 | Compatibility bug | Medium | Replace with `np.prod` |
| `adult_plot.py` brittle parsing | Compatibility bug | Low/Medium | Use regex or explicit casting |
| Missing dependency file | Reproducibility enhancement | Medium | Add `requirements.txt` or `pyproject.toml` |
| Manual `PYTHONPATH` assumptions | Reproducibility enhancement | Low/Medium | Export source path inside scripts |
| scikit-learn pickle version skew | Documentation issue | Low | Document original sklearn version and warning |

---

# Neptune

## Current Status

Neptune testing is currently in progress. The goal is to perform the same reproducibility pass as for Pasado: first checking whether the artifact can be set up, imported, and benchmarked, then documenting any dependency or toolchain issues.

## Planned Checks

| Check | Command / Script | Result | Notes |
|---|---|---:|---|
| GPU visibility | `nvidia-smi` | TBD | Confirm that the host GPU is visible |
| Docker GPU support | `docker run --gpus all ... nvidia-smi` | TBD | Confirm Docker can access the GPU |
| Neptune environment setup | TBD | TBD | Docker or source build |
| Python import | `python -c "import tvm.neptune as neptune"` | TBD | Basic import sanity check |
| Basic benchmark | `python -m scripts.neptune_bench profile prefill_global 1,8192` | TBD | Initial benchmark from README |
| Test suite | TBD | TBD | Record pass/fail status |

## Neptune Notes to Collect

For each setup or benchmark attempt, I will record:

- repository commit hash;
- setup method;
- OS;
- GPU model;
- NVIDIA driver version;
- CUDA version;
- Python version;
- Docker version, if applicable;
- exact commands used;
- pass/fail status;
- full traceback for failures;
- whether the issue appears to be local setup-related or artifact-related.

---

# Next Steps

## Immediate Next Steps

1. Finish the Neptune reproducibility pass.
2. Save full logs for failing commands.
3. Record exact environment information for both Pasado and Neptune.
4. Prepare one short reproducibility report for discussion.
5. Decide which Pasado issue is clean enough to open first on GitHub.

## Recommended First GitHub Issue / PR Candidate

The cleanest first Pasado issue appears to be:

```text
NumPy 2.0 compatibility: replace np.product with np.prod
```

This issue is small, easy to reproduce, and likely fixable with a minimal patch.

## Next Research Step After Reproducibility

After the artifact status is documented, the next research step should be profiling Pasado, starting with the Section 5.4 Lipschitz benchmark.

The goal of profiling is to determine whether Pasado contains tensor-heavy or memory-heavy computation patterns that could plausibly benefit from compiler-level optimizations such as:

- `torch.compile`;
- TorchInductor;
- Triton kernels;
- TVM schedules;
- XLA-style fusion;
- Neptune-style operator fusion.

The first profiling questions should be:

1. Where does Pasado spend most of its runtime?
2. Are the bottlenecks Python loops, NumPy operations, PyTorch tensor operations, file I/O, plotting, or model loading?
3. Are there repeated intermediate tensors that could benefit from fusion?
4. Can `torch.compile` capture any of the tensorized computation?
5. If `torch.compile` fails, what causes the graph breaks?
6. Is there a small recurring hotspot that could be isolated as a Triton/TVM microbenchmark?

---

# Questions for Advisor

1. Should I treat Pasado primarily as the workload to optimize, with Neptune/TorchInductor/TVM/Triton as optimization inspiration and tools?
2. Which Pasado benchmark should I profile first: FFNN Lipschitz, CNN Lipschitz, ODE sensitivity, financial models, or Adult Income?
3. Should the first contribution focus on artifact modernization and GitHub issues, or should I move quickly toward profiling?
4. Is `torch.compile` / TorchInductor a good first optimization path, or should I look directly at TVM/Triton/Neptune-style kernels?
5. Should I aim for a small PR first, such as the NumPy 2.0 compatibility fix?
6. What level of result would be useful for a potential research writeup: profiling report, compiler-friendliness analysis, or an actual fused-kernel speedup?

---

# Short Summary

The initial Pasado reproducibility pass shows that most tests and Section 5 scripts still run successfully. The main issues found so far are small modernization problems related to newer dependency versions, especially NumPy 2.0 compatibility and brittle parsing in plotting scripts. Pasado would also benefit from machine-readable dependency metadata and more robust script-level `PYTHONPATH` handling.

Neptune testing is currently in progress. Once both artifacts are documented, the next step is to move from reproducibility to profiling, starting with Pasado’s Section 5.4 Lipschitz benchmarks, and then evaluate whether modern compiler techniques such as TorchInductor, TVM, Triton, or Neptune-style fusion are relevant for optimizing Pasado-style AD/static-analysis workloads.
