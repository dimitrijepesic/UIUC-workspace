# Known Issues (from a full local re-run, 2026-07-08)

Ran via the prebuilt Docker image `evanzhao16/neptune-env:nvidia-latest` (GPU passthrough through
Docker Desktop on Windows, WSL2 backend). Local GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`, 4GB VRAM.
Native build from source was not attempted — no C++/CUDA/LLVM toolchain set up on this Windows
machine, and Neptune/TVM+CUDA native builds are not well supported on Windows anyway.

## Results

**Sanity import**: Pass — `python3 -c "import tvm.neptune as neptune"` succeeds, GPU visible
inside the container (`nvidia-smi` reports 3964/4096 MiB free).

**All 10 attention operators** (`prefill_global`, `prefill_causal`, `prefill_gqa`, `prefill_alibi`,
`prefill_softcap`, `prefill_windowed`, `decode_causal`, `decode_gqa`, `decode_alibi`,
`decode_softcap`), run via:
```bash
python3 -m scripts.neptune_bench profile <operator> 1,1024
```
**Pass (exit 0) for all 10** — but only after two local fixes described below. Without them,
every operator fails immediately.

## Root causes (not local setup issues — reproducible from the actual sources)

### 1. `GPU_INFO` allowlist in `scripts/neptune_bench/utils.py:99-104` is hardcoded to 4 specific GPUs

```python
GPU_INFO = {
    "NVIDIA RTX A5000": ("nvidia/rtx-a6000", 1695, 8000),
    "NVIDIA RTX 6000 Ada Generation": ("nvidia/rtx-a6000", 2505, 9500),
    "NVIDIA A100-SXM4-40GB": ("nvidia/nvidia-a100", 1410, None),
    "NVIDIA H100 PCIe": ("nvidia/nvidia-h100", 1755, None),
}
...
target_name, gpu_clock, mem_clock = GPU_INFO[device_name]  # KeyError on any other GPU
```
Any GPU not in this list (including our RTX 3050 Laptop) hits a `KeyError` in
`get_current_gpu_info()` before a single kernel is compiled or run. This reproduces on the
current GitHub `main` branch too — it's a genuine portability gap for anyone without one of
these four specific cards, not a Docker/environment artifact.

**Local workaround** (container only, nothing committed): added an entry mapping
`"NVIDIA GeForce RTX 3050 Laptop GPU"` to `("nvidia/geforce-rtx-3060", 1500, 7001)` — same Ampere
architecture / compute capability 8.6, closest matching TVM target tag available
(`tvm.target.tag.list_tags()` has no exact `rtx-3050` entry).

**Suggested real fix**: fall back to a best-effort target lookup by compute capability /
`torch.cuda.get_device_properties()` when the exact device name isn't in the table, instead of a
hard `KeyError`, with a warning that clock-locking (needed for stable benchmark numbers) is
unavailable for unrecognized GPUs.

### 2. The `evanzhao16/neptune-env:nvidia-latest` Docker image is stale relative to `main`

`scripts/neptune_bench/__main__.py:77` inside the image reads:
```python
*nep.NeptuneGQARunner.e(
```
instead of:
```python
*nep.NeptuneGQARunner.create_flex_from_schedulers(
```
This breaks `prefill_softcap` (and anything constructed after it in the `RUNNERS` dict) with an
`AttributeError`. Confirmed via `git log`/`git diff` inside the container: the image is checked
out at commit `3aa55c12a` ("Accumulative squashed commit for Neptune over TVM v0.18"), and this
exact typo was already fixed in a later commit that's part of current `main`
(`git diff 3aa55c12a HEAD -- scripts/neptune_bench/__main__.py` shows precisely this one-line fix).
The Docker image simply hasn't been rebuilt/republished since that fix landed.

**Local workaround** (container only): `sed -i` to restore `create_flex_from_schedulers`.

**Suggested real fix**: rebuild and republish the `neptune-env:nvidia-latest` image from current
`main`.

## Non-fatal but worth knowing

- For every operator, Neptune's `*-tuned` scheduler variants (`sch-*-tuned`) are skipped with
  "does not support this setup" — expected, since the image doesn't ship autotuning logs
  (`logs/neptune-tuning/...`). Tuning is a separate opt-in step (`neptune_bench tune`) per the
  README; not a bug.
- For 5 of the 10 operators (`prefill_global`, `prefill_causal`, `prefill_gqa`, `prefill_alibi`,
  `decode_alibi`) the harness logged a "result mismatch" warning (tolerance `atol=1e-2, rtol=0`).
  In every case the mismatch was between two *third-party baselines* (mainly the Tri Dao Triton
  reimplementation, and PyTorch's FlexAttention on the ALiBi variant) against the reference
  output — not Neptune's own kernel (`sch-*-manual`) failing. Wherever Neptune's manual schedule
  ran, its output matched the reference. Worth a closer numeric look eventually, but doesn't look
  like a Neptune-side correctness bug from this pass.
- The image is ~15.6GB on the wire / ~43GB on disk once extracted — worth knowing before pulling
  on a constrained connection or disk.
- The initial `docker pull` failed once with `short read: ... unexpected EOF` on a ~6.2GB layer
  (transient network interruption); a plain retry succeeded and reused the already-pulled layers.

## Not run

- Building Neptune from source (native Windows toolchain not set up; not attempted given the
  prebuilt image was available and functional).
- `scripts/neptune_bench` with `--profiler ncu/nsys` (no Nsight tooling installed) or `tune` mode
  (autotuning; would need meaningful wall-clock time and isn't needed to answer "does it work").
- Larger shapes (e.g. the README's `1,8192` example) — deliberately used a smaller `1,1024` shape
  throughout given the 4GB VRAM budget on this laptop GPU; larger shapes were not attempted and
  may OOM.
- The inherited TVM test suite under `tests/python/**` (1192 files) — out of scope; almost all of
  it exercises TVM backends/hardware unrelated to Neptune's actual contribution (Hexagon, Ethos
  NPU, Android, etc.).

## Reference: local commands run (nothing committed to the repo)

```bash
docker pull evanzhao16/neptune-env:nvidia-latest
docker run --gpus all -d --name neptune-test evanzhao16/neptune-env:nvidia-latest tail -f /dev/null
docker exec neptune-test bash -lc ". /workspace/venv/bin/activate && cd /workspace/neptune && python3 -c 'import tvm.neptune as neptune'"
# local fix 1: added a GPU_INFO entry in scripts/neptune_bench/utils.py (container only)
# local fix 2: sed -i fix for the .e( typo in scripts/neptune_bench/__main__.py (container only)
docker exec neptune-test bash -lc ". /workspace/venv/bin/activate && cd /workspace/neptune && python3 -m scripts.neptune_bench profile <operator> 1,1024"  # x10, one per operator
```
