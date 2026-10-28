"""Builds COLAB_COMPILED_PRECISE_SIGMOID.ipynb (workspace root) around colab_harness.py.

The harness goes into the notebook verbatim through %%writefile, so the notebook runs
exactly the file that was tested locally (colab_harness.py --smoke --device cpu).
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(os.path.dirname(HERE))
NOTEBOOK = os.path.join(WORKSPACE, "COLAB_COMPILED_PRECISE_SIGMOID.ipynb")
HARNESS = os.path.join(HERE, "colab_harness.py")


def md(text):
    return dict(cell_type="markdown", metadata={}, source=text.strip("\n").splitlines(keepends=True))


def code(text):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=text.strip("\n").splitlines(keepends=True))


INTRO = r"""
# Pasado `gpu-hybrid`: ceo `precise_sigmoid` kompajliran TorchInductor-om na A100 (v2)

**Verzija 2.** Prvi run (v1, harness `df0e88f5…`) zaustavio je gate pre merenja vremena: 868/876 provera, 4 pada G4 i 4 pada G5 (Pasado `results/compiled_sigmoid_cuda_2026-10-28_gate_stop/`, commit `1ac62f9`). v2 menja tačno dve stvari, nijedna tolerancija nije podignuta:
1. G4: generatorska dimenzija G0 se pre svakog kompajliranog poziva označava kao dinamička (`torch._dynamo.mark_dynamic`), pa jedan graf služi sve slojeve od prvog prolaza; G4 ostaje "isti seed, isti bitovi", G2 dodaje "tačno jedan graf po (mreža, S)".
2. G5: `dg` se ocenjuje po blokovima (generatori i nule do 1e-12 normwise), a D ugovorom G1 iz `compare_lstsq_solvers.py` (mpmath, manjak relativno na W) na unapred fiksiranom uzorku boksova pri S = 64, eps 8, sa (A, B, C) i boksom koje je izračunao baš kompajlirani graf.

Pre pokretanja: **Runtime > Change runtime type > GPU (A100)**, zatim **Runtime > Disconnect and delete runtime**, pa **Runtime > Run all**.

Eksperiment, bez izmene koda biblioteke: Pasado na commitu `a57b560`. Režimi se biraju spolja (zastavice modula, i `batched_splits.precise_sigmoid` zamenjen kompajliranom funkcijom). Sve u `float64`, seed 7:

1. `sequential_G`: sekvencijalna petlja po splitovima, `gelsd` na CPU-u;
2. `batched_G`: batched prolaz, `gelsd` (podrazumevani put);
3. `batched_A2_eager`: batched, eager `fit_planes`;
4. `batched_A2_compiled_fit`: batched, kompajliran samo `fit_planes` (`PASADO_COMPILED_FIT`, proizvodni omotač);
5. `batched_compiled_sigmoid`: `torch.compile(precise_sigmoid, fullgraph=True)` sa eager A2 unutra. Bez ugnježdene kompilacije.

Mreže 3layer, 4layer, 5layer, big; S = 1, 8, 32, 64; 3 slike; eps indeksi 0 i 8. `big` ide u chunk-ovima od 32 splita na 40 GB, 64 na 80 GB.

Redom:
1. **Korektnost (gate)**, G1-G8 na celoj mreži konfiguracija, nov kompajlirani callable po (mreža, S). Ako bilo šta padne, sveska preuzima zip i staje.
2. **Rekompilacije**: po (mreža, S) iz gate-a (grafovi, oblik koji je izazvao kompilaciju, dinamičke dimenzije, limit); zatim jedan deljeni callable preko cele mreže sa podrazumevanim limitom (ne povećava se).
3. **Vreme**: (a) hladan i topao disk keš u zasebnim procesima za svaku (mreža, S, kompajliran režim), 64 procesa; (b) steady state svih 5 režima u jednom procesu, naizmenično, uz kernele po sloju i peak VRAM.
4. **Compiler logovi**: `TORCH_LOGS="graph_breaks,recompiles,fusion,output_code"` postavljen pre pokretanja procesa, plus generisani kod iz keša.
5. **Kriterijumi**, primenjeni mehanički (`summarize`).
6. Zip.

**Pre-registrovano** (puna verzija na engleskom je u zaglavlju harness-a, ćelija 4; sveska beleži njegov sha256 u `colab_environment.json`):
- C1: korektnost je obavezan gate. G1-G8 se proveravaju pre merenja vremena; uz njih ulaze i provere steady procesa (finalni bound svakog režima naspram sekvencijalnog, kompajlirani regioni po prolazu, `lstsq` samo u G režimima, bez fallback-a) i start procesa (bez fallback-a, bar jedan graf, bound naspram sekvencijalnog). Pad bilo koje daje ishod `stop`.
- C2: ceo kompajliran `precise_sigmoid` nije sporiji od compiled-fit-only puta za više od 5% (medijana od 30 uzoraka, u svakoj (mreža, S)).
- C3: cilj je bar 1.2× naspram compiled-fit-only na 3layer, 4layer i 5layer pri S = 1.
- C4: hladan start i break-even se prijavljuju odvojeno (hladan i topao keš) i ne kriju se iza warm rezultata.
- Ishod: `stop` (C1 pada); `opt-in` (C2 ili C3 pada, ili je topli break-even naspram najboljeg puta bez kompilacije veći od 480 prolaza, jednog standardnog sweep-a od 30 slika × 16 eps, za neku malu mrežu pri S = 1); `auto candidate` (C1-C3 važe i taj topli break-even je najviše 480); `default candidate` (i hladni je najviše 480); `incomplete` (nedostaje konfiguracija).

Očekivano trajanje: oko 45 do 80 minuta (procena; v1 gate je trajao 7 minuta, najviše odlazi na 64 start procesa). Kad završi, pošalji `compiled_sigmoid_v2_cuda.zip`.
"""

GPU_CHECK = r"""
import json
import os
import subprocess
import sys

subprocess.run(["nvidia-smi"], check=True)
query = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
GPU, VRAM_MIB, DRIVER = [s.strip() for s in query.split(",")]
VRAM_GIB = float(VRAM_MIB) / 1024
print("GPU:", GPU, "| VRAM GiB:", round(VRAM_GIB, 1), "| driver:", DRIVER)
REQUIRE_A100 = True
assert "A100" in GPU or not REQUIRE_A100, f"Merenje je dogovoreno na A100, a ovo je {GPU}."
# CUDA runs only in the subprocesses; this kernel never initialises it
"""

CLONE = r"""
import shutil

REPO = "/content/Pasado"
URL = "https://github.com/dimitrijepesic/Pasado.git"
COMMIT = "a57b560"

os.chdir("/content")
shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(["git", "-c", "http.version=HTTP/1.1", "clone", "--branch", "gpu-hybrid",
                "--single-branch", URL, REPO], check=True)
subprocess.run(["git", "-C", REPO, "checkout", "-q", COMMIT], check=True)
head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
assert head.startswith(COMMIT), f"HEAD je {head}, a ocekivan je {COMMIT}"
assert not subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"], text=True).strip()
print("HEAD:", head)
"""

PIP = "%pip install -q scikit-learn mpmath"

SETUP = r"""
import hashlib
import math
import shlex
import time
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from IPython.display import display

HARNESS = "/content/compiled_sigmoid_harness.py"
OUT = "/content/compiled_sigmoid_v2_cuda"
LOGS = f"{OUT}/logs"
CACHES = "/content/inductor_caches"
shutil.rmtree(OUT, ignore_errors=True)
shutil.rmtree(CACHES, ignore_errors=True)
os.makedirs(LOGS)
shutil.copy2(HARNESS, OUT)
with open(HARNESS) as f:  # trailing newlines normalised: %%writefile may add one
    HARNESS_SHA256 = hashlib.sha256((f.read().rstrip("\n") + "\n").encode()).hexdigest()
EXPECTED_SHA256 = "@@HARNESS_SHA256@@"
assert HARNESS_SHA256 == EXPECTED_SHA256, "harness u ćeliji 4 nije pre-registrovana verzija"
# as in COLAB_V3_COMPILED: about 0.5 GiB per split of big, at most 60% of the VRAM
BIG_CHUNK = min(64, 2 ** int(math.log2(max(1, 0.6 * VRAM_GIB / 0.5))))
GATE_CACHE = f"{CACHES}/gate"
T0 = time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("harness sha256:", HARNESS_SHA256, "| big chunk:", BIG_CHUNK)


def harness(command, name, *extra, cache=None, torch_logs=None, echo=True):
    # One harness subcommand in a new Python process. stdout goes to the screen (echo) and to
    # logs/<name>.stdout.log, stderr (where TORCH_LOGS writes) to logs/<name>.stderr.log.
    env = dict(os.environ, PASADO_REPO=REPO, PYTHONUNBUFFERED="1")
    for key in ("TORCH_LOGS", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        env.pop(key, None)
    if torch_logs:
        env["TORCH_LOGS"] = torch_logs  # set before the process starts
    if cache:
        env["TORCHINDUCTOR_CACHE_DIR"] = f"{cache}/inductor"
        env["TRITON_CACHE_DIR"] = f"{cache}/triton"
    cmd = " ".join(shlex.quote(str(a)) for a in
                   [sys.executable, HARNESS, command, "--out", OUT, "--big-chunk", BIG_CHUNK, *extra])
    sink = f"| tee {LOGS}/{name}.stdout.log" if echo else f"> {LOGS}/{name}.stdout.log"
    script = f"set -o pipefail\n{cmd} 2> {LOGS}/{name}.stderr.log {sink}\n"
    return subprocess.run(["bash", "-c", script], cwd=REPO, env=env).returncode


def show(path, fn=lambda df: df):
    # A table from OUT, or the reason it is missing; never stops the notebook.
    try:
        display(fn(pd.read_csv(f"{OUT}/{path}")))
    except Exception as e:
        print(f"{path}: {type(e).__name__}: {e}")


def package(**extra):
    # colab_environment.json, then everything under OUT in one zip, downloaded.
    from google.colab import files
    probe = ("import json, torch\n"
             "try:\n    import triton; tv = triton.__version__\nexcept Exception:\n    tv = None\n"
             "print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, triton=tv)))")
    try:
        info = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    except Exception as e:
        info = dict(versions_error=f"{type(e).__name__}: {e}")
    environment = dict(gpu=GPU, vram_gib=VRAM_GIB, driver=DRIVER, python=sys.version.split()[0], commit=head,
                       harness_sha256=HARNESS_SHA256, big_chunk=BIG_CHUNK,
                       wall_minutes=round((time.time() - T0) / 60, 1), **info, **extra)
    with open(f"{OUT}/colab_environment.json", "w") as f:
        json.dump(environment, f, indent=2)
    archive = "/content/compiled_sigmoid_v2_cuda.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as zf:
        for root, _, names in os.walk(OUT):
            for n in sorted(names):
                path = os.path.join(root, n)
                zf.write(path, os.path.join("compiled_sigmoid_v2_cuda", os.path.relpath(path, OUT)))
    print(json.dumps(environment, indent=2))
    print(archive, round(os.path.getsize(archive) / 2**20, 2), "MiB")
    files.download(archive)
"""

GATE_MD = r"""
## 1. Korektnost (gate)

Jedan proces, prazan keš `GATE_CACHE` (posle ga deljeni callable i steady state koriste kao topao keš). `TORCH_LOGS="recompiles,graph_breaks,dynamic"` je postavljen pre pokretanja procesa, za odeljak 2.

Na svakoj (mreža, S, slika, eps): G1 `fullgraph=True` prolazi; G2 nema fallback-a (svaki poziv kroz kompajlirani callable, jedan "Torch-Compiled Region" po pozivu u profileru, bez `lstsq`, `compile_fallback` prazan, unutra eager `fit_planes`) i tačno jedan graf služi sve slojeve; G3 NumPy stanje kao eager A2; G4 isti seed, isti bitovi; G5 `rc`, `rg`, `dc` i blokovi `dg` (generatori, nule) kao eager do 1e-12 normwise, isti oblici i nonzero flag; pri S = 64, eps 8 još i D: dijagnostički graf bit-identičan proizvodnom, pa manjak kompajliranog D ispod mpmath maksimuma relativno na W najviše 10 × max(1e-12, manjak eager-a), po mreži, na fiksiranom uzorku (do 128 boksova po mreži, slici i sloju); G6 per-split naspram eager A2 do 1e-10; G7 finalni bound naspram sekvencijalnog G do 1e-10; G8 autograd (bound nije ispod uzorkovanog maksimuma; i bez seed-a na 3layer).

Ako bilo šta padne: zip se preuzima i sveska staje.
"""

GATE = r"""
rc_gate = harness("gate", "gate", cache=GATE_CACHE, torch_logs="recompiles,graph_breaks,dynamic")
checks_path = f"{OUT}/gate_checks.csv"
if os.path.exists(checks_path):
    checks = pd.read_csv(checks_path)
    print(checks.groupby("check").ok.agg(["count", "sum"]).rename(columns={"sum": "passed"}).to_string())
if rc_gate != 0 or not os.path.exists(checks_path):
    if os.path.exists(checks_path):
        display(checks[~checks.ok].head(50))
    package(stopped_at="gate", gate_rc=rc_gate)
    raise RuntimeError("Gate nije prošao, sveska staje. Pošalji compiled_sigmoid_v2_cuda.zip.")
"""

RECOMPILE_MD = r"""
## 2. Rekompilacije

Prvo iz gate-a: za svaku (mreža, S), sa novim callable-om posle `torch._dynamo.reset()`, broj grafova (u v2 očekivano 1), oblici `(S_chunk, G0, N)` koji su izazvali kompilaciju, dimenzije koje su postale dinamičke (`create_symbol` iz `TORCH_LOGS=dynamic`), razlozi rekompilacije i da li je dostignut limit.

Zatim jedan **deljeni** callable preko cele mreže (3layer → big, S = 1 → 64, dva kruga), sa podrazumevanim Dynamo limitom koji se ne menja, za ceo sigmoid i za compiled fit. Ako se limit dostigne, `fullgraph=True` baca grešku: to se beleži i eksperiment za taj režim staje (proizvodni compiled fit umesto toga prelazi na eager, što se vidi u koloni `new_fallback`).
"""

RECOMPILE = r"""
harness("summarize", "summarize_after_gate", echo=False)
show("recompiles.csv")
rc_shared = harness("shared", "shared", cache=GATE_CACHE, torch_logs="recompiles,graph_breaks,dynamic", echo=False)
print("shared rc:", rc_shared)
show("shared_callable.csv", lambda df: df.drop(columns=["seconds"]).assign(seconds=df.seconds.round(2)))
"""

START_MD = r"""
## 3a. Hladan i topao start

Za svaku (mreža, S, kompajliran režim) nov, prazan keš (Inductor i Triton): prvi proces je **hladan**, drugi, sa istim kešom, **topao**. Svaki proces: import, jedan eager A2 prolaz (CUDA, cuBLAS, alokator), nov kompajlirani callable, prvi sweep od 6 prolaza (3 slike × 2 eps, uključuje kompilaciju) = `e2e_s`, zatim 5 ponavljanja svake (slika, eps). `overhead_s = e2e_s - zbir 6 steady medijana` istog procesa. 64 procesa; neuspeh jednog se beleži i ne zaustavlja svesku.
"""

START = r"""
start_failures = []
for network in ["3layer", "4layer", "5layer", "big"]:
    for S in [1, 8, 32, 64]:
        for mode in ["batched_A2_compiled_fit", "batched_compiled_sigmoid"]:
            cache = f"{CACHES}/start_{network}_S{S}_{mode}"
            shutil.rmtree(cache, ignore_errors=True)
            for state in ("cold", "warm"):
                name = f"start_{network}_S{S}_{mode}_{state}"
                rc = harness("start", name, "--network", network, "--splits", S, "--mode", mode,
                             "--cache", state, cache=cache, torch_logs="recompiles,graph_breaks", echo=False)
                with open(f"{LOGS}/{name}.stdout.log") as f:
                    result = [line.strip() for line in f if line.startswith("RESULT")]
                print(result[0] if result else f"{name}: no result", f"(rc {rc})" if rc else "", flush=True)
                if rc:
                    start_failures.append(name)
            shutil.rmtree(cache, ignore_errors=True)
print("failures:", start_failures or "none")
"""

STEADY_MD = r"""
## 3b. Steady state

Jedan proces, topao keš iz gate-a. Po (mreža, S) nov kompajlirani state; svih 5 režima naizmenično, redosled se rotira svako ponavljanje; 5 ponavljanja po (slika, eps) = 30 uzoraka po (mreža, S, režim), pod `torch.compiler.set_stance("fail_on_recompile")`. Zatim peak VRAM jednog prolaza, kerneli po prolazu i po sigmoid sloju (profiler), i provere putanje (kompajlirani regioni po prolazu, `lstsq` samo u G režimima, finalni bound naspram sekvencijalnog).
"""

STEADY = r"""
rc_steady = harness("steady", "steady", cache=GATE_CACHE, torch_logs="recompiles,graph_breaks")
print("steady rc:", rc_steady)
show("steady_checks.csv", lambda df: df[~df.ok])
"""

LOGS_MD = r"""
## 4. Compiler logovi i generisani kod

`TORCH_LOGS="graph_breaks,recompiles,fusion,output_code"` se postavlja u okruženje novog procesa pre njegovog pokretanja; svaki proces ima prazan keš, iz kog se posle kopiraju generisani `.py` moduli (Triton kerneli) u `kernel_code/`. Jedan prolaz po konfiguraciji: ceo sigmoid na 3layer S = 1 i big S = 32, compiled fit na big S = 32 za poređenje.
"""

LOGS = r"""
for network, S, mode in [("3layer", 1, "batched_compiled_sigmoid"), ("big", 32, "batched_compiled_sigmoid"),
                         ("big", 32, "batched_A2_compiled_fit")]:
    name = f"compiler_{network}_S{S}_{mode}"
    cache = f"{CACHES}/logs_{network}_S{S}_{mode}"
    shutil.rmtree(cache, ignore_errors=True)
    rc = harness("logs", name, "--network", network, "--splits", S, "--mode", mode,
                 cache=cache, torch_logs="graph_breaks,recompiles,fusion,output_code")
    print(name, "rc", rc, f"stderr log {os.path.getsize(f'{LOGS}/{name}.stderr.log') / 2**20:.1f} MiB", flush=True)
    shutil.rmtree(cache, ignore_errors=True)
"""

CRITERIA_MD = r"""
## 5. Kriterijumi

`summarize` primenjuje pravila iz zaglavlja harness-a i upisuje `criteria.json`. Tabele ispod su izvedene iz sirovih CSV-ova (koji ostaju u zip-u).
"""

CRITERIA = r"""
harness("summarize", "summarize", echo=False)
try:
    with open(f"{OUT}/criteria.json") as f:
        criteria = json.load(f)
    print("ISHOD:", criteria["outcome"])
    c1 = criteria["C1_correctness"]
    print("C1_correctness", "passed" if c1["passed"] else "FAILED", "| gate failed:", c1["gate"]["failed"],
          "| steady failed:", c1["steady"]["failed"], "of", c1["steady"]["checks"],
          "| start failed:", c1["start"]["failed"], "of", c1["start"]["checks"])
    for c in c1["steady"]["failed_checks"] + c1["start"]["failed_checks"]:
        print("   ", c)
    for key in ("C2_not_slower_than_compiled_fit", "C3_gain_small_networks_S1"):
        print(key, "passed" if criteria[key].get("passed") else "FAILED")
    c4 = criteria["C4_break_even_small_networks_S1"]
    print("C4 warm", "passed" if c4["warm"]["passed"] else "FAILED", "| cold",
          "passed" if c4["cold"]["passed"] else "FAILED")
    if criteria["missing"]:
        print("missing:", criteria["missing"])
except Exception as e:
    criteria = {}
    print("criteria.json:", type(e).__name__, e)

print("steady state, median ms")
show("steady_summary.csv", lambda df: df.pivot_table(index=["network", "S"], columns="mode", values="median_ms").round(2))
print("compiled sigmoid: speedup")
show("steady_summary.csv", lambda df: df[df["mode"] == "batched_compiled_sigmoid"][
    ["network", "S", "speedup_vs_batched_A2_compiled_fit", "speedup_vs_batched_A2_eager", "speedup_vs_batched_G"]].round(2))
print("cold and warm start")
show("start_summary.csv", lambda df: df.pivot_table(index=["mode", "cache", "network"], columns="S",
                                                     values="overhead_s").round(2))
print("break-even (passes) against the best path without compilation")
show("break_even.csv", lambda df: df.pivot_table(index=["mode", "cache", "network"], columns="S",
                                                  values="break_even_vs_best_without_compile").round(0))
print("CUDA kernels per sigmoid layer (first chunk)")
show("steady_kernels.csv", lambda df: df[df.layer.astype(str) != "pass"].pivot_table(
    index=["network", "S", "layer"], columns="mode", values="kernels"))
print("CUDA kernels per pass")
show("steady_kernels.csv", lambda df: df[df.layer.astype(str) == "pass"].pivot_table(
    index=["network", "S"], columns="mode", values="kernels"))
print("peak VRAM of one pass, MiB")
show("steady_memory.csv", lambda df: df.pivot_table(index=["network", "S"], columns="mode", values="peak_mb").round(0))
"""

ZIP_MD = r"""
## 6. Preuzimanje rezultata (`.zip`)

Sirovi CSV-ovi i JSON-ovi, `colab_environment.json`, svi logovi (uključujući compiler logove), generisani kod i kopija harness-a.
"""

ZIP = r"""
package(start_failures=start_failures, shared_rc=rc_shared, steady_rc=rc_steady,
        outcome=criteria.get("outcome"))
"""


def main():
    with open(HARNESS) as f:
        harness = f.read()
    sha = hashlib.sha256((harness.rstrip("\n") + "\n").encode()).hexdigest()
    cells = [md(INTRO), code(GPU_CHECK), code(CLONE), code(PIP),
             dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                  source=("%%writefile /content/compiled_sigmoid_harness.py\n" + harness).splitlines(keepends=True)),
             code(SETUP.replace("@@HARNESS_SHA256@@", sha)),
             md(GATE_MD), code(GATE), md(RECOMPILE_MD), code(RECOMPILE), md(START_MD), code(START),
             md(STEADY_MD), code(STEADY), md(LOGS_MD), code(LOGS), md(CRITERIA_MD), code(CRITERIA),
             md(ZIP_MD), code(ZIP)]
    nb = dict(cells=cells, nbformat=4, nbformat_minor=0,
              metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                            language_info=dict(name="python"), accelerator="GPU",
                            colab=dict(provenance=[], gpuType="A100", machine_shape="hm")))
    with open(NOTEBOOK, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(NOTEBOOK, "harness sha256", sha)


if __name__ == "__main__":
    main()
