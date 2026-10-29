# Handoff: Pasado GPU hybrid, upstream bug fixes & compiler plan — 22.10.2026.

Privatna radna beleška za nastavak u novom razgovoru. Fajl je izvan `Pasado`
repoa i ne treba ga dodavati ni u jedan `Pasado` commit ili PR.

---

## 1. Lokacije i Git stanje

- Workspace: `/Users/dimitrije/prakse/UIUC`
- Glavni radni repo: `/Users/dimitrije/prakse/UIUC/Pasado` (grana `gpu-hybrid`, poslednji commit `0eb14cd` na `fork/gpu-hybrid`, plus lokalne izmene opisane ispod)
- Upstream worktree: `/Users/dimitrije/prakse/UIUC/Pasado-original` (grana `fix-zonotope-neg-and-boundary` napravljena od `origin/main` = `e857413`, sa 6 linija ispravke za upstream PR)
- Fork remote: `https://github.com/dimitrijepesic/Pasado`
- Upstream remote: `https://github.com/uiuc-arc/Pasado`

---

## 2. Šta je završeno između 10.10. i 22.10.2026.

### A. Merenje CPU fallback-a na CUDA-i (`a4dd045`, `03bd011`)
- Direktorijum: `Pasado/results/gpu_hybrid_fallback_2026-10-16/`
- Skripte: `experiments/measure_hybrid_fallback.py`, `experiments/measure_sync_free_hybrid.py`
- Nalaz (`float64`, 3 slike × 3 epsilona, 5 warm-up + 10 merenih ponavljanja):
  - Na `3layer`, `4layer`, `5layer`: CPU fallback (`D2H + design + gelsd + H2D`) nosi `18.0%–19.1%` `precise` vremena (`2.2–4.6 ms` od `12.5–24.2 ms`).
  - Na `big` (`n = 1024`): medijana `precise` prolaza je `532.31 ms` (Colab CPU, 6 niti) naspram `63.34 ms` (CUDA hibrid baseline) i `58.62 ms` (`sync_free`, bez suvišnih host-sync provera u petlji slojeva).
  - Na `big`, CPU fallback nosi **`54.54%`** (`34.57 ms` od `63.34 ms`, od čega sam `gelsd` nosi `32.11 ms`, a D2H/H2D prenosi `1.40 ms`).

### B. Layerwise CPU/CUDA divergencija (`3b7b0a2`, `3b35688`)
- Direktorijum: `Pasado/results/gpu_hybrid_layerwise_2026-10-17/`
- Skripta: `experiments/measure_layerwise_device_diff.py`
- Nalaz:
  - Stari kompleksni Cardano rešavač (`complex64` + `torch.isreal`): na CUDA-i odbacuje `2348–2812` od `6144` korena po sloju na `big`, pa relativna razlika bounda raste kroz slojeve do `65.2%` na izlazu. Takođe i na samom CPU-u gubi korenove blizu $s \approx 0$ i $s \approx 1$ zbog `float32/complex64` preciznosti pre `inv_sigmoid`.
  - Realni Viète rešavač (`PASADO_REAL_CUBIC=1`): `0` izgubljenih korena, CPU i CUDA se poklapaju do mašinske preciznosti (`<= 1.41e-15`) u svakom sloju.

### C. Interval splitting (`--n-splits = 1..64`) sweep i VRAM (`e894470`, `600ba2c`, `603a48c`, `0eb14cd`)
- Direktorijum: `Pasado/results/gpu_hybrid_split_sweep_2026-10-17/`
- Skripte: `experiments/measure_split_sweep.py`, `experiments/plot_split_sweep.py`
- Vreme izvršavanja raste strogo linearno `O(n)` sa brojem splitova jer se podintervali trenutno izvršavaju u sekvencijalnoj `for` petlji, dok `peak_vram_mb` ostaje konstantan `O(1)` (`10.78 MB` na `3layer`, `480.52 MB` na `big`).
- Sašina kalibracija sa grafika: `n_splits = 16` ili `32` (najčešće `32`) je glavni "point of diminishing return" gde se izvuče skoro svo smanjenje bounda pre nego što se kriva zaravni ka `64`.

### D. Kompleksni kubni fix-evi (`25e6092`)
- Skripta: `Pasado/experiments/verify_complex_cubic_fixes.py`
- Upoređeni Metod 1 (fiksni prag `|Im(x)| <= 1e-6` + kubni ostatak) i Metod 2 (analitička procena zaokruživanja inspirisana Miné 2004, `k_ops = 12`), uz popravku eksponenta `0.33333 -> 1.0/3.0`. Oba zadržavaju `3000/3000` validnih korena na CUDA-i.

---

## 3. Dva upstream baga u originalnom Pasado kodu (otkriveno 20.10., ispravljeno 22.10.)

Autograd (`jvp`) provera otkrila je dva baga koji postoje u originalnom `uiuc-arc/Pasado` (`origin/main`):

1. **`Zonotope.__neg__` i `DualZonotope.__neg__` (`SimpleZono.py:117, 370`) nisu negirali generatore:**
   - Vraćali su `Zonotope(-self.centers, self.generators)` umesto `-self.generators`.
   - Posledica: u `get_lipschitz.py`, `(one - zono_eps) * img_f + zono_eps` je davao generator `g_t * (1 + img_f)` umesto `g_t * (1 - img_f)`. Za `n_splits = 1` ulazni zonotop je bio veštački naduvan, a za `n_splits > 1` (podintervali $[l, r]$ sa $l > 0$) realni zonotop je mašio stvarnu putanju zamagljenja pa je `lc_precise` u nekim slučajevima padao **ispod** stvarnog izvoda mreže (npr. `4layer`, slika 0, `eps_idx=8`, `n_splits=16`: `177.1394 < 177.1797`).
2. **`check_nonlinear_boundary`, `check_nonlinear_boundary_softplus` i `_max_objective_over_x_candidates` (`precise_transformer.py`) nisu uzimali `.abs()`:**
   - Gledali su samo `torch.max(evaluation)` umesto `torch.max(evaluation.abs())`, ignorišući negativnu grešku aproksimacije u kritičnim tačkama na nelinearnim ivicama (dok `check_corners` gleda oba znaka).

### Status ispravki (22.10.2026.):
- **U `/Users/dimitrije/prakse/UIUC/Pasado` (grana `gpu-hybrid`):**
  - Ispravka je uneta u `forward_mode_tensorized_src/SimpleZono.py` i `forward_mode_tensorized_src/precise_transformer.py`.
  - Napravljena je verifikaciona skripta `Pasado/experiments/verify_upstream_bugs.py` koja algebarski i kroz `autograd` potvrđuje ispravku (`ALL CHECKS PASSED`).
  - Na `big` (`image_id = 0`, `eps_idx = 0`), za `n_splits = 1` bound pada sa naduvanih `14,466.40` na `9,820.98`, a za `n_splits = 16` iznosi `576.54` (svuda `>= 248.01` autograd maksimuma).
  - Svih 9 testova i `mutation_check.py` prolaze.
  - Čeka `git commit` u korisnikovom terminalu.
- **U `/Users/dimitrije/prakse/UIUC/Pasado-original` (grana `fix-zonotope-neg-and-boundary`):**
  - Napravljena čista grana od `origin/main` (`e857413`) sa samo 6 izmenjenih linija u `SimpleZono.py` i `precise_transformer.py` za PR ka `uiuc-arc/Pasado`.

---

## 4. Sledeći koraci (po dogovoru sa Sašom od 20–22.10.2026.)

1. **Commit i PR za 2 upstream baga:**
   - Komitovati izmene u `Pasado` (`gpu-hybrid`) i u `Pasado-original` (`fix-zonotope-neg-and-boundary`), te otvoriti PR na `uiuc-arc/Pasado`.
2. **GPU paralelizacija (batching) po `n_splits` (`[S, N]` i `[S, G, N]`):**
   - Prebaciti `n_splits` petlju u batch dimenziju tako da svih `S` podintervala (npr. `n_splits = 32` ili `64`) prolazi paralelno kroz `AffineDualZonotope` i `PreciseSigmoidDualZonotope`.
   - U `PreciseSigmoidDualZonotope`, nezavisni boksovi `(split s, neuron j)` mogu da se spljošte u `[S * N]` za izračunavanje `(A, B, C)` i maksimalne greške, pa vrate u `[S, N]`.
   - Ponoviti split sweep (`1..64`) sa ispravljenim upstream bagovima i paralelnim splitovima.
3. **Kompajlerska evaluacija na `big` sa `n_splits = 32` (Sašin predlog):**
   - Saša je predložio da se uzme `n = 32` verzija na `big` mreži i provuče kroz **3 kompajlera**:
     1. **TorchInductor** (`torch.compile(backend="inductor")`),
     2. **TVM**,
     3. **XLA** (`torch_xla` / OpenXLA).
   - Uporediti vremena izvršavanja i izvući iz logova kompajlera (npr. `TORCH_LOGS="fusion,output_code"`) koje su fuzije uspeli da urade (posebno za elementwise lanac kubnog rešavača i boundary check-a).
   - Ako standardni kompajleri već fuzionišu sve što se može fuzionisati na ovom paternu, onda Neptune nije neophodan za ovaj deo; u suprotnom identifikovati šta ostaje nefuzionisano.
4. **Dugoročna opcija koju je Saša pomenuo (za diskusiju, ne još za kucanje):**
   - **AWS Trainium / TorchNeuron (NKI)** i njegov CPU simulator — grupa dobija pristup Trainium mašinama od Amazona, pa čist tenzorski/batch-ovani PyTorch kod otvara mogućnost skaliranja na Trainium akceleratorima.
