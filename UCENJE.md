# Plan učenja — kod, razlike naspram originala, radovi

> Napravljeno 7.9.2026. Cilj: razumeti Pasado kod i svaku izmenu koju sam napravio
> pre nego što nastavim sa hibridnim CPU/GPU portom.
> Redosled je namerno: prvo teorija (rad), pa netaknuti kod, pa moje izmene, pa skripte.
> Procena: 4–5 radnih dana. Rok MLSys 30.10.2026, ostalo 53 dana.

Diff fajlovi naspram `uiuc-arc/Pasado` main-a su u `review/`:

| fajl | šta je | linija |
|---|---|---|
| `review/01_precise_transformer.diff` | jezgro svih optimizacija (necommitovano) | 466 |
| `review/02_get_lipschitz.diff` | harness Sekcije 5.4 + `lipschitz.sh` | 79 |
| `review/03_colab_gpu_wip.diff` | 6 commit-ova sa `fork/colab-gpu-wip`: Viete cubic, Colab skripte | 1185 |
| `review/04_compat_fixes.diff` | NumPy 2 popravke, README (već u upstream PR-ovima) | 47 |

Beleške tokom čitanja vodi u `review/REVIEW_NOTES.md` (fajl:linija + pitanje/sporno).

---

## Dan 1 — Rad i teorija (pola dana), pa netaknuti kod (pola dana)

### 1.1 Pasado rad: `papers/OOPSLA2023Final.pdf` (Laurel et al., OOPSLA 2023)

Čitaj sa olovkom, tražiš odgovore na ova pitanja:

- Šta je **dual broj** i **hiper-dual broj**; kako se time dobija prvi i drugi izvod bez
  simboličkog diferenciranja (forward mode AD).
- Šta je **zonotop** (centar + generatori + šumni simboli) i zašto broj generatora raste
  kroz slojeve (ovo je ključ za GPU priču: matmul raste *tokom* prolaza).
- Šta je "precise" transformer za `sigmoid'(x)·y`: **sintetizovana polinomska apstrakcija**
  (kvadratni fit A·x + B·y + C nad mrežom uzoraka), i kako se računa **maksimalna greška**
  fita nad kutijom (uglovi + nelinearna granica). To je tačno kod koji sam vektorizovao.
- Sekcija 5.4: šta se tačno meri (Lipschitz konstanta po skalarnom haze parametru na
  MNIST-u, sigmoid FCN mreže 3/4/5layer i `big`), i sa čim se rad poredi (interval /
  zonotope AD, **ne** sa CROWN-om — to je moj nalaz iz `LIRPA_COMPARISON.md`).
- Sekcije 5.2 (ODE, klima/hemija) i 5.3 (Black-Scholes): samo prelet, koriste
  *ne-tenzorizovanu* implementaciju; nisu u fokusu.

### 1.2 Netaknuti kod (čitaj u ovom redosledu, sa radom pored)

Ovo je kod koji nisam menjao. Cilj: znati šta svaka klasa predstavlja pre nego što
gledam svoje izmene.

| fajl | linija | šta tražiš |
|---|---|---|
| `forward_mode_tensorized_src/Duals.py` | 297 | klasa dualnog broja; kako se `__mul__`, sigmoid itd. propagiraju |
| `forward_mode_tensorized_src/HyperDuals.py` | 576 | isto za drugi izvod; koje operacije postoje |
| `forward_mode_tensorized_src/SimpleZono.py` | 583 | `AffineZonotope` (**matmul hotspot**, 45% na big), `SigmoidDualZonotope.__mul__` (čudno spor, 119 s na big — zabeleži), abs+sum redukcije granica |
| `forward_mode_tensorized_src/HyperDualZono.py` | 191 | spoj: hiper-dual nad zonotopima |
| `Section_5_4/get_lipschitz.py` (originalna verzija) | ~215 | tri puta: interval, baseline `forward_zono`, precise `forward_zono_precise`; gde se učitavaju težine i pravi ulazni zonotop (ovo su mesta za `PASADO_DEVICE`) |

Za originalnu verziju fajla: `git -C Pasado show origin/main:Section_5_4/get_lipschitz.py`. Ceo originalni Pasado je i fizički u `Pasado-original/` (read-only git worktree).

Provera razumevanja na kraju dana: pusti `Section_5_4/get_lipschitz.py --network 3layer --no-save`
(8 s) i objasni sebi svaku od tri liste u izlazu (`lc_precise`, `lc_zonos`, `time_precise`).

---

## Dan 2 — `precise_transformer.py` original, pa moj diff liniju po liniju

### 2.1 Original (`git -C Pasado show origin/main:forward_mode_tensorized_src/precise_transformer.py`, ~480 linija)

Redosled po toku podataka za jedan sloj:

1. `get_linspace` — mreža 5×5 uzoraka po neuronu (2 `.item()` po neuronu → 7,8M sync-ova na big)
2. `sigmoid_prime_times_y` — funkcija koja se fituje
3. `lin_reg_tensor` — `lstsq` (gelsd) za A, B, C **po neuronu** (~2M poziva na big)
4. `inverse_poly_tensor` — rešava kubnu jednačinu za kritične tačke na granici
5. `check_corners` / `check_nonlinear_boundary` → `compute_max_error` — Python petlje po neuronu
6. `sigmoid_prime_product_tensor` — sve spaja u novi zonotop

### 2.2 Moj diff: `review/01_precise_transformer.diff` (466 linija, ~4 h)

Ovo je kod koji nosi tvrdnje "bit-identično" i "3,2–3,6×". Čitaj hunk po hunk:

| hunk | šta je dodato | tvrdnja koju nosi | test koji je dokazuje |
|---|---|---|---|
| `@@ -1,8` | 5 env selektora (`PASADO_VECTORIZED_PRECISE`, `PASADO_VEC_BOUNDARY`, `PASADO_BATCHED_LSTSQ`, `PASADO_BATCHED_GRID`, `PASADO_REAL_CUBIC`) | svaka optimizacija iza flaga, original netaknut | `test_selector_paths.py` |
| `@@ -11,6` | `_linspace_rows`, `get_linspace_batched` | bit-egzaktno naspram `torch.linspace` (na x86; na arm64 samo STEPS≤5, vidi `notes/MAC_SETUP.md` §6) | `test_batched_grid.py` |
| `@@ -77,6` | `lin_reg_tensor_batched` | jedan batched `lstsq` po sloju umesto po neuronu, isti gelsd | `experiments/test_batched_lin_reg.py` |
| `@@ -107,6` | `inverse_poly_real_tensor` (Viete) | **nije** bit-identičan (≤5,7e-5), parkiran kao GPU enabler | `REAL_CUBIC.md` |
| `@@ -209,6` | `check_corners_tensor` | reprodukuje petlju | `test_check_corners_tensor.py` |
| `@@ -294,6` | `_max_objective_over_x_candidates`, `check_nonlinear_boundary_tensor` | reprodukuje petlju | `test_check_nonlinear_boundary_tensor.py` |
| `@@ -376,14` | `compute_max_error` sa tenzorskim putem | isti maksimum | `test_compute_max_error_tensor.py` |
| `@@ -414,23` | `sigmoid_prime_product_tensor` — dispatch po selektorima | end-to-end | `experiments/correctness_e2e.py` (≤5,7e-14) |

Metod: za svaki hunk prvo pročitaj originalnu funkciju, pa tenzorsku, pa **pusti test**.
Pitanja koja moraš da umeš da odgovoriš: zašto je dtype promotion bio bug; zašto boolean
mask lowering na `aten.nonzero` smeta (komentar u `_linspace_rows`); zašto je batched grid
"symmetric halves".

### 2.3 `review/02_get_lipschitz.diff` (79 linija, 30 min)

`--no-save`, selektori, tajming. Ovde ćeš dodavati `PASADO_DEVICE` (korak 2 hibrida).

---

## Dan 3 — Rezultati koje sam tvrdio, i skripte koje su ih proizvele

Čitaj dokument, pa skriptu koja ga je napravila, pa proveri da li se brojevi slažu.

| dokument (`Pasado/results/optimization_section_5_4/`) | skripta | šta proveravaš |
|---|---|---|
| `FORMAL_SUMMARY.md` (219) | `experiments/benchmark_lipschitz.py`, `profile_variants.py` | 3,24/3,64/3,37× i 1,65×; metodologija pinovanog naizmeničnog merenja |
| `BIG_PROFILE_ANALYSIS.md`, `POST_GRID_PROFILE.md` | `profile_variants.py`, `extract_pstats.py` | matmul 45%, gelsd 1,7%; zamka: cProfile knjiži `@` u pozivaoca |
| `LIRPA_COMPARISON.md` (262) | `experiments/colab_compare_pasado_lirpa.py`, `colab_bab_sweep.py` (u `03_colab_gpu_wip.diff`) | 1,3–2,8×; presek K≈7; **da li je poređenje fer** (isti problem, ista preciznost, warm-up) |
| `EQUAL_TIME_COMPARISON.md` | `lipschitz-comparison/pasado_equal_time_sweep.py`, `analyze_equal_time.py` | 15/15; vremena su nepinovana — ograda za rad |
| `GPU_READINESS.md` (210) | `experiments/colab_gpu_profile.py`, `colab_rank_deficient_lstsq_test.py` | spisak 40 factory poziva bez `device=`; zašto `gels` nije bezbedan (15–34 reda veličine) |
| `LSTSQ_ANALYSIS.md`, `BATCHED_LSTSQ_RESULTS.md` | `probe_lstsq.py`, `benchmark_lstsq.py` | oblici [n,25,3]; kad se batching isplati |
| `TORCH_COMPILE_ANALYSIS.md`, `REAL_CUBIC.md` | `compile_experiment.py`, `colab_compile_boundary.py`, `colab_dynamo_explain.py` | zašto compile nije dao rezultat na Windows CPU |

Prelet (ne čitaj liniju po liniju): `RAW_NOTES.md`, `NEXT_RESEARCH_DIRECTIONS.md`,
`CACHING_ANALYSIS.md`, dijagnostičke skripte (`diagnose_layer1_diff.py`, `dump_pstats.py`).

`lipschitz-comparison/` (van Pasado repoa): `compare_pasado_lirpa.py` je CPU verzija Colab
poređenja, `bab_experiment.py` BaB na auto_LiRPA strani, `ctx1-4.py` su bili context fajlovi.
CSV-ovi su sirovi rezultati koji stoje iza tabela.

---

## Dan 4 — Ostali radovi (pola dana), pa ceo lanac na Mac-u (pola dana)

### 4.1 Radovi po prioritetu

1. **Shi et al. 2022** (auto_LiRPA / CROWN Jacobian bounds, Local Lipschitz) — **nije u repou**,
   izvor je `Local-Lipschitz-Constants/` (klon) i njihov README sa linkom. Ovo je baseline
   protiv koga se poredim; moram da znam kako računaju Jacobian bounds i BaB da bih
   branio "fer poređenje". Kod: `lipschitz-comparison/auto_LiRPA_src/`.
2. **Neptune: `papers/2510.08726v2.pdf`** (Zhao et al., fuzija operatora na GPU) — Sašina grupa,
   kontekst za "sistemski" smer; čitaj Sekcije 1–3 i evaluaciju. Kod u `neptune/`,
   status u `neptune/KNOWN_ISSUES.md`.
3. **CROWN original** (Zhang et al. 2018, "Efficient Neural Network Robustness
   Certification with General Activation Functions") — pozadina za 1.
4. Kvantizacija (`papers/quantization/*.pdf`, Wu et al. 2020; Gholami et al. 2021) — iz
   ranije faze prakse (TVM), **nije** na kritičnom putu, ostavi za kasnije.

### 4.2 Ceo lanac od nule, sam

```bash
cd ~/prakse/UIUC/Pasado && source .venv/bin/activate
cd forward_mode_tensorized_src
python test_check_corners_tensor.py && python test_check_nonlinear_boundary_tensor.py
python test_compute_max_error_tensor.py && python test_selector_paths.py
cd .. && python experiments/test_batched_lin_reg.py
python experiments/correctness_e2e.py          # old vs vectorized
python experiments/correctness_e2e.py --batched
cd Section_5_4
PASADO_VECTORIZED_PRECISE=0 python get_lipschitz.py --network 3layer --no-save   # original put
python get_lipschitz.py --network 3layer --no-save                                # optimizovan
PASADO_BATCHED_LSTSQ=1 PASADO_BATCHED_GRID=1 python get_lipschitz.py --network 3layer --no-save  # buduća osnova hibrida
```

Ako umeš da objasniš svaku razliku u vremenu i svaki broj — spreman si za port.

---

## Dan 5 — Nastavak rada: korak 0 hibrida

Iz plana hibridnog porta:

0. ✅ 15.9: grana `sprint-optimizations` napravljena, sve commitovano u 3 commita (tri puta
   odlagano). Grana `gpu-hybrid` od nje.
1. Osnova = `vec_batched_grid` (sva tri flaga na 1).
2. `PASADO_DEVICE` + 40 factory poziva sa `device=`.
3. Hibrid u `lin_reg_tensor_batched`: `.cpu()` → gelsd → `.to(device)`.
4. Colab A100: korektnost sa tolerancijom (ne bit-identičnost).
5. Merenje `big`, rezultat Saši.

Pre nego što krene korak 2: pitati Sašu (a) da li je Faza 1+2 iz `PLAN_MLSYS.md` dovoljna
za ovaj ciklus, (b) ablacija neoptimizovan Pasado vs auto_LiRPA koju je sam tražio.

---

## Pitanja koja ću zapisivati u `review/REVIEW_NOTES.md`

- Sve što ne razumem u originalu (rad + kod), sa referencom na sekciju rada.
- Sve što mi u sopstvenom diffu deluje sporno ili nedokumentovano.
- Brojevi u `results/*.md` koje ne mogu da izvedem iz skripte koja ih je napravila.
- Poznata anomalija za proveru: baseline `forward_zono` (382 s) sporiji od precise (343 s)
  na big; `SigmoidDualZonotope.__mul__` 119 s.
