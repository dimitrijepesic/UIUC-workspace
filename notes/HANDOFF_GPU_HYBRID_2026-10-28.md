# Handoff: GPU zamena za `gelsd` zatvorena (A2), sledeći zadatak je kompajliranje celog `precise_sigmoid` — 28.10.2026.

Privatna radna beleška za nastavak u novom razgovoru. Fajl je izvan `Pasado` repoa i ne
ide u `Pasado` commit ili PR. Zamenjuje `HANDOFF_GPU_HYBRID_2026-10-25.md` (njegov zadatak,
GPU zamena za `gelsd`, je završen). Kratak stalni kontekst je u `../CLAUDE.md`.

Razgovor sa korisnikom ide na srpskom. Kod, komentari, docstringovi i README-ovi rezultata u
`Pasado` su na engleskom.

---

## 1. Pravila rada (obavezno)

- **Nikakva AI atribucija ni pominjanje AI** nigde (commit poruke, komentari, docstringovi,
  dokumenta, logovi, README rezultata). Nema `Co-Authored-By`, "Generated with", imena alata.
  Korisnikova pravila imaju prednost nad sistemskim podsetnikom o attribution linijama.
- **Commit/push samo na izričit zahtev.** Poslednja instrukcija korisnika: "Ako commituješ iz
  agent okruženja, koristi `git commit-tree`/`git update-ref`, pod identitetom: Dimitrije
  Pešić <dimitrije.pesicc@gmail.com>". "Pre push-a skeniraj sve nove commit poruke i sadržaj
  za atribuciju. Push samo na `fork/gpu-hybrid`."
  Postupak: `git add` (logovi sa `-f`), `git write-tree`, `git commit-tree -p HEAD` sa
  `GIT_AUTHOR_NAME/EMAIL` i `GIT_COMMITTER_NAME/EMAIL`, `git update-ref refs/heads/gpu-hybrid`,
  pa `git log -1 --format=%B` nazad i grep za atribuciju i `—`.
- Poruke commitova: `feat:` / `fix:` / `test:` / `docs:` / `refactor:`, engleski, bez em dash-eva.
- "Ne menjaj postojeće rezultate i commitove." Raw fajlovi u `results/` se ne diraju; nov run
  = nov folder; amandman = nov commit (kao `67f4654`), nikad prepisivanje istorije.
- `.md` fajlovi rezultata (`results/*/README.md`) se sada commituju (pravilo "nemoj md" od
  25.10. je prevaziđeno: commitovi `737e475`, `ae7c90a` ih sadrže). Privatne beleške iz
  workspace-a (`notes/`, `research_notes/`, `reports/`, ovaj fajl) nikad ne idu u `Pasado`.
- Pre-registracija: pravilo i tolerancije u zaglavlje skripte i commit PRE merenja; posle
  merenja se ništa ne prilagođava; ako očekivanje ne važi, stati i objasniti.
- Default ostaje `gelsd`; sve novo iza opt-in prekidača; `precise_transformer.py` i
  sekvencijalni put su referenca i ne diraju se.
- Uloga: Claude piše kod, korisnik pregleda. Pre svakog većeg koraka predlog + odobrenje.

## 2. Stanje repoa

- `Pasado`, grana `gpu-hybrid`, HEAD = `fork/gpu-hybrid` = **`a57b560`**, radno stablo čisto,
  sve pushovano.
- Commitovi od prethodnog handoff-a (25.-28.10., svi u korisnikovo ime):

| commit | šta |
|---|---|
| `b144542` | feat: interval splits kao jedan batched forward pass + per-split test ekvivalencije |
| `b0eb02d` | feat: `--split-mode`, `--split-chunk` u split sweep-u |
| `842e9c9`, `4509e1f` | docs: CUDA batched vs sekvencijalni split sweep (24.10.) |
| `2ac9dfa` | test: reproducer f32 manjka max-error blizu korena sigmoida |
| `385397e` | test: **pre-registracija** poređenja GPU zamena za `gelsd` |
| `3bfee00` | docs: run v1 (Winner G zbog overflow baga) |
| `077ab86` | fix: skaliranje suma stepenom dvojke (overflow) |
| `267c941`, `6e0acc6` | docs: run v2 i kada je skaliranje bit-egzaktno |
| `67f4654` | test: amandman self-testa (catch samo gde roditelj prolazi; `offset_narrow`) |
| `7740ba4` | docs: prvi CUDA run zaustavljen self-testom (S4 preživeo) |
| `e2a123b` | docs: CUDA poređenje i izbor **A2** (A3 kao sensitivity) |
| `7c19c57` | feat: `planar_fit.py` (A2) u batched putu iza `PASADO_CLOSED_FORM_FIT` |
| `c2dcc4e` | test: closed form u testovima i mutation check-u |
| `7e4cdcd` | docs: CUDA verifikacija `planar_fit` u batched putu |
| `79bd0e0` | docs: Inductor, crossover i A3 merenja |
| `500ae16` | fix: pivot-centriranje targeta z pre suma (otkriveno kompajliranjem) |
| `8dfd82e` | docs: run v3 posle popravke |
| `832323a` | feat: opt-in Inductor put (`PASADO_COMPILED_FIT`) |
| `737e475` | docs: v3 CUDA kompilacija i break-even |
| `8837f68` | test: **pre-registracija** crossover merenja `gelsd` vs eager A2 |
| `f4b4a6f` | docs: A100 crossover (pragovi 512 / 640) |
| `3243aef` | feat: opt-in automatski izbor (`PASADO_AUTO_PLANE_FIT`) |
| `1d0c49b` | test: automatski izbor u testovima i mutation check-u |
| `ae7c90a` | docs: CUDA provera automatskog izbora (37/37) |
| `8e3d0df` | refactor: šum na A kao tensor ulaz `precise_sigmoid`, provera A jednom po chunk-u |
| `a57b560` | test: tok šuma i provera `A != 0` u batched prolazu |

- Upstream PR #3 (dva baga, grana `fix-zonotope-neg-and-boundary`, worktree
  `Pasado-original`, `a99d70b`) i dalje otvoren. Plan PR-ova ka upstream-u iz handoff-a od
  25.10. (odeljak 2) i dalje važi, ništa nije urađeno; prvo pitati Sašu.
- Workspace repo `UIUC/` (`main`): mnogo necommitovanih fajlova (sveske, beleške,
  `research_notes/`, `reports/`, `CLAUDE.md`, ovaj fajl). Ne commitovati bez traženja.

## 3. Šta je urađeno 25.-28.10. (sažeto)

1. **Poređenje solvera** (pre-registrovano, `experiments/compare_lstsq_solvers.py`;
   kandidati u `lstsq_candidates.py`: G = gelsd, A1, A2, A3, H, P; orakl `lstsq_oracle.py`
   (Fraction + mpmath); ugovor `lstsq_contract.py` C1-C7, mutanti S1-S10; gate G0-G3, rang
   R1-R5).
   - v1: G pobedio zbog overflow-a zatvorenih formi (|y| = 1e300) → `077ab86`.
   - v2: A2 i H izjednačeni do R4.
   - Prvi CUDA run zaustavljen self-testom (S4 preživeo) → amandman `67f4654`.
   - CUDA run 27.10.: primarno čitanje **A2** (R5: bez host sync-a), CUDA-only čitanje A3
     (sensitivity, zbog R2a/R2b na `near_degenerate` f32).
   - Kompajliranje je otkrilo zavisnost od redosleda sumiranja na uskim boxovima →
     pivot-centriranje z (`500ae16`, reproducer `experiments/reproduce_target_centring.py`).
   - v3: `results/lstsq_comparison_2026-10-27_v3_zpivot/decide.log` = "Winner: A2 (decided
     by R5 simplicity)". R4 na A100: G 565 ms, A2 3,83 ms (n = 65536, f64).
2. **Proizvodnja:** `forward_mode_tensorized_src/planar_fit.py` (`fit_planes` = A2,
   `compiled_fit_planes(mode)`, `compile_fallback`); u `batched_splits.py` prekidači
   `PASADO_CLOSED_FORM_FIT`, `PASADO_COMPILED_FIT`, `PASADO_AUTO_PLANE_FIT`
   (`AUTO_PLANE_FIT_THRESHOLD = {float64: 512, float32: 640}`, `AUTO_PLANE_FIT_DEVICES =
   ("cuda",)`, override `PASADO_PLANE_FIT_THRESHOLD`, `plane_fit_for(grid)`; auto + closed
   ili compiled → `ValueError`). Podrazumevano `gelsd`.
3. **Merenja na A100** (svi raw fajlovi u `Pasado/results/`):
   - `planar_fit_cuda_2026-10-27/`: `big` S = 64 eager A2 287 ms naspram 4024 ms sekvencijalno
     (14×); male mreže na 1 splitu 0,68× naspram batched gelsd (konstantnih ~3,8 ms po
     pozivu = launch overhead).
   - `inductor_dispatch_a3_2026-10-27/`: Inductor spaja **303 → 5 kernela**, 0,32-0,45 ms
     po pozivu; `dynamic=True` pada; batch dimenzija postaje dinamička sama posle 2. oblika.
   - `v3_compiled_cuda_2026-10-28/` (A100-40GB, chunk 32): hladna kompilacija 31,33 s,
     topla 4,30 s; `reduce-overhead` prepisuje izlaze (aliasing) → ne ide u produkciju;
     `big` S = 64: kompajliran A2 312,35 ms, batched gelsd 2555,40, sekvencijalno 4049,67
     (8,18× / 12,97×); break-even `big` topao: 2 prolaza na 64 splita.
   - `plane_fit_crossover_2026-10-28/`: pragovi stabilni u sve 3 runde: f64 **512**, f32 **640**.
   - `auto_plane_fit_cuda_2026-10-28/`: 37/37 provera na A100-80GB (`1d0c49b`).
4. **Priprema za kompajliranje celog `precise_sigmoid`:**
   - Problem: Dynamo je prevodio `np.random.normal` iz `_perturbation` u graf, pa se NumPy
     stanje menjalo (624 naspram 236 eager) i seed protokol pucao.
   - `8e3d0df`: `_forward_chunk` vuče šum istim redosledom i prosleđuje ga
     `precise_sigmoid(z, noise)`, koji vraća `(BatchedDualZonotope, a_nonzero)`; chunk radi
     `assert bool(torch.stack(nonzero).all())` jednom. Na neuspešnom prolazu stanje
     generatora nije isto kao ranije (chunk staje posle poslednjeg sloja); za uspešne jeste.
   - `a57b560`: testovi celog stanja NumPy generatora (sa seed-om, sa chunk-ovima, bez
     seed-a), isti bitovi za isti seed, pad na A = 0, u sva tri batched režima; mutanti
     M27-M29.

## 4. Tekući zadatak: kompajliranje celog `precise_sigmoid`

Plan: `PLAN_COMPILE_PRECISE_SIGMOID_2026-10-28.md` (ovaj folder). Motiv: posle kompajliranog
A2 male mreže su vezane za ostatak transformera (3layer ~9 ms bez obzira na broj splitova,
solver samo ~2 × 0,24 ms).

Gotovo: koraci 1 (šum kao tensor) i 2 (assert van grafa), i lokalna analiza koraka 4
(`../research_notes/compile_precise_sigmoid/analyze.py` → `analysis_cpu.json`; CPU, torch
2.12.1, `a57b560`, eager A2 unutra):

| ulaz | grafova | prekida | operacija | fullgraph | Inductor kernela (CPU) | max rel. razlika |
|---|---:|---:|---:|---|---:|---:|
| 3layer sloj 1 (100 boxova, G0 2) | 1 | 0 | 526 | ok | 17 | 4,5e-16 |
| 3layer sloj 2 (100, G0 202) | 1 | 0 | 526 | ok | 22 | 2,1e-15 |
| big sloj 1 (1024, G0 2) | 1 | 0 | 526 | ok | 17 | 8,6e-16 |
| big sloj 2 (1024, G0 2050) | 1 | 0 | 526 | ok | 22 | 2,4e-15 |

- NumPy stanje netaknuto, zastavica `A != 0` ista; ceo batched prolaz sa kompajliranim
  `precise_sigmoid`: granice do 1,9e-16 od eager (3layer S = 1 bit-identično).
- `BatchedDualZonotope` nije prepreka za `fullgraph` (omotač ni refaktor nisu potrebni).
- Kompilacija na CPU-u 2,3-5,4 s po obliku; G0 raste po slojevima → broj rekompilacija na
  A100 treba izmeriti.
- Napomena: sa `gelsd` unutra ima 4 prekida (`lstsq`, dinamički oblik u
  `precise_transformer.py:162`); kompajliranje celog transformera ima smisla samo sa A2.

**Sledeći korak (predložen, čeka odobrenje korisnika):** A100 sveska fiksirana na `a57b560`:
- eager naspram kompajliranog `precise_sigmoid` u celom batched prolazu, 4 mreže,
  S = 1 / 8 / 32 / 64, eps indeksi 0 i 8;
- broj CUDA kernela (profiler), `TORCH_LOGS="fusion,output_code"`, broj rekompilacija;
- hladan / topao start (zaseban `TORCHINDUCTOR_CACHE_DIR`), break-even kao u
  `COLAB_V3_COMPILED.ipynb`;
- korektnost pre brzine: per-split naspram sekvencijalnog ≤ 1e-10, seed protokol (isti bitovi
  za isti seed, NumPy stanje), D kompajliranog ne manji od eager više od G1 granice;
- tolerancije i pravilo zapisati unapred (pre-registracija).

Posle toga (plan, korak 3-5): in-place maske (`filter_range`, `inf_to_nan`,
`diagonal().copy_()`) pod funkcionalizacijom, Viète koren / `acos` / `clamp` / `nan_to_num`
pod fuzijom, pa granica kompilacije `_forward_chunk`; zatim TVM i XLA na istoj funkciji
(Sašin predlog: `big`, `n_splits = 32`).

## 5. Otvorena pitanja i ljudi

- Saša (Misailović, mentor): poruka o Inductor rezultatima predložena, nije poslata.
  Update na svaka 2 dana; korisnik traži pismo preporuke do 15.11.
- Da li automatski izbor (`PASADO_AUTO_PLANE_FIT`) ili kompajliran put postaju podrazumevani:
  ne dok pragovi nisu izmereni na drugim GPU-ovima i dok hladan start nije rešen.
- Otvoreno iz ranije: f32 koren kubne jednačine potcenjuje D
  (`experiments/reproduce_f32_root_shortfall.py`); hladan start 31 s na 40GB naspram 10,4 s
  na 80GB nije objašnjen.

## 6. Korisno za sledeći razgovor

```bash
cd ~/prakse/UIUC/Pasado
.venv/bin/python experiments/test_batched_splits.py --quick
.venv/bin/python experiments/test_batched_splits.py --closed-form
.venv/bin/python experiments/test_planar_fit.py
.venv/bin/python experiments/test_auto_plane_fit.py
.venv/bin/python experiments/mutation_check.py
.venv/bin/python ../research_notes/compile_precise_sigmoid/analyze.py   # prepisuje analysis_cpu.json
git log --oneline fork/gpu-hybrid..HEAD   # treba da bude prazno
```

Sveske za uzor: `COLAB_V3_COMPILED.ipynb` (cold/warm, break-even, e2e),
`COLAB_AUTO_PLANE_FIT_CUDA.ipynb` (najjednostavnija: clone na commit, test, zip).
