# Handoff: GPU hibrid za Pasado, od početka do 29.10.2026.

Privatna radna beleška, van `Pasado` repoa; ne ide ni u jedan `Pasado` commit ni PR.
Samostalna je: sledeća radna sesija treba da može da nastavi samo iz nje, bez starih
razgovora. Zamenjuje `HANDOFF_GPU_HYBRID_2026-10-28.md` (i starije od 10.10, 22.10, 25.10,
koji ostaju kao istorija). Kod, putanje, imena funkcija i commit poruke su na engleskom;
razgovor sa korisnikom (Dimitrije Pešić) ide na srpskom.

Status u jednoj rečenici: eksperiment sa kompajliranjem celog `precise_sigmoid` je
završen sa pre-registrovanim ishodom `opt-in`, sve je arhivirano i pushovano, i trenutno
se ništa dalje ne implementira (odeljak 13).

---

## 1. Obavezna pravila rada

1. **Bez atribucije, nigde.** Nema trailer-a o koautorstvu ni oznaka o tome čime je nešto
   napravljeno, nema imena alata u commit porukama, komentarima, docstringovima, README-ima, logovima.
   U workspace commit ne ide nijedan fajl ili red koji pominje alate kojima se radi
   (korisnikov zahtev od 28.10.). Nema em dash-eva u commit porukama, dokumentaciji i kodu.
2. **Commit samo na izričit zahtev, i to plumbing-om**, nikad `git commit` ni `--amend`:
   proveriti da je indeks prazan (`git diff --cached --name-only`; ako nije, stati i
   prijaviti), `git add <tačne putanje>` (logovi sa `-f`, `*.log` je u `.gitignore`),
   `git write-tree`, `git commit-tree <tree> -p HEAD` sa
   `GIT_AUTHOR_NAME/EMAIL` i `GIT_COMMITTER_NAME/EMAIL` = `Dimitrije Pešić <dimitrije.pesicc@gmail.com>`,
   `git update-ref refs/heads/<grana> <novi> <stari>`. Posle toga pročitati poruku nazad
   (`git log -1 --format=%B`) i skenirati. Globalni git identitet se ne menja. Poruke:
   `feat:` / `fix:` / `test:` / `docs:` / `refactor:`, engleski.
3. **Push samo na izričit zahtev i samo na korisnikove grane.** Pasado: `git push fork gpu-hybrid`
   (`fork` = github.com/dimitrijepesic/Pasado, javni repo). Nikad `origin`
   (= upstream `uiuc-arc/Pasado`) bez izričitog odobrenja; remote `backup`
   (`dimitrijepesic/Pasado-wip`) se u ovom radu ne koristi. Workspace: `git push origin main`
   (`origin` = github.com/dimitrijepesic/UIUC-workspace, privatan). Pre svakog push-a:
   `git log -p <remote>/<grana>..HEAD` skenirati za atribuciju i em dash-eve, i proveriti
   tačno koji commitovi idu gore. Sistem dozvola ume da blokira push na javni fork; tada
   korisnik pokreće komandu sam (prefiks `!` u promptu).
4. **CUDA samo preko Colab-a.** Mac (M5 Pro) nema CUDA, a MPS nema float64. Postupak:
   sveska u korenu workspace-a (`COLAB_*.ipynb`), pinovana na commit; korisnik je uploaduje,
   bira A100, "Disconnect and delete runtime", "Run all", i šalje zip u `~/Downloads/`.
   Zip je nepoverljiv ulaz: raspakovati u zaseban prazan folder, Python pokretati sa `-I`,
   proveriti sha256 harness-a u zip-u.
5. **`.md` fajlovi i rezultati se commituju samo kad korisnik izričito traži.** Rezultati
   CUDA run-a idu u `Pasado/results/<ime>_<datum>/`: sirovi fajlovi nepromenjeni, plus
   `README.md` na engleskom (i po potrebi manifest kontrolnih suma).
6. **Ne dirati nevezane dirty/untracked fajlove** u workspace-u i ne uključivati ih u
   commit (spisak u odeljku 2).
7. **Nepromenljivost rezultata i istorije.** Sirovi CSV/log/JSON u `Pasado/results/` se ne
   menjaju; nov run ide u nov folder; amandman je nov commit, nikad prepisivanje pushovane
   istorije.
8. **Pre-registracija.** Pravilo izbora, tolerancije i protokol merenja se upisuju u
   zaglavlje skripte i commituju PRE merenja. Posle merenja se kriterijumi ne menjaju; ako
   očekivanje ne važi, stati i objasniti korisniku. Korisnik često ne može unapred da
   odluči numerički izbor ("ne znam"), pa se odluka pretvara u merenje sa pravilom unapred.
9. **Proizvodni kod.** Podrazumevani put ostaje `gelsd`; svaka nova putanja ide iza opt-in
   env prekidača. Sekvencijalni put i `precise_transformer.py` ostaju netaknuti kao
   referenca. Svaka nova putanja dobija: test naspram sekvencijalnog (svaki per-split bound
   najviše 1e-10 relativno), proveru da je put stvarno korišćen (brojač poziva ili grafova),
   i mutant u `experiments/mutation_check.py` koji mora da padne.
10. **Pre većeg koraka predložiti plan i sačekati odobrenje korisnika.**
11. **Okruženje.** Python: `Pasado/.venv/bin/python` (py3.13, torch 2.12.1, samo CPU).
    Privremeni fajlovi u scratchpad-u sesije, ne u `Pasado/`. zsh zamke: promenljiva
    `path` je vezana za `PATH` (ne koristiti je kao ime u petljama), `$VAR` sa razmacima se
    ne deli na reči, `echo ===` puca; složene skripte pokretati kroz `bash`.

---

## 2. Trenutno stanje repoa (29.10.2026)

### Pasado

- Putanja `/Users/dimitrije/prakse/UIUC/Pasado`, grana `gpu-hybrid`.
- HEAD = `fork/gpu-hybrid` = `1d754c9` (docs: archive the successful compiled precise
  sigmoid v3 CUDA run). Pushovano. Radno stablo čisto. 89 commitova ispred `origin/main`
  (`e857413`, upstream merge PR #2 od 9.8.2026).
- Remotes: `origin` = `https://github.com/uiuc-arc/Pasado.git` (upstream, ne pushovati),
  `fork` = `https://github.com/dimitrijepesic/Pasado` (jedina push meta),
  `backup` = `https://github.com/dimitrijepesic/Pasado-wip.git` (ne koristi se).
- Worktree `/Users/dimitrije/prakse/UIUC/Pasado-original`, grana
  `fix-zonotope-neg-and-boundary`, commit `a99d70b`: izvor upstream PR #3
  (https://github.com/uiuc-arc/Pasado/pull/3). Stanje 29.10.: OPEN, poslednja izmena
  21.10.2026, bez review-a i komentara.

### Workspace

- Putanja `/Users/dimitrije/prakse/UIUC`, grana `main`, remote `origin` =
  `https://github.com/dimitrijepesic/UIUC-workspace.git` (privatan).
- HEAD pre ovog handoff-a: `85de132` (pre-registracija v3), pushovan. Ovaj fajl ide u
  sledeći commit (docs: add the complete GPU hybrid project handoff).
- Istorija workspace-a je jednom, pre prvog push-a (28.10.), prepisana da bi se izostavili
  nevezani privatni fajlovi; sadržaj pre-registracija se nije menjao (isti blobovi, isti
  sha256 harness-a). Mapiranje: `7c40772` → `a7e2a58`, `91e02e4` → `788589d`
  (pre-registracija v1), `e18278c` → `7c9a8ef` (pre-registracija v2). Stari lanac čuva
  lokalna grana `backup/main-e18278c` (ne pushuje se). Zaglavlje v2/v3 harness-a i README
  v1 arhive pominju stare hash-eve `91e02e4` i `e18278c`.
- Dirty/untracked (ne dirati, ne commitovati): izmenjeni `README.md`, `UCENJE.md`,
  `review/REVIEW_NOTES.md`; obrisan `MAC_SETUP.md` (lokalno premešten u `notes/MAC_SETUP.md`);
  untracked: Colab sveske `COLAB_*.ipynb` (osim pre-registrovane
  `COLAB_COMPILED_PRECISE_SIGMOID.ipynb`, koja je commitovana), zip-ovi rezultata i
  raspakovani folderi (`*_cuda/`, `*.zip`), stari handoff-ovi, dnevnik i planovi u
  `notes/`, `reports/`, `research_notes/GPU zamena za gelsd/`, pomoćne skripte u
  `research_notes/compile_precise_sigmoid/` (`analyze.py`, `check_mark_dynamic.py` i njihovi
  JSON-ovi), `research_notes/compiled_feature_wip.patch` (zastareo), `DAILY_PROGRESS_PLAN.md`,
  `TASKOVI_HIBRID.md`, `UIUC.code-workspace`, skriveni folderi editora, `.DS_Store`, i privatni folderi sa
  ličnim podešavanjima i memorijom radnog okruženja (nikad se ne commituju).

### Env prekidači

| prekidač | podrazumevano | šta radi |
|---|---|---|
| `PASADO_VECTORIZED_PRECISE` | `1` | vektorizovani precizni transformer (`precise_transformer.py`) |
| `PASADO_VEC_BOUNDARY` | `1` | vektorizovana provera nelinearne granice |
| `PASADO_BATCHED_LSTSQ` | `0` | jedan `gelsd` poziv po sloju umesto po neuronu |
| `PASADO_BATCHED_GRID` | `0` | batched 5x5 grid |
| `PASADO_REAL_CUBIC` | `0` | realni Viète kubni rešavač umesto kompleksnog Cardano-a |
| `PASADO_DEVICE` | `cpu` | uređaj za `get_lipschitz.py` (`--device`) |
| `PASADO_CLOSED_FORM_FIT` | `0` | batched put: A2 (`planar_fit.fit_planes`) eager umesto `gelsd` |
| `PASADO_COMPILED_FIT` | `0` | uz prethodni: `planar_fit.compiled_fit_planes()` (TorchInductor) |
| `PASADO_AUTO_PLANE_FIT` | `0` | batched put, samo CUDA: `gelsd` ispod praga, eager A2 od praga (f64 512, f32 640 boksova po pozivu) |
| `PASADO_PLANE_FIT_THRESHOLD` | nije postavljen | prepisuje prag automatskog izbora za oba dtype-a |

Sve merne skripte i harness-i postavljaju prvih pet na `True` (`set_production_flags`).
Kombinacija `PASADO_AUTO_PLANE_FIT` sa `PASADO_CLOSED_FORM_FIT` ili `PASADO_COMPILED_FIT`
baca `ValueError`.

- **Podrazumevano:** `gelsd` na CPU-u (preko mosta) u svakom putu.
- **Opt-in:** eager A2, kompajliran planar fit, automatski izbor G/A2.
- **Kompajliran ceo `precise_sigmoid` nije u proizvodnom kodu.** Postoji samo u
  pre-registrovanom harness-u workspace-a (`precise_sigmoid_instrumented` u
  `research_notes/compile_precise_sigmoid/colab_harness.py`).

---

## 3. Originalni problem

- **Haze put.** Za MNIST sliku `img` (784 piksela) posmatra se put zamagljivanja
  `x(t) = (1 - t)·img + t` za `t ∈ [0, ε]`; ε dolazi iz rasporeda
  `EPS_SCHEDULE = [2·10^(-k/4) for k in 2..17]` (16 vrednosti, indeks 0 je najveći ε ≈ 0,632).
- **Lipschitz bound.** Cilj je gornja granica `max_t |d logit / dt|` duž puta, za
  potpuno povezane sigmoid mreže `3layer`, `4layer`, `5layer` (širina 100) i `big`
  (širina 1024, 4 skrivena sloja). Pasado je računa forward-mode diferenciranjem kroz
  apstraktnu interpretaciju.
- **Dual zonotop.** Svaka veličina ima realni deo (vrednost) i dualni deo (izvod po t),
  oba kao zonotop: centar plus generatori, po jedan za svaki simbol šuma. Afini sloj je
  tačan (matmul na centrima i generatorima); nelinearnost dodaje nove simbole.
- **Precizni sigmoid transformer.** Za svaki neuron se iz zonotopa dobija boks
  `[lx, ux] × [ly, uy]` (x je pre-aktivacija, y njen izvod). Realni deo ide kroz linearnu
  relaksaciju sigmoida (λ, μ1, μ2). Dualni deo je `σ'(x)·y`: aproksimira se ravni
  `A x + B y + C` dobijenom regresijom najmanjih kvadrata na 5x5 mreži tačaka boksa, a
  greška `D = max |σ'(x)y - (Ax + By + C)|` nad boksom (uglovi plus koreni na y-ivicama,
  preko kubne jednačine) ulazi kao novi generator. Originalni kod dodaje šum
  `N(0, 1e-4)` na A i proverava `A != 0`. Sa n_splits (S) se `[0, ε]` deli na S
  podintervala i bound je maksimum po njima (tesnije granice).
- **Zašto `gelsd`.** Regresija mora da bude rešenje minimalne norme i sa otkrivanjem ranga,
  jer su mnogi boksovi degenerisani: u prvom sloju je `ly == uy` za svaki neuron, postoje i
  tačkasti boksovi. `gelsd` (LAPACK, SVD) to radi sa pravilom `s_i <= rcond·s_1`.
- **Zašto CUDA `gels` nije bezbedan.** PyTorch na CUDA-i ima samo `driver='gels'`: QR bez
  pivotiranja, pretpostavlja pun rang; na rank-deficient boksovima greši do 3,35e34
  (`results/optimization_section_5_4/GPU_READINESS.md`). Nema batched rank-revealing least
  squares u cuSOLVER-u/cuBLAS-u.
- **Zašto CPU most postaje usko grlo.** Hibrid radi sve na GPU-u osim `gelsd`-a, koji ide
  GPU → CPU → `gelsd` → GPU. Na `big` je taj most nosio 54,5% CUDA vremena (16.10.), a
  posle batching-a po splitovima 65% do 88% batched prolaza (24.10.), jer LAPACK rešava
  S·N malih problema u petlji.

---

## 4. Hronologija rada

Svaki korak: problem / promena / dokaz / rezultat / commit / folder rezultata.

1. **Vektorizacija preciznog transformera (15.9.).** Problem: Python petlje po neuronima.
   Promena: vektorizovani precise, boundary, batched lstsq i grid iza env selektora.
   Dokaz: `test_batched_grid.py`, `test_selector_paths.py`, `test_check_corners_tensor.py`,
   `test_check_nonlinear_boundary_tensor.py`, `test_compute_max_error_tensor.py`,
   `test_batched_lin_reg.py`. Commitovi `f8476dc`, `c0e05e3`, `8ebbe19`, `6705010`.
   Folder `results/optimization_section_5_4/`.
2. **Testovi i mutation harness (1.10. do 3.10.).** Problem: test grid-a je imao pogrešan
   exit kod; nije bilo dokaza da testovi mogu da padnu. Promena: strogi/informativni režim
   grid testa, docstringovi, `profile_precise_only.py`, `analyze_cubic_chain.py`,
   `mutation_check.py`, svi selektori postavljeni u benchmark varijantama, merenje šuma
   seed-a. Rezultat: neseedovan šum na A daje do 1,55e-5 relativno u `lc_precise`, pa
   referenca na 1e-10 traži seed. Commitovi `24df7fe`, `a1b7167` do `9a14099`, `87521c3`,
   `bb309d3`, `bd36a4d`, `c360485`, `dcefdc3`, `f62df48`.
3. **Seed/device/CLI (4.10.).** `get_lipschitz.py` dobija `--seed`, `--device`
   (`PASADO_DEVICE`), `--num-images`, `--precise-only`; CPU bit-identično pre i posle;
   `test_get_lipschitz_cli.py` (11 provera). Commitovi `708c5a8`, `f374eec`.
4. **GPU plumbing (4.10.).** Novi tenzori na uređaju tenzora sa kojima se kombinuju
   (`Duals.py`, `SimpleZono.py`); `check_device_plumbing.py`: ceo precizni prolaz na MPS-u,
   razlika 1,5e-7 (f32). Commitovi `ca122eb`, `792223c`.
5. **Dijagnoza CPU/CUDA razlike i realni Viète rešavač (5.10.).** Problem: na CUDA-i 3,9%
   razlike; kompleksni rešavač ostavlja imaginarne ostatke do 3,7e-8 i `torch.isreal` baca
   korene. Promena: `--real-cubic` (`PASADO_REAL_CUBIC`), merenje pomaka. Rezultat: realni
   rešavač daje CPU/CUDA 1,4e-16; pomak `lc_precise` naspram kompleksnog do 8,3e-6, oba
   validna prema autograd-u. Commitovi `d398d21`, `66efdad`, `5ec4d9b`, `2ce8c6b`.
6. **CPU `gelsd` most (5.10.).** `lin_reg_tensor_batched` rešava na CPU-u kad su ulazi na
   GPU-u i vraća rezultat na uređaj; CPU bit-identično. Commitovi `4ba249e`, `d20368f`,
   `dbbe7a3`.
7. **CPU/CUDA verifikacija (5.10. do 9.10.).** Puna matrica na A100-80GB: 4 mreže, 30
   slika, 16 ε, `28/28 PASS`; f64 najviše 1,786e-15, f32 8,3e-7; kompleksni rešavač kao
   negativna kontrola divergira 14% do 64%. Napomena: kod pre soundness popravki.
   Commitovi `f51f977`, `c2a2c65`, `d765ab3`, `125ddf2`. Folder
   `results/gpu_hybrid_verification_2026-10-08/`.
8. **Profilisanje fallback-a (16.10.).** `big`, precizni prolaz: 532,3 ms Colab CPU naspram
   63,3 ms CUDA hibrid (58,6 ms bez suvišnih sync-ova); CPU most 54,5% (`gelsd` 32,1 ms).
   Commitovi `a4dd045`, `03bd011`. Folder `results/gpu_hybrid_fallback_2026-10-16/`.
9. **Layerwise divergencija (17.10.).** Kompleksni rešavač na CUDA-i gubi 2348 do 2812 od
   6144 korena po sloju na `big`, izlaz se razlikuje do 65,2%; Viète najviše 1,41e-15.
   Commitovi `3b7b0a2`, `3b35688`. Folder `results/gpu_hybrid_layerwise_2026-10-17/`.
10. **Interval splitting (17.10.).** `--n-splits` sweep; vreme raste linearno sa S
    (sekvencijalna petlja). Commitovi `e894470`, `600ba2c`. Folder
    `results/gpu_hybrid_split_sweep_2026-10-17/`.
11. **Sweep do 64, peak VRAM, kompleksni kubni fix-evi (19.10.).** VRAM konstantan
    (10,78 MB `3layer`, 480,52 MB `big`); fiksni prag i Miné-ova procena oba zadržavaju
    3000/3000 korena. Commitovi `603a48c`, `25e6092`, `0eb14cd`.
12. **Upstream bagovi (20.10. do 22.10.).** Problem (nađen autograd proverom sweep-a):
    `Zonotope.__neg__` i `DualZonotope.__neg__` ne negiraju generatore, a
    `check_nonlinear_boundary*` i `_max_objective_over_x_candidates` uzimaju `max(E)` umesto
    `max(|E|)`; posledica, `lc_precise` ispod stvarnog izvoda (npr. 177,14 < 177,18).
    Promena: obe popravke, `check_sweep_soundness.py`, `verify_upstream_bugs.py`. Rezultat:
    posle popravke 264/264 reda iznad uzorkovanog maksimuma (pre: 13 + 6 ispod).
    Commitovi `b526762`, `def845d`, `69a7163`. Folder `results/soundness_fix_2026-10-22/`.
13. **PR #3 (21.10.).** Čista grana od `origin/main` sa 6 linija popravke
    (`Pasado-original`, `a99d70b`); otvoren, bez odgovora.
14. **Batching po splitovima (24.10. do 25.10.).** Promena: `batched_splits.py` (svih S
    podintervala u jednom prolazu: centri `[S, N]`, generatori `[S, G, N]`, boksovi
    spljošteni u `[S·N]`, `chunk` ograničava memoriju) i `test_batched_splits.py`.
    Rezultat na CUDA-i: svaka granica najviše 2,3e-14 od sekvencijalne; 7,5x na `3layer`
    pri S = 64 (7,0x / 6,7x na `4layer`/`5layer`), ali `big` samo 1,6x; memorija `big`
    503 MB → 29,8 GB pri S = 64. Commitovi `b144542`, `b0eb02d`, `842e9c9`, `4509e1f`.
    Folder `results/gpu_hybrid_batched_splits_2026-10-24/`.
15. **`gelsd` kao usko grlo (25.10.).** CPU `gelsd` nosi 65% (S = 1) do 88% (S = 64)
    batched prolaza na `big`, oko 34 ms po splitu. Otvoren zadatak zamene
    (`notes/PROMPT_GPU_REGRESSION_2026-10-25.md`).
16. **Kandidati G/A1/A2/A3/H/P i pre-registracija (25.10.).** G = `gelsd`, A1 = zatvorena
    forma sa tačnom detekcijom degeneracije, A2 = zatvorena forma sa `gelsd`-ovim rcond
    pravilom preko σ(K), A3 = separabilna forma (5 evaluacija σ'), H = zatvorena forma plus
    `gelsd` na označenim boksovima, P = `torch.linalg.pinv`. Harness
    `experiments/compare_lstsq_solvers.py` (+ `lstsq_candidates.py`, `lstsq_oracle.py`,
    `lstsq_contract.py`), reproducer `reproduce_f32_root_shortfall.py`. Commitovi `2ac9dfa`,
    `385397e`.
17. **Run v1 (25.10.).** Ishod: G pobeđuje, jer A1, A2 i H daju ne-konačne rezultate na
    `|y| = 1e300` (overflow suma). Commit `3bfee00`. Folder `results/lstsq_comparison_2026-10-25/`.
18. **Overflow popravka i run v2 (25.10.).** Skaliranje stepenom dvojke (`077ab86`);
    na 160 000 kontrolnih izlaza 159 682 bit-identično, 318 ranije ne-konačnih sada
    konačno. Ishod v2: A2 i H izjednačeni pre R4. Commitovi `077ab86`, `267c941`, `6e0acc6`.
    Folder `results/lstsq_comparison_2026-10-25_v2_scaled/`.
19. **Prvi CUDA run zaustavljen self-testom (26.10.).** S4 (centriranje oko fl(mean))
    preživeo; amandman: catch se broji samo gde roditelj prolazi, nova familija
    `offset_narrow`. Commitovi `67f4654`, `7740ba4`. Folder
    `results/lstsq_comparison_cuda_2026-10-26_selftest_stop/`.
20. **CUDA poređenje i izbor A2 (27.10.).** Primarno čitanje: A2 (A2 i H izjednačeni do R4,
    R5 bira A2 jer nema host sync); CUDA čitanje A3 kao sensitivity. Commit `e2a123b`.
    Folder `results/lstsq_comparison_cuda_2026-10-27/`.
21. **`planar_fit` u proizvodnji (27.10.).** `planar_fit.py` (A2) iza
    `PASADO_CLOSED_FORM_FIT`, testovi i mutanti, CUDA verifikacija: `big` S = 64 eager A2
    287,3 ms naspram 4023,9 sekvencijalno (14,0x) i 2485,2 batched `gelsd` (8,65x); male
    mreže pri S = 1 0,68x naspram batched `gelsd` (oko 3,8 ms launch overhead-a po pozivu).
    Commitovi `7c19c57`, `c2dcc4e`, `7e4cdcd`. Folder `results/planar_fit_cuda_2026-10-27/`.
22. **Inductor nad `fit_planes`, crossover, A3 (27.10.).** 303 → 5 CUDA kernela, 0,32 do
    0,45 ms po pozivu; `dynamic=True` pada; batch dimenzija postaje dinamička sama posle
    drugog oblika. Commit `79bd0e0`. Folder `results/inductor_dispatch_a3_2026-10-27/`.
23. **Pivot-centriranje targeta z i run v3 (27.10.).** Problem otkriven kompajliranjem:
    rezultat je zavisio od redosleda sumiranja na uskim boksovima. Promena: z se centrira
    oko tačke iz podataka pre suma (`500ae16`, reproducer
    `experiments/reproduce_target_centring.py`). v3: "Winner: A2 (decided by R5
    simplicity)"; R4 na A100: G 565 ms, A2 3,83 ms (n = 65536, f64). Commitovi `500ae16`,
    `8dfd82e`. Folder `results/lstsq_comparison_2026-10-27_v3_zpivot/`.
24. **Opt-in kompajliran planar fit (27.10. do 28.10.).** `PASADO_COMPILED_FIT`,
    `compiled_fit_planes(mode)` sa `compile_fallback`. A100-40GB, chunk 32: hladna
    kompilacija 31,33 s, topla 4,30 s; `big` S = 64: 312,35 ms naspram batched `gelsd`
    2555,40 (8,18x) i sekvencijalnog 4049,67 (12,97x); `reduce-overhead` prepisuje izlaze
    (aliasing), ne ide u proizvodnju. Commitovi `832323a`, `737e475`. Folder
    `results/v3_compiled_cuda_2026-10-28/`.
25. **Crossover `gelsd` naspram eager A2 (28.10.).** Pre-registrovano; pragovi stabilni u
    sve tri runde: f64 512, f32 640 boksova po pozivu. Commitovi `8837f68`, `f4b4a6f`.
    Folder `results/plane_fit_crossover_2026-10-28/`.
26. **Automatski izbor (28.10.).** `PASADO_AUTO_PLANE_FIT`; CUDA test 37/37. Commitovi
    `3243aef`, `1d0c49b`, `ae7c90a`. Folder `results/auto_plane_fit_cuda_2026-10-28/`.
27. **Eksplicitan tensor šuma (28.10.).** Problem: Dynamo je prevodio `np.random.normal` iz
    `_perturbation` u graf i trošio NumPy stanje drugačije od eager-a. Promena:
    `_forward_chunk` vuče šum istim redosledom i prosleđuje ga
    `precise_sigmoid(z, noise)`; `A != 0` se proverava jednom po chunk-u. Commitovi
    `8e3d0df`, `a57b560`.
28. **Kompajliran ceo `precise_sigmoid`, v1 (28.10.).** Pre-registracija workspace
    `788589d` (bivši `91e02e4`), harness `df0e88f5...`. Gate stao: 868/876; G4 je zavisio
    od istorije oblika (prvi sloj u prvom prolazu na statičkom grafu, kasnije na
    dinamičkom), G5 je jednom normwise metrikom merio ceo `dg`. Arhiva `1ac62f9`, folder
    `results/compiled_sigmoid_cuda_2026-10-28_gate_stop/`.
29. **v2 (29.10.).** Pre-registracija `7c9a8ef` (bivši `e18278c`), harness `9e8535ff...`.
    `mark_dynamic` i G5 po blokovima uspeli (G4 96/96, jedan graf po konfiguraciji), svi
    gate-ovi granica prošli, ali D ugovor nije izvršen: dijagnostički graf (drugi, drugačije
    fuzionisan) nikad nije bio bit-identičan merenom. Arhiva `aff244b`, folder
    `results/compiled_sigmoid_v2_cuda_2026-10-29_gate_stop/`.
30. **v3 (29.10.).** Pre-registracija `85de132`, harness `898980d9...`; kandidat je
    instrumentovana kopija `precise_sigmoid` koja sama vraća A, B, C, boks i D. Sve provere
    korektnosti prolaze, ishod `opt-in`. Arhiva `1d754c9`, folder
    `results/compiled_sigmoid_v3_cuda_2026-10-29/`.

---

## 5. Matematika A2

- **Mreža.** Za boks se uzima 5 vrednosti x (`linspace(lx, ux)`) i 5 vrednosti y; 25 tačaka
  `(x_i, y_j)` su Dekartov proizvod (x ide najsporije). Dizajn `X = [1, x, y]` je 25x3,
  target `z = σ'(x)·y`, računat kao `(1 - σ(x))·(σ(x)·y)`. Traže se `(C, A, B)`.
- **Ortogonalnost centriranih kolona.** Sa `x = x̄ + dx`, `y = ȳ + dy`: kolone `1`,
  `dx⊗1`, `1⊗dy` su međusobno ortogonalne jer je `Σ dx_i = 0` i `Σ dy_j = 0` (ravnomeran
  razmak nije potreban). Normalne jednačine se razdvajaju:
  `A = Σ dx·dz / Σ dx²`, `B = Σ dy·dz / Σ dy²`, `C = z̄ - A·x̄ - B·ȳ` (sume po 25 tačaka).
  To je tačno LS rešenje, ne aproksimacija.
- **K matrica.** `X = Q·(5K)`, Q sa ortonormiranim kolonama,
  `K = [[1, x̄, ȳ], [0, σx, 0], [0, 0, σy]]`, `σx² = (1/5)Σ(x_i - x̄)²`. Zato su singularne
  vrednosti `X` tačno `5·sing(K)`: rang se odlučuje iz 3x3 matrice, bez SVD-a.
- **Rank cutoff kao `gelsd`.** `gelsd` odbacuje `s_i <= rcond·s_1`, `rcond = eps·max(m, n)
  = 25·eps` (5,55e-15 u f64, 2,98e-6 u f32; `planar_fit.default_rcond`). Sopstvene
  vrednosti `(5K)ᵀ(5K)` dobijaju se iz invarijanti (sume nenegativnih članova, visoka
  relativna tačnost): `e1 = 25(1 + x̄² + ȳ² + σx² + σy²)`,
  `e2 = 625[σx²(1 + ȳ²) + σy²(1 + x̄²) + σx²σy²]`, `e3 = 15625·σx²σy²`; λ1 sa 8 Newton
  koraka od `e1`, pa `q = e3/λ1`, `p = (e2 - q)/λ1`, `λ2 = (p + √(p² - 4q))/2`,
  `λ3 = q/λ2`. Osa se odbacuje kad je `λ <= rcond²·λ1`; sa dve male obe, sa jednom ona čiji
  je raspon manji u odnosu na pomak od nule (`gram_eigenvalues`, `gelsd_cutoff_axes`).
  Ovo je pogodilo `gelsd`-ov rang na 1400/1400 boksova u f64 i f32.
- **Degenerisani slučajevi (minimalna norma).** x konstantno (`x0`): `k = z̄ - B·ȳ`,
  `C = k/(1 + x0²)`, `A = k·x0/(1 + x0²)`; y simetrično; obe ose:
  `(C, A, B) = z̄·(1, x0, y0)/(1 + x0² + y0²)`. Degeneracija se detektuje tačno po krajevima
  (`lx == ux`), ne po `Σ dx² == 0` (srednja vrednost 25 jednakih brojeva nije tačno taj
  broj), a odsečena osa se tretira kao konstantna (`x0 := x̄`).
- **Skaliranje stepenom dvojke** (`077ab86`, `_pow2_scale`): centrirani x, y i z se dele
  sa `2^e` blizu svog maksimuma pre suma kvadrata i proizvoda, pa se množe nazad. Deljenje
  stepenom dvojke je tačno dok međurezultat nije subnormalan, pa su bitovi isti gde su stare
  sume bile konačne, a nema overflow-a (bivši v1 problem na `|y| = 1e300`).
- **Pivot-centriranje x, y i z** (`_centred`, `500ae16`): prvo `v - v[:, :1]` (razlika do
  tačke iz podataka je tačna za uzak boks daleko od nule, Sterbenz), pa oduzimanje srednje
  vrednosti malih razlika, `x̄ = v0 + mean(v - v0)` (Chan, Golub, LeVeque 1983). Centriranje
  oko `fl(mean)` unosi `eps·|mean|` u svaku tačku (mutant S4). Targeti z se centriraju isto,
  inače rezultat zavisi od redosleda sumiranja (otkriveno kompajliranjem).
- **Zašto ne naivne normalne jednačine.** Formiranje `XᵀX` kvadrira uslovni broj i razrešava
  singularne vrednosti tek do oko `√u·s1 ≈ 1,5e-8·s1`, šest redova iznad `gelsd`-ovog praga,
  pa ne može da reprodukuje `gelsd` na skoro degenerisanim boksovima. Razdvojene centrirane
  sume sa pivotom imaju uslovni broj oko √3.
- **Poznata razlika prema `gelsd`-u.** Blizu praga `gelsd` radi projekciju skraćenog SVD-a, a
  A2 tretira odsečenu osu kao konstantnu: blizu, ali ne identično (4,8e-9 na odsečenim
  `mixed_scale` boksovima). Na samom prelazu ranga rešenje skače za O(1), kako mora. Na
  uskim boksovima A2 je tačniji od `gelsd`-a (greška `gelsd`-a raste otprilike sa 1/w², A2
  sa 1/w, prema egzaktnom racionalnom rešenju).

---

## 6. Numerička verifikacija

- **Fraction orakl** (`lstsq_oracle.exact_min_norm`): tačno racionalno rešenje minimalne
  norme na istim float vrednostima mreže i targeta, rang odlučen tačno.
- **mpmath tačan maksimum** (`lstsq_oracle.true_max_error`, 50 cifara): uglovi i koreni
  `2s³ - 3s² + s - A/y0 = 0` na y-ivicama.
- **Ugovor C1-C7** (`lstsq_contract.py`): C1 oblik, dtype, konačnost; C2 nezavisnost od
  batch-a; C3 skaliranje z stepenom dvojke bit po bit; C4 zamena osa; C5 normalne jednačine
  (rezidual tačno); C6 minimalna norma na degenerisanim osama; C7 rastojanje do orakla.
- **Familije boksova:** sintetičke (`normal`, `degenerate_x`, `degenerate_y`, `point`,
  `near_degenerate`, `saturated`, `mixed_scale`, pojasevi oko praga, `large_magnitude`,
  `offset_narrow` za self-test) i boksovi snimljeni iz pravih prolaza (2000), float32 i
  float64.
- **Uređaji:** CPU (Mac), MPS (plumbing, f32; `frexp` na MPS-u preko CPU-a), CUDA (A100
  40GB i 80GB preko Colab-a).
- **Mutacioni testovi:** `experiments/mutation_check.py` M1 do M29 (M15 do M29 za batched,
  closed form, auto izbor, šum i `A != 0`); self-test S1 do S10 u poređenju solvera; 10
  mutanata gate-a v3 (lokalno, svi se ponašaju kako treba).
- **Uzorkovani autograd:** `check_sweep_soundness.sampled_max_derivative` (1001 ili 4001
  tačaka duž puta); svaki bound mora biti iznad.
- **Pre-registracija:** poređenje solvera (`385397e`), crossover (`8837f68`), kompajliran
  `precise_sigmoid` v1/v2/v3 (workspace `788589d`, `7c9a8ef`, `85de132`).
- **Pronađeni bagovi i kako:**
  1. kompleksni kubni rešavač gubi korene na CUDA-i (CPU/CUDA razlika, `diagnose_device_diff.py`);
  2. upstream `__neg__` i `max(E)` (autograd soundness provera sweep-a);
  3. `net.to(...)` je pomerao bias `Parameter`-e (Colab; `extract_affine_layers` sada `detach()`);
  4. overflow zatvorenih formi na `|y| = 1e300` (run v1, gate G2);
  5. self-test S4 preživeo (prvi CUDA run zaustavljen; amandman `67f4654`);
  6. zavisnost od redosleda sumiranja targeta (kompajliran naspram eager; `500ae16`);
  7. `reduce-overhead` aliasing izlaza (v3 compiled sveska);
  8. NumPy šum preveden u kompajlirani graf (eksplicitan tensor šuma, `8e3d0df`);
  9. graf prvog sloja zavisan od istorije oblika (compiled sigmoid v1, G4; `mark_dynamic`);
  10. normwise metrika nad celim `dg` dominirana D-om (v1, G5; podela po blokovima);
  11. dva različito fuzionisana grafa nikad bit-identična na CUDA-i (v2; instrumentovani kandidat).
- **D i floating point.** U v3 (S = 64, eps indeks 8, 4992 boksa) eager D je ispod tačnog
  maksimuma na 51,3% boksova, kompajliran na 47,7%; najveći manjak relativno na W je
  3,4e-9 (eager) i 3,9e-9 (kompajliran), apsolutno oko 2,5e-14, a u odnosu na veličinu
  članova greške najviše 2,4 ulp-a (medijana 0,2). U f32 manjak je veći
  (`reproduce_f32_root_shortfall.py`). **Pasado nije formalno floating-point sound:** D se
  računa u floating point-u, sa kancelacijom i bez outward rounding-a. Uzorkovani autograd
  i pre-registrovani D/W ugovor prolaze.

---

## 7. Finalni proizvodni kod i režimi

- **`forward_mode_tensorized_src/planar_fit.py`**: `fit_planes(grid, z, rcond=None)` = A2
  (vraća `[n, 3]` = (C, A, B), elementwise, bez LAPACK-a, host sync-a i transfera; isti
  bitovi kao kandidat A2); pomoćne `default_rcond`, `_centred`, `_pow2_scale`, `_stats`,
  `gram_eigenvalues`, `gelsd_cutoff_axes`; `compiled_fit_planes(mode=None)`:
  `torch.compile(fit_planes, fullgraph=True)`, keširano po modu; ako kompilacija padne
  (npr. nema Triton-a), upozorava jednom i dalje radi eager, a razlog upisuje u
  `compile_fallback[mode]`.
- **`forward_mode_tensorized_src/batched_splits.py`**: `split_bounds(layers, img, epsilon,
  n_splits, seed=None, chunk=None)` i `lipschitz_bound(...)`; `_forward_chunk` (vuče šum
  `_perturbation` istim redosledom kao sekvencijalni kod, proverava `A != 0` jednom po
  chunk-u); `precise_sigmoid(z, noise)` → `(BatchedDualZonotope, a_nonzero)`;
  `plane_fit_for(grid)` bira solver iz prekidača; `extract_affine_layers`, `hazed_input`,
  `affine`.
- **Automatski izbor**: samo CUDA (`AUTO_PLANE_FIT_DEVICES = ("cuda",)`), prag po broju
  boksova u pozivu `AUTO_PLANE_FIT_THRESHOLD = {float64: 512, float32: 640}` (A100
  crossover); ispod praga `gelsd`, od praga eager A2; na drugim uređajima `gelsd`;
  `PASADO_PLANE_FIT_THRESHOLD` prepisuje prag.
- **Chunk**: `chunk` ograničava broj splitova po prolazu; broj boksova po pozivu je
  `S_chunk·N`, pa chunk utiče i na automatski izbor. U Colab sveskama chunk za `big` ide iz
  VRAM-a: 32 na 40 GB, 64 na 80 GB.
- **Kompajliran planar fit**: opt-in, prva dva oblika se kompajliraju (sekunde), posle je
  batch dimenzija dinamička; `reduce-overhead` se ne koristi (aliasing).
- **Default ostaje `gelsd`.** Kompajliran ceo sigmoid je samo eksperiment (harness u
  workspace-u), nije u biblioteci.

---

## 8. Finalne performanse

Napomena o okruženjima: runovi su na različitim A100 mašinama i sa različitim chunk-om, pa
apsolutna vremena nisu direktno uporediva preko redova. 24.10. (verovatno 80 GB, chunk 64),
27.10. A100-80GB chunk 64, 28.10. v3 compiled A100-40GB chunk 32, 29.10. v3 compiled
sigmoid A100-80GB chunk 64.

**A. Sekvencijalni naspram batched splitova** (`gpu_hybrid_batched_splits_2026-10-24`, eps 0):
male mreže 7,5x / 7,0x / 6,7x pri S = 64 (`3layer` / `4layer` / `5layer`); `big` samo 1,6x
dok je `gelsd` 88% prolaza.

**B. CPU `gelsd` naspram A2** (`planar_fit_cuda_2026-10-27`, A100-80GB, chunk 64): `big`
S = 64 eager A2 287,3 ms naspram 4023,9 sekvencijalno (14,0x) i 2485,2 batched `gelsd`
(8,65x); S = 32 12,98x. Raspon 12x do 14x za `big` pri S = 32 i 64. Male mreže pri S = 1:
0,68x (launch overhead eager A2-a). Solver sam (n = 65536, f64): `gelsd` 565 ms, A2 3,83 ms.

**C. Kompajliran fit naspram kompajliranog celog sigmoida.**
- `v3_compiled_cuda_2026-10-28` (A100-40GB, chunk 32): kompajliran A2 na `big` S = 64
  312,35 ms naspram batched `gelsd` 2555,40 (8,18x) i sekvencijalnog 4049,67 (12,97x).
- `compiled_sigmoid_v3_cuda_2026-10-29` (A100-80GB, chunk 64), medijane (ms):

| mreža, S | sekv. G | batched G | eager A2 | kompajliran fit | kompajliran ceo sigmoid | naspram batched G | naspram sekv. |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3layer, 1 | 12,412 | 10,658 | 15,854 | 8,894 | 2,445 | 4,36x | 5,08x |
| 3layer, 64 | 988,714 | 106,938 | 22,218 | 12,144 | 3,429 | 31,19x | 288,36x |
| 5layer, 1 | 29,754 | 26,065 | 42,422 | 22,294 | 4,822 | 5,41x | 6,17x |
| big, 1 | 67,721 | 57,556 | 43,552 | 23,212 | 10,643 | 5,41x | 6,36x |
| big, 64 | 4589,672 | 2298,359 | 283,646 | 274,980 | 220,565 | 10,42x | 20,81x |

- **Kerneli** (isti run): 3layer S = 1 po prolazu 481 (batched G) → 65; `big` S = 64
  977 → 103; po sigmoid sloju oko 12 do 13 naspram oko 223 kod kompajliranog fit-a.
  Brojevi važe za to okruženje (torch 2.11, CUDA 13.0, Triton 3.6).
- **VRAM** `big` S = 64: 29803,89 MiB (batched G, eager A2, kompajliran fit) → 20543,89 MiB
  (oko 31% manje). Batched memorija i dalje raste linearno sa S; chunk je ograničava.
- **Kompilacija i break-even** (kompajliran ceo sigmoid, male mreže pri S = 1): hladno
  16,8 do 16,9 s, toplo 4,7 do 4,8 s, jedan graf po konfiguraciji. Break-even naspram
  najboljeg puta bez kompilacije: toplo 582,66 / 295,53 / 221,65 prolaza, hladno 2048,06 /
  1049,82 / 795,78 (`3layer` / `4layer` / `5layer`). Kompajliran fit za poređenje: oko
  14,8 s hladno, 2,2 s toplo.

---

## 9. Pre-registrovani ishodi

| run | pre-registracija | ishod | folder |
|---|---|---|---|
| poređenje `gelsd` zamena v1 | `385397e` | G pobeđuje zbog overflow-a zatvorenih formi | `lstsq_comparison_2026-10-25/` |
| v2 (skaliranje) | isto pravilo | A2 i H izjednačeni pre R4 | `lstsq_comparison_2026-10-25_v2_scaled/` |
| prvi CUDA run | isto pravilo | zaustavljen self-testom (S4) | `lstsq_comparison_cuda_2026-10-26_selftest_stop/` |
| CUDA poređenje | isto pravilo | A2 (primarno, R5), A3 sensitivity | `lstsq_comparison_cuda_2026-10-27/` |
| v3 (z pivot) | isto pravilo | A2 pobeđuje po R5 | `lstsq_comparison_2026-10-27_v3_zpivot/` |
| crossover | `8837f68` | pragovi 512 (f64), 640 (f32) | `plane_fit_crossover_2026-10-28/` |
| compiled sigmoid v1 | workspace `788589d` | zaustavljen gate-om (G4, G5) | `compiled_sigmoid_cuda_2026-10-28_gate_stop/` |
| compiled sigmoid v2 | workspace `7c9a8ef` | zaustavljen (D identitet dva grafa) | `compiled_sigmoid_v2_cuda_2026-10-29_gate_stop/` |
| compiled sigmoid v3 | workspace `85de132` | `opt-in` | `compiled_sigmoid_v3_cuda_2026-10-29/` |

Svaki raniji rezultat ostaje kakav je zapisan; nijedan se ne prepisuje. Popravka ide u nov
commit, ponovljen run u nov folder, a README novog run-a navodi stari.

---

## 10. Upstream i plan PR-ova

- PR #3 (dva baga) je jedini otvoren ka upstream-u; čeka pregled od 21.10.
- `gpu-hybrid` je istraživačka grana (89 commitova, rezultati, logovi, eksperimenti); ne
  merge-uje se u upstream kao takva.
- Upstream PR-ovi moraju biti mali i čisti: za svaki nova grana od `origin/main`, fajlovi
  preuzeti sa `gpu-hybrid` (`git checkout gpu-hybrid -- <fajlovi>`), bez cherry-pick-a cele
  istorije. `results/`, logovi, sveske, beleške i profilisanje ne idu upstream.
- Predloženi redosled tema: (1) bagovi (PR #3), (2) vektorizovani precizni transformer i
  testovi, (3) podrška za uređaje, (4) realni Viète rešavač, (5) interval splitting i
  batching, (6) `planar_fit`.
- Ništa se ne otvara bez dogovora sa Sašom Misailovićem (mentor, UIUC), koji treba prvo da
  kaže da li upstream uopšte želi GPU i splitting deo.

---

## 11. Trenutna ograničenja

- Pokrivene su samo potpuno povezane sigmoid mreže na MNIST-u i haze perturbacija.
- Pragovi automatskog izbora (512 / 640) izmereni su samo na A100.
- Formalni kompajliran kandidat je instrumentovan (vraća dijagnostiku); njegova cena je u
  merenjima.
- Hladna kompilacija (oko 17 s) se ne isplati u jednom standardnom sweep-u na malim
  mrežama pri S = 1; zato je ishod `opt-in`.
- Pasado nije formalno floating-point sound (D na uskim boksovima, f32 koreni).
- Verzija bez dijagnostičkih izlaza (ablacija) traži sopstvenu verifikaciju; nije proverena
  za proizvodnju.
- CNN i druge aktivacije nisu obrađene.
- TVM i XLA nisu započeti.

---

## 12. Indeks artefakata

**Izvorni kod (Pasado):** `forward_mode_tensorized_src/planar_fit.py`,
`forward_mode_tensorized_src/batched_splits.py`, `forward_mode_tensorized_src/precise_transformer.py`
(referenca), `SimpleZono.py`, `Duals.py`, `Section_5_4/get_lipschitz.py`.

**Testovi i harness-i (`Pasado/experiments/`):** `test_batched_splits.py`, `test_planar_fit.py`,
`test_auto_plane_fit.py`, `test_batched_lin_reg.py`, `test_get_lipschitz_cli.py`,
`test_benchmark_variants.py`, `test_soundness.py`, `mutation_check.py`,
`compare_lstsq_solvers.py`, `lstsq_candidates.py`, `lstsq_oracle.py`, `lstsq_contract.py`,
`measure_plane_fit_crossover.py`, `measure_split_sweep.py`, `check_sweep_soundness.py`,
`verify_upstream_bugs.py`, `reproduce_f32_root_shortfall.py`, `reproduce_target_centring.py`.

**Komande (iz `Pasado/`, `.venv/bin/python`):**

```bash
python experiments/test_batched_splits.py --quick
python experiments/test_batched_splits.py [--closed-form | --compiled] [--device cuda]
python experiments/test_planar_fit.py [--compiled] [--device cuda]
python experiments/test_auto_plane_fit.py [--device cuda]
python experiments/mutation_check.py
python experiments/verify_upstream_bugs.py
git log --oneline fork/gpu-hybrid..HEAD   # treba da bude prazno
```

**Sveske (koren workspace-a):** `COLAB_GPU_HYBRID_VERIFICATION.ipynb`, `..._FALLBACK`,
`..._LAYERWISE`, `..._SPLIT_SWEEP`, `..._BATCHED_SPLITS`, `COLAB_LSTSQ_COMPARISON.ipynb`,
`COLAB_PLANAR_FIT_VERIFICATION.ipynb`, `COLAB_INDUCTOR_DISPATCH_A3.ipynb`,
`COLAB_V3_COMPILED.ipynb`, `COLAB_PLANE_FIT_CROSSOVER.ipynb`, `COLAB_AUTO_PLANE_FIT_CUDA.ipynb`,
`COLAB_COMPILED_PRECISE_SIGMOID.ipynb` (v3, commitovana; generiše je
`research_notes/compile_precise_sigmoid/build_notebook.py` iz `colab_harness.py`).

**Colab postupak:** sveska klonira `gpu-hybrid` sa fork-a i radi checkout pinovanog commita,
proverava HEAD; harness se upisuje kroz `%%writefile` i proverava njegov sha256; korisnik
bira A100, "Disconnect and delete runtime", "Run all"; na kraju se preuzima zip. Ako se kod
promeni, sveska se mora ponovo generisati i ponovo uploadovati.

**Rezultati (`Pasado/results/`):** `optimization_section_5_4/`,
`gpu_hybrid_verification_2026-10-08/`, `gpu_hybrid_fallback_2026-10-16/`,
`gpu_hybrid_layerwise_2026-10-17/`, `gpu_hybrid_split_sweep_2026-10-17/`,
`soundness_fix_2026-10-22/`, `gpu_hybrid_batched_splits_2026-10-24/`,
`lstsq_comparison_2026-10-25/`, `lstsq_comparison_2026-10-25_v2_scaled/`,
`lstsq_comparison_cuda_2026-10-26_selftest_stop/`, `lstsq_comparison_cuda_2026-10-27/`,
`planar_fit_cuda_2026-10-27/`, `inductor_dispatch_a3_2026-10-27/`,
`lstsq_comparison_2026-10-27_v3_zpivot/`, `v3_compiled_cuda_2026-10-28/`,
`plane_fit_crossover_2026-10-28/`, `auto_plane_fit_cuda_2026-10-28/`,
`compiled_sigmoid_cuda_2026-10-28_gate_stop/`, `compiled_sigmoid_v2_cuda_2026-10-29_gate_stop/`,
`compiled_sigmoid_v3_cuda_2026-10-29/`.

**Izveštaji i beleške (workspace):** `reports/GPU zamena za gelsd.md` (objedinjen izveštaj o
zameni), `research_notes/GPU zamena za gelsd/` (pet beleški po pravcima i `scripts/`),
`research_notes/compile_precise_sigmoid/` (harness, builder, lokalne analize),
`notes/PLAN_COMPILE_PRECISE_SIGMOID_2026-10-28.md`, `notes/PROMPT_GPU_REGRESSION_2026-10-25.md`,
`notes/DAILY_LOG.md`, stari handoff-ovi `notes/HANDOFF_GPU_HYBRID_2026-10-{10,22,25,28}.md`.

**Važni commitovi:** Pasado `1d754c9` (HEAD, arhiva v3), `aff244b` (arhiva v2), `1ac62f9`
(arhiva v1), `a57b560` (kod na koji su pinovani compiled sigmoid run-ovi), `3243aef`
(automatski izbor), `832323a` (kompajliran fit), `500ae16` (z pivot), `7c19c57`
(`planar_fit` u proizvodnji), `077ab86` (skaliranje), `385397e` (pre-registracija
poređenja), `b144542` (batching), `b526762` (upstream bagovi). Workspace `85de132`
(pre-registracija v3), `7c9a8ef` (v2), `788589d` (v1).

---

## 13. Trenutni status i pauza

- Eksperiment sa kompajliranjem celog `precise_sigmoid` je završen i arhiviran.
- Pre-registrovani ishod je `opt-in`: tehnički rezultat je jako pozitivan (velika
  steady-state korist, manje kernela i memorije), ali kompilacija se ne amortizuje u svakom
  slučaju u jednom standardnom sweep-u. Default se ne menja.
- Ništa dalje se trenutno ne implementira.
- Otvorene stavke, samo kao spisak: TVM i XLA na istoj funkciji (Sašin predlog: `big`,
  `n_splits = 32`); verifikacija verzije bez dijagnostičkih izlaza; potpuna
  floating-point popravka D-a; generalizacija na CNN i druge aktivacije; pakovanje za
  publikaciju; upstream PR-ovi posle dogovora sa Sašom.
- Sledeća radna sesija ne sme automatski da započne nijednu od ovih stavki bez novog,
  izričitog zahteva korisnika.
