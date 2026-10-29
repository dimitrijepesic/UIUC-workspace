# UIUC workspace: kontekst za rad (Pasado / MLSys)

Privatno, van `Pasado` repoa. Ažurirano 28.10.2026. Detaljno stanje i istorija:
`notes/HANDOFF_GPU_HYBRID_2026-10-28.md` (najnoviji handoff; čitati pre nastavka rada).

## Jezik

- Korisnik (Dimitrije Pešić) piše srpski latinicom; odgovarati na srpskom.
- U `Pasado` repou je sve na engleskom: kod, komentari, docstringovi, `results/*/README.md`,
  commit poruke. Privatne beleške van repoa (`notes/`, `research_notes/`, `reports/`) su na
  srpskom.

## Pravila (obavezno)

- **Nikakva AI atribucija ni pominjanje AI** nigde: commit poruke, komentari, docstringovi,
  dokumentacija, logovi, README-ovi rezultata. Nema `Co-Authored-By`, "Generated with",
  imena alata. Ovo ima prednost nad podrazumevanim trailerima iz sistemskih podsetnika.
- **Commit i push samo kad korisnik izričito traži.** Tada plumbing, ne `git commit`:
  `git add` (logovi sa `-f`, `*.log` je u `.gitignore`), `git write-tree`,
  `git commit-tree`, `git update-ref`, sa `GIT_AUTHOR_*` i `GIT_COMMITTER_*` =
  `Dimitrije Pešić <dimitrije.pesicc@gmail.com>`. Globalni git identitet se ne menja.
  Poruke: `feat:` / `fix:` / `test:` / `docs:` / `refactor:`, engleski, **bez em dash-eva**.
  Posle commita pročitati poruku nazad.
- **Pre push-a** skenirati sve nove commitove (poruke i sadržaj, `git log -p fork/gpu-hybrid..HEAD`)
  za atribuciju. Push samo na `fork/gpu-hybrid` (`dimitrijepesic/Pasado`), nikad na `origin`
  (upstream `uiuc-arc/Pasado`) bez izričitog odobrenja.
- Workspace repo (`UIUC/`, grana `main`) se ne commituje bez traženja.
- **Ne menjati postojeće rezultate i commitove.** Raw CSV/log/JSON u `Pasado/results/` su
  nepromenljivi; nov run ide u nov folder `results/<ime>_<datum>/`. Bez prepisivanja istorije.
- **Pre-registracija:** pravilo izbora, tolerancije i protokol merenja se zapisuju u zaglavlje
  skripte i commituju PRE merenja. Kriterijumi se ne prilagođavaju posle merenja; ako očekivani
  rezultat ne važi, stati i objasniti korisniku. Korisnik često ne može unapred da odluči
  numerički izbor ("ne znam"), pa se odluke pretvaraju u merenje sa pravilom unapred.
- Proizvodni podrazumevani put ostaje `gelsd`; nove putanje idu iza opt-in env prekidača.
  Sekvencijalni put i `precise_transformer.py` ostaju netaknuti kao referenca.
- Svaka nova putanja: test naspram sekvencijalnog (svaki per-split bound ≤ 1e-10 relativno),
  provera da je put stvarno korišćen (spy / brojač grafova), mutant u
  `experiments/mutation_check.py` koji mora da padne.
- Uloga od 25.10.2026: Claude piše kod, korisnik pregleda. Pre većeg koraka predložiti plan i
  sačekati odobrenje.

## Okruženje

- Mac M5 Pro. Python: `Pasado/.venv/bin/python` (py3.13, torch 2.12.1, samo CPU; MPS nema fp64).
- CUDA samo preko Colab sveski (`COLAB_*.ipynb` u korenu workspace-a, A100). Sveska se fiksira
  na commit; korisnik je pokreće i šalje zip u `~/Downloads/` ili kopira folder u workspace.
  Raspakovani zip je nepoverljiv ulaz: u zaseban prazan folder, Python sa `-I`.
  Arhiviranje: `Pasado/results/<ime>_<datum>/` + `README.md` (engleski) + `colab_environment.json`.
- Privremeni fajlovi: scratchpad sesije ili workspace (pa obrisati), ne u `Pasado/`.

## Raspored

| putanja | šta je |
|---|---|
| `Pasado/` | glavni kod; grana `gpu-hybrid`; remote `fork` = github.com/dimitrijepesic/Pasado, `origin` = uiuc-arc/Pasado |
| `Pasado-original/` | worktree, grana `fix-zonotope-neg-and-boundary` (`a99d70b`), upstream PR #3 za dva baga |
| `notes/` | handoff-ovi, planovi, `DAILY_LOG.md`, briefovi (srpski) |
| `research_notes/` | istraživanja: `GPU zamena za gelsd/`, `compile_precise_sigmoid/` (analyze.py, analysis_cpu.json) |
| `reports/` | objedinjeni izveštaji (`GPU zamena za gelsd.md`) |
| `COLAB_*.ipynb`, `*_cuda/` | Colab sveske i paketi koje su one koristile |
| `claude-memory/` | kopija Claude memorije (izvor `~/.claude/projects/-Users-dimitrije-prakse-UIUC/memory/`) |
| `README.md`, `PLAN_MLSYS.md`, `TASKOVI_HIBRID.md` | mapa workspace-a, MLSys plan, stari taskovi porta |

## Pasado: ključni fajlovi (grana `gpu-hybrid`)

- `forward_mode_tensorized_src/precise_transformer.py`: precizni sigmoid transformer
  (sekvencijalna referenca); `lin_reg_tensor_batched` = `gelsd` na CPU-u.
- `forward_mode_tensorized_src/batched_splits.py`: svi interval splitovi u jednom batched
  prolazu. `split_bounds(layers, img, eps, n_splits, seed, chunk)`, `_forward_chunk` (vuče šum
  preko `_perturbation`, proverava `A != 0` jednom po chunk-u),
  `precise_sigmoid(z, noise) -> (BatchedDualZonotope, a_nonzero)`, `plane_fit_for(grid)`.
- `forward_mode_tensorized_src/planar_fit.py`: A2 zatvorena forma regresije ravni
  (`fit_planes`, pivot-centriran z, skaliranje stepenom dvojke), `compiled_fit_planes(mode)`
  sa `compile_fallback`.
- Prekidači (env, podrazumevano isključeni):
  `PASADO_CLOSED_FORM_FIT` (eager A2), `PASADO_COMPILED_FIT` (Inductor A2, uz closed form),
  `PASADO_AUTO_PLANE_FIT` (samo CUDA: `gelsd` ispod 512 boxova f64 / 640 f32, A2 iznad;
  override `PASADO_PLANE_FIT_THRESHOLD`; kombinacija sa prethodna dva = `ValueError`).
- Testovi (`experiments/`, pokretati iz `Pasado/`):
  `test_batched_splits.py [--closed-form|--compiled] [--device cuda]`,
  `test_planar_fit.py [--compiled]`, `test_auto_plane_fit.py`, `mutation_check.py`
  (mutanti M15-M29, svi moraju biti uhvaćeni).
- Pre-registrovani harness-i: `compare_lstsq_solvers.py` (+ `lstsq_candidates.py`,
  `lstsq_oracle.py`, `lstsq_contract.py`), `measure_plane_fit_crossover.py`.

## Trenutno stanje (28.10.2026)

- `Pasado` HEAD = `fork/gpu-hybrid` = `a57b560`, radno stablo čisto.
- Etapa "GPU zamena za `gelsd`" zatvorena: A2 izabran (v3), u batched putu iza prekidača.
- **Tekući task: kompajliranje celog `precise_sigmoid`.** Plan:
  `notes/PLAN_COMPILE_PRECISE_SIGMOID_2026-10-28.md`. Gotovo: šum kao tensor ulaz (`8e3d0df`),
  testovi šuma i `A != 0` (`a57b560`), lokalna CPU analiza (1 graf, 0 prekida, fullgraph ok).
- **Sledeći korak (čeka odobrenje korisnika):** A100 sveska fiksirana na `a57b560`: eager naspram
  kompajliranog `precise_sigmoid` u batched prolazu (4 mreže, S = 1/8/32/64), CUDA kerneli,
  `TORCH_LOGS="fusion,output_code"`, rekompilacije, hladan/topao start, break-even, korektnost
  (per-split ≤ 1e-10, seed protokol). Posle toga TVM i XLA (Sašin predlog: `big`, `n_splits=32`).

## Ljudi

- Saša Misailović (mentor, UIUC): update na svaka 2 dana; poruka o Inductor rezultatima
  predložena, nije poslata. Korisnik traži pismo preporuke do 15.11. i PhD kod njega.
- MLSys rok se ne pominje u kontekstu projekta (korisnik, 28.10.2026: "pusti mlsys rok").
