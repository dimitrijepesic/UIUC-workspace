# Zatvorena forma nadmašuje `gelsd`, ali ne bit po bit

**Gotova batched GPU zamena za LAPACK `gelsd` ne postoji nigde, ali vašem problemu nije ni potrebna: pošto je 5×5 mreža Kartezijanski proizvod, zatvorena forma je tačno rešenje najmanjih kvadrata, tačnija od samog `gelsd`-a, a `gelsd`-ovo rcond pravilo ranga može se reprodukovati elementwise iz tri invarijante jedne 3×3 matrice, bez SVD-a i bez host sync-a.** PyTorch na CUDA-i, zaključno sa 2.14.1 i main granom, prihvata samo `driver='gels'` (QR bez pivotiranja, pretpostavlja pun rang). Zahtev za `gelsd` na CUDA-i otvoren je od 2022, a jedini PR koji ga dodaje je zastao. cuSOLVER i cuBLAS nemaju batched rank-revealing least squares, pa je jedini gotov GPU put `torch.linalg.pinv`/`svd` preko `gesvdjBatched`, sa host sync-om po pozivu i bez ijednog javnog A100 merenja za 25×3. U radnom prostoru je već pokazano da `gels` na degenerisanim kutijama greši za 1e15–1e34 i da CPU `gelsd` troši 65–88% batched prolaza na `big`. Merenja iz ovog istraživanja (Mac CPU, ne GPU) pokazuju da je zatvorena forma ~30× brža od batched `gelsd`-a već na CPU-u i tačnija na svim postojećim tipovima kutija. Ista merenja pokazuju i da postojeći tipovi `near_degenerate` i `mixed_scale` po konstrukciji padaju na poređenju sa `gelsd`-om na 1e-12, jer tu greši `gelsd`, a ne zamena. Ispravnost (soundness) u realnoj aritmetici ne zavisi od izbora ravni, pod četiri uslova koje treba eksplicitno testirati. Preporuka: dizajn A kao primarni, sa ugovorom zasnovanim na toleranciji konačne granice i na tačnom racionalnom oraklu umesto na `gelsd`-u, i sa analitičkim rcond pravilom kao „gelsd-compat" režimom koji je obavezan u float32. C (`pinv`) služi samo kao tačka poređenja, B dolazi u obzir tek ako logovanje ranga na stvarnim podacima nađe kutije preko praga, a D treba odbaciti jer ni uz idealno skaliranje ne prelazi ~6×.

> **Oznake porekla.** **[I]** spoljni izvor (link). **[P]** zapis, odluka ili merenje iz projekta (link na fajl u radnom prostoru). **[M]** merenje urađeno tokom ovog istraživanja na Mac-u (M5 Pro, CPU, torch 2.12.1 iz `Pasado/.venv`, numpy/scipy sa Apple Accelerate LAPACK-om); **nijedno nije na GPU-u**, a rezultati tik uz prag ranga mogu se na MKL/OpenBLAS-u razlikovati u poslednjim bitovima. **[Z]** zaključak izveden iz prethodnog. Skripte za [M] su u privremenom scratchpad folderu `/private/tmp/claude-501/-Users-dimitrije-prakse-UIUC/8452154d-3ed3-4bd4-a692-8368b543e7e0/scratchpad/` (podfolderi `kinds/` i `exp/`, plus `plane_check.py`). Kopirajte ih ako želite da ih sačuvate.
>
> **Ograničenja vlasnika važe kroz ceo izveštaj.** Sekvencijalni put i `precise_transformer.py` ostaju netaknuti kao referenca. Novi solver živi pored `batched_splits.py`, bira se prekidačem i podrazumevano je isključen. Nigde nema AI atribucije: ni u commit porukama, komentarima, docstring-ovima i dokumentaciji, ni u logovima. Komentari i dokumentacija unutar `Pasado` repoa su na engleskom. Agent ne pokreće `git commit`, a CUDA se pokreće na Colab-u preko `%%writefile` ćelija ([P] [PROMPT_GPU_REGRESSION_2026-10-25.md](/Users/dimitrije/prakse/UIUC/notes/PROMPT_GPU_REGRESSION_2026-10-25.md)).

## Svet nudi samo SVD put, a `gelsd` na CUDA-i ne postoji ni u jednom izdanju

Stanje u PyTorch-u je jednoznačno. `get_default_lstsq_driver` na CUDA-i odbija svaki drajver osim `gels` porukom „driver other than gels is not supported on CUDA". Isto važi za tagove v2.12.0, v2.13.0, v2.14.0 i v2.14.1 i za main ([I] [BatchLinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/BatchLinearAlgebra.cpp)). Za 25×3 sa batch-om od bar dva problema CUDA `gels` ide u cuBLAS `gelsBatched` (uslov `m <= 256 && batchCount >= max(2, m/16)`). Pri tome se `rank`, `singular_values` i `rcond` na CUDA-i ignorišu ([I] [cuda/linalg/BatchLinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp)). Ni signal greške nije pouzdan: rang-deficijentan sistem „silently fails in non-batched-mode but throws a `_LinAlgError` in batched mode" ([I] [#117122](https://github.com/pytorch/pytorch/issues/117122)). Autor PR-a #176746 je sa tačnom linearnom zavisnošću kolona dobio pogrešan rezultat bez ikakve greške ([I] [PR #176746](https://github.com/pytorch/pytorch/pull/176746)). Zahtev #72711 za `gelsd` preko `linalg.svd` otvoren je od februara 2022 ([I] [#72711](https://github.com/pytorch/pytorch/issues/72711)), a PR #126652 je zatvoren nespojen 2024 ([I] [PR #126652](https://github.com/pytorch/pytorch/pull/126652)). PR #176746 je tri puta označen kao stale i nije ni u jednom izdanju. Prijava #180157 („GPU gives wrong answer for rank-deficient lstsq") zatvorena je jednom rečju, „documented" ([I] [#180157](https://github.com/pytorch/pytorch/issues/180157)).

**Jedini ugrađeni put do `gelsd` semantike na CUDA-i je SVD.** `torch.linalg.pinv` računa `S_pinv = where(S > tol, 1/S, 0)` sa `tol = max(atol, rtol·σ_max)` i podrazumevanim `rtol = eps·max(m, n)`. To je isto pravilo kao kod `gelsd`-a, uključujući to da se `σ ≤ tol` nulira ([I] [LinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LinearAlgebra.cpp)). `svd` na CUDA-i za m, n ≤ 32 poziva `cusolverDn<t>gesvdjBatched` sa tolerancijom eps i najviše 400 sweep-ova. Alocira pun U (25×25 po problemu) i kopira prve kolone nazad, a posle svakog poziva kopira `infos` na host („A device-host sync will be performed") ([I] [BatchLinearAlgebraLib.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp)). [Z] Za n = 65 536 to je oko 328 MB privremenih U matrica u float64 i jedan sync po sloju. Bez eksplicitnog `driver=` PyTorch nekonvergirane matrice tiho ponovo računa sa `gesvd`, a sa eksplicitnim samo upozorava, pa treba ostaviti `driver=None` ([I] isti fajl). Performansna litica iz #200486 (n = 32 → 33) ne pogađa 25×3, koji ostaje na batched putu ([I] [#200486](https://github.com/pytorch/pytorch/issues/200486)). Otvoreni #75494, međutim, beleži da float32 `pinverse` na CUDA-i daje neoptimalne rezultate za loše kolonski skalirane matrice, a `[1, x, y]` sa |ȳ| ≫ 1 je upravo takva ([I] [#75494](https://github.com/pytorch/pytorch/issues/75494)). Na Mac-u, torch 2.12 nema MPS kernel za `lstsq` ni za `svd`. Nativni MPS SVD stigao je u 2.14.0, samo za float32, i to sa padom i tri tihe greške koje su ispravljene u 2.14.1. Ispravka za odbacivanje malih singularnih vrednosti spojena je tek u `release/2.15` ([I] [PR #198394](https://github.com/pytorch/pytorch/pull/198394), [PR #200536](https://github.com/pytorch/pytorch/pull/200536)).

NVIDIA biblioteke ne nude ništa bolje. cuSOLVER 13.4 od batched gustih rutina ima samo `potrfBatched`/`potrsBatched`, `syevjBatched`, `gesvdjBatched`, `gesvdaStridedBatched` i `XsyevBatched`. Nema batched `gels`, `geqrf` ni `geqp3`, a `gesvda` radi preko AᵀA i NVIDIA ga sama preporučuje „only with well-conditioned data" ([I] [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html)). cuBLAS `gelsBatched` je QR bez pivotiranja, a njegov `devInfoArray` prijavljuje samo tačno nulti dijagonalni element ([I] [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)). [Z] Svaki put preko Gram matrice, uključujući `syevjBatched` na XᵀX i Cardano na formiranom XᵀX, razrešava singularne vrednosti tek do ~√u·σ1 ≈ 1,5e-8·σ1. To je šest redova veličine iznad `gelsd`-ovog praga 5,55e-15·σ1, pa takav put ne može da reprodukuje `gelsd` na skoro-degenerisanim kutijama. MAGMA 2.10 (februar 2026) dodala je `magma_<T>gesvj_batched` i `gesvj_qr_batched` sa fused kernelom za male matrice ([I] [MAGMA ReleaseNotes](https://github.com/icl-utk-edu/magma/blob/master/ReleaseNotes)). Abdelfattah i Fasi mere da je taj solver na 10 000 vrlo malih matrica u FP64 2,2–33,4× brži od cuSOLVER-a na GH200 ([I] [arXiv 2601.17979](https://arxiv.org/html/2601.17979v2)). PyTorch, međutim, MAGMA tretira kao zastareo backend i SVD/lstsq uvek šalje u cuSOLVER ([I] [cuda BatchLinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp)), pa bi MAGMA tražila sopstveni C++/CUDA extension.

Van PyTorch-a, `jax.vmap(jnp.linalg.lstsq)` je SVD-based, spušta se na `gesvdjBatched` i ne radi sync. Nekonvergenciju vraća kao NaN, a od `gelsd`-a se razlikuje samo na tačnoj jednakosti `s == rcond·s0` ([I] [JAX linalg.py](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py)). CuPy batched `pinv(X, rcond=25*eps)` tačno reprodukuje `≤` pravilo, ali mu je podrazumevani rcond 1e-15 ([I] [CuPy _decomposition.py](https://github.com/cupy/cupy/blob/main/cupy/linalg/_decomposition.py)). TensorFlow min-norm put (`fast=False`) postoji samo na CPU-u ([I] [linalg_ops.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_ops.py)). Za sopstveni kernel postoje tri polazišta:

- cuSOLVERDx `Thread()` režim, preporučen za M×N ≤ 100 (25×3 = 75), sa GELS bez pivotiranja i GESVD preko QR algoritma, dostupan iz Pythona kroz nvmath-python ili Warp ([I] [cuSOLVERDx execution ops](https://docs.nvidia.com/cuda/cusolverdx/api/execution_ops.html));
- McAdams-ov 3×3 SVD bez grananja, orijentisan na float32 ([I] [TR1690](https://research.cs.wisc.edu/techreports/2011/TR1690.pdf));
- Kopp-ova poređenja 3×3 eigen-solvera, po kojima analitički put „becomes inaccurate … if the matrix entries differ greatly in magnitude" ([I] [arXiv physics/0610206](https://arxiv.org/abs/physics/0610206)).

**Nijedan javni izvor ne daje A100 FP64 vreme za 10⁴–10⁵ problema veličine 25×3.** Jedini način da se taj broj dobije je vaše Colab merenje.

| Opcija | Rang i min-norm | Batched na GPU-u | Host sync | Ocena za Pasado |
|---|---|---|---|---|
| `lstsq(driver='gels')`, CUDA | nema; pretpostavlja pun rang | da (cuBLAS `gelsBatched`) | ne | isključeno: greške 1e15–1e34 na degenerisanim kutijama [P] |
| `pinv` / `svd` + prag | isto pravilo kao `gelsd` | da (`gesvdjBatched`) | da, po pozivu | dizajn C; brzina na A100 nepoznata |
| PR #176746 (`gelsd` na CUDA) | `gelsd` preko SVD-a | — | — | nije spojen, stale |
| `syevjBatched` na XᵀX, `gesvda` | gubi sve ispod ~1,5e-8·σ1 | da | — | ne reprodukuje prag |
| MAGMA 2.10 `gesvj(_qr)_batched` | SVD, prag ručno | da, fused | zavisi od wrapper-a | traži C++ extension |
| JAX `vmap(lstsq)` | kao `gelsd` (osim jednakosti na pragu) | da | ne | drugi framework |
| CuPy `pinv(rcond=25·eps)` | tačno `≤` pravilo | da | nije provereno | drugi framework |
| TF `lstsq(fast=False)` | min-norm | samo CPU | — | isključeno |
| cuSOLVERDx / sopstveni QR + 3×3 Jacobi | po izboru | da, thread po problemu | ne | najviše posla |
| Zatvorena forma (A) | tačne min-norm grane, rcond pravilo opciono | elementwise | ne | **preporuka** |

## Radni prostor je već odbacio `gels`, ubrzao `gelsd` i napisao prototip

Istorija je dosledna. Upstream kod rešava jedan `gelsd` po neuronu: 96 000 poziva na `3layer` i 1 966 080 na `big`, sa matricama koje se nikad ne ponavljaju i cenom od 37–61 µs po pozivu, gotovo sve overhead ([P] [LSTSQ_ANALYSIS.md](/Users/dimitrije/prakse/UIUC/Pasado/results/optimization_section_5_4/LSTSQ_ANALYSIS.md)). Faza 6 je to spojila u jedan batched `gelsd` po sloju. Rezultat je bio bit-identičan na 26 konfiguracija, a `3layer` je pao sa 28,01 s na 19,36 s ([P] [BATCHED_LSTSQ_RESULTS.md](/Users/dimitrije/prakse/UIUC/Pasado/results/optimization_section_5_4/BATCHED_LSTSQ_RESULTS.md)). Ista analiza je eksplicitno odbacila alternativu: „Closed-form / normal equations / pinv: NO — unnecessary (batched gelsd already native and bit-exact) and numerically riskier (squares the condition number)" ([P] [LSTSQ_ANALYSIS.md](/Users/dimitrije/prakse/UIUC/Pasado/results/optimization_section_5_4/LSTSQ_ANALYSIS.md)). [Z] Oba razloga su danas zastarela. `gelsd` više nije 1–5% vremena nego 65–88% batched `big` prolaza. Prototip ne formira 3×3 Gram matricu, nego dijagonalnost dobija iz ortogonalnosti mreže, bez kvadriranja uslovnog broja. Kada se zatvorena forma integriše, u `LSTSQ_ANALYSIS.md` i `NEXT_RESEARCH_DIRECTIONS.md` treba dodati kratku napomenu na engleskom da je taj negativni rezultat revidiran.

GPU istorija je bila još jasnija. Na A100 (faza 12) `gels` je na point kutijama dao medijanu greške 2,6e31 i maksimum 3,4e34, a na kutijama sa degenerisanim x medijanu 1,3e15. Na kutijama punog ranga medijana je bila 4,4e-16, pa je zaključak glasio „`gels` on GPU must never run on a degenerate box unchecked" ([P] [GPU_READINESS.md](/Users/dimitrije/prakse/UIUC/Pasado/results/optimization_section_5_4/GPU_READINESS.md)). Saša je u avgustu pitao Jacoba da li je `gelsd` neophodan i pomenuo port na CUDA ili MAGMA, uz napomenu da „cuSolver nema `gelsd`" ([P] [misailo-feedback-2026-08-22.md](/Users/dimitrije/prakse/UIUC/claude-memory/misailo-feedback-2026-08-22.md)). Profil od 22.8. dao je `gelsd`-u samo 1,7% celog `big` runa, pa je tema odložena ([P] [gelsd-profiling-result-2026-08-22.md](/Users/dimitrije/prakse/UIUC/claude-memory/gelsd-profiling-result-2026-08-22.md)). Pravilo zapisano 10.10. glasi: „iznad 15%: prototip GPU SVD/minimum-norm solvera; sopstveni kernel tek ako postojeće CUDA SVD/cuSOLVER opcije ne daju potrebnu semantiku ili brzinu" ([P] [HANDOFF_GPU_HYBRID_2026-10-10.md](/Users/dimitrije/prakse/UIUC/notes/HANDOFF_GPU_HYBRID_2026-10-10.md)). Hybrid merenje od 16.10. pokazalo je da CPU fallback čini 54,5% precise forward-a na `big` ([P] [gpu_hybrid_fallback README](/Users/dimitrije/prakse/UIUC/Pasado/results/gpu_hybrid_fallback_2026-10-16/README.md)). Batched splits su 24.10. podigli udeo `gelsd`-a na 65% (S=1), 84% (S=8), 86% (S=32) i 88% (S=64), uz ~34 ms po splitu bez obzira na S. Bez regresije bi preostalih 287 ms pri S=64 dalo plafon od ~14× ([P] [batched splits README](/Users/dimitrije/prakse/UIUC/Pasado/results/gpu_hybrid_batched_splits_2026-10-24/README.md)). Isti README beleži da prvi sigmoid sloj ima `ly == uy` za svaki neuron. **Min-norm grana je zato glavni put, a ne ivični slučaj.**

Prototip od 25.10. postoji samo kao listing u privatnoj belešci. Na realnim podacima (CPU, float64, `3layer` i `big`) pri eps_idx 15 imao je grešku prema tačnom racionalnom rešenju medijane 7e-16 i maksimuma 9e-15, naspram `gelsd`-ovih 2–3e-13 i 7,6e-12. Uz to je zadržao razliku per-split granica ≤ 1,2e-13 pri S=4. Netestirani su ostali adversarial tipovi kutija, float32, CUDA, veće S, `4layer`/`5layer` i GPU timing ([P] [PROMPT_GPU_REGRESSION_2026-10-25.md](/Users/dimitrije/prakse/UIUC/notes/PROMPT_GPU_REGRESSION_2026-10-25.md)).

Postojeća test infrastruktura ima četiri sloja:

- `test_batched_lin_reg.py` traži bitwise jednakost sa per-neuron `gelsd`-om, plus `assert_close` od 1e-12/1e-14 (f64) i 1e-5/1e-6 (f32), i završava sa exit kodom 1 ako bilo šta nije bitwise ([P] [test_batched_lin_reg.py](/Users/dimitrije/prakse/UIUC/Pasado/experiments/test_batched_lin_reg.py)).
- `test_batched_splits.py` traži rtol 1e-10 po splitu prema seeded sekvencijalnoj petlji i proverava soundness `lc ≥ sampled max`. Pri tome forsira svih pet prekidača, uključujući `USE_REAL_CUBIC` ([P] [test_batched_splits.py](/Users/dimitrije/prakse/UIUC/Pasado/experiments/test_batched_splits.py)).
- `test_soundness.py` i `check_sweep_soundness.py` proveravaju `compute_max_error` naspram brute force pretrage i `lc_precise` naspram autograd maksimuma.
- `mutation_check.py` nosi mutante M1–M14 i preko ključa `file` podržava mutante u drugim fajlovima. Iz okruženja briše sve `PASADO_*` promenljive ([P] [mutation_check.py](/Users/dimitrije/prakse/UIUC/Pasado/experiments/mutation_check.py)).

[Z] Kada bi novi solver ušao u `lin_reg_tensor_batched`, `test_batched_lin_reg.py` i `test_batched_grid.py` (koji za `vec=False` traži toleranciju 0,0) pali bi po konstrukciji. Mutant M10 (`gelsy`, 4,4e-12) ih već obara ([P] [mutation_check_2026-10-03.txt](/Users/dimitrije/prakse/UIUC/Pasado/logs/mutation_check_2026-10-03.txt)). Dispatch zato pripada `batched_splits.precise_sigmoid`, čija linija 128 sada bezuslovno zove `pt.lin_reg_tensor_batched`. Ćelija 21 u `COLAB_GPU_HYBRID_BATCHED_SPLITS.ipynb` meri udeo regresije monkeypatch-ovanjem baš te funkcije. Sa novim solverom pokazala bi ~0 i zavarala bi, pa mora da meri novu funkciju. Postoje i četiri rupe:

- `pinv` i `svd` nisu nikad pokrenuti nigde u repou;
- „7 planted mutants" za batched splits nisu u `MUTANTS` i ne mogu se reprodukovati;
- GPU model za run od 24.10. nije zabeležen;
- prototip ne postoji kao `.py` fajl.

## Kartezijanska mreža svodi problem na tačan 3×3

### `X = Q·(5K)` važi za svaku product mrežu

[Z, numerički provereno [M]] Napišimo x = x̄·1 + dx i y = ȳ·1 + dy. Na Kartezijanskoj mreži kolone 1, dx⊗1 i 1⊗dy su međusobno ortogonalne, jer je (dx⊗1)ᵀ(1⊗dy) = (Σdx_i)(Σdy_j) = 0. Za to je dovoljno da su sume centriranih vrednosti nula, a ravnomeran razmak nije potreban, pa zaokruživanje `torch.linspace` ne kvari dekuplovanje. Sledi X = Q·(5K), gde Q ima ortonormirane kolone, a K = [[1, x̄, ȳ], [0, σx, 0], [0, 0, σy]], sa σ² = (1/5)Σ_i (x_i − x̄)². Zato je sing(X) = 5·sing(K), tačno. To je standardni rezultat faktorskih dizajna: kada je XᵀX dijagonalna, ocene koeficijenata su nekorelisane i računaju se nezavisno ([I] [Berkeley handout](https://www.stat.berkeley.edu/~cheng/Handout13.pdf)). **Zatvorena forma je tačno LS rešenje, a ne aproksimacija.** Dve napomene ograničavaju tu tvrdnju. Prvo, centriranje čuva LS rešenja, ali ne i izbor minimalne norme u originalnim koeficijentima (C, A, B). Zato min-norm grane moraju da se računaju u originalnim koordinatama, kao što prototip i radi. Drugo, eventualni prelaz na kvadratni surogat izgubio bi dekuplovanje, jer x² nije ortogonalno na 1, dok bi bilinearni član xy i dalje dekuplovao [Z].

### Pivot centriranje drži grešku na nivou podataka

Chan, Golub i LeVeque pokazuju da pomeranje podataka za bilo koju tačku podataka daje uslovni broj κ̃² ≤ 1 + N. Pomeranje za izračunatu srednju vrednost daje samo κ̃² ≤ 1 + N²κ²u², a izračunata srednja vrednost može čak da padne van [min x, max x] ([I] [Chan, Golub, LeVeque 1983](https://www.math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf)). [Z] Za pivot na krajnjoj tački mreže od pet ravnomernih tačaka dobija se κ̃ = √3, nezavisno od odnosa |x̄|/w. Za usku kutiju koja nije oko nule Sterbenzova lema čini razliku x_i − x_0 tačnom ([I] [Sterbenz lemma](https://en.wikipedia.org/wiki/Sterbenz_lemma)). Curenje kod centriranja na fl(mean) iznosi ≈ (x̄ − c)·z̄/σx². Za x̄ = 2, z̄ = 0,3 i w = 1e-4 to je ~2,6e-7, istog reda kao opaženih 8e-8 iz R2. Prototip već radi sva tri preporučena koraka: pivot `x - x[:, :1]`, oduzimanje srednje vrednosti malih razlika i centrirano z u brojiocu. Taj poslednji korak je „corrected sums of products" ideja, pa pogrešan pivot ulazi tek u drugom redu. [Z] Ostaju dva sitna poboljšanja. Za degenerisanu osu treba uzeti x0 = `x[:, 0]` (tačno), a ne fl(mean(x)), jer srednja vrednost 25 jednakih brojeva nije tačno taj broj (R2). Inače veza A = x0·C prestaje da bude tačna, što test reda R18 (dole) hvata. U k = z̄ − A·x̄ − B·ȳ treba koristiti x̄ = x0 + mean(d) umesto fl(mean(x)).

### `gelsd` ima pojas smeća iznad praga i skok ispod njega

Za 25×3, `dgelsd` prvo radi QR (jer je MNTHR = 4 ≤ 25), zatim bidijagonalizuje 3×3 R, a onda zove `dlalsd` sa N = 3 ≤ SMLSIZ = 25, što znači `dlasdq`, a ne divide-and-conquer. Rang odlučuje upoređivanje *izračunatog* s_i ≤ RCND·s_1, gde je poređenje ne-striktno. RCND je jednak RCOND samo kada je 0 < RCOND < 1, a inače je 2⁻⁵³. Matrica se skalira samo jednim globalnim skalarom, bez ekvilibracije kolona ([I] [dgelsd.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgelsd.f), [dlalsd.f](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dlalsd.f)). Torch podrazumeva rcond = `finfo.eps·max(m, n)`, što je 5,55e-15 u f64 i 2,98e-6 u f32, i upozorava da se taj default može promeniti ([I] [torch.linalg.lstsq](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.lstsq.html)). [Z] Referentni kod zato treba da prosledi rcond eksplicitno. **Odluka o rangu kod `gelsd`-a zavisi od |x̄| i |ȳ|, a ne samo od širine kutije.**

Osetljivost LS rešenja ima član κ²·tanθ ([I] [Grcar, arXiv 1003.2733](https://arxiv.org/pdf/1003.2733)), a `gelsd` radi na necentriranom dizajnu, pa ga plaća pun. Merenje sa sweep-om širine w_y u y-skoro-degenerisanoj kutiji, kao relativna greška prema tačnom racionalnom rešenju istih zaokruženih podataka, dalo je sledeće ([M]):

| w_y | `gelsd` | `gelss` | `gelsy` | pivot-centrirana zatvorena forma |
|---|---|---|---|---|
| 1e-12 | 7e6 | 7e6 | 4e6 | 1,2e-5 |
| 1e-10 | 1,5e3 | | | 8e-8 |
| 1e-8 | 5,8e-2 | | | 9,5e-10 |
| 1e-6 | 1,3e-5 | | | 1,2e-11 |
| 1e-4 | 3,4e-10 | | | 1,2e-13 |
| 1e-1 | 5,7e-15 | | | 1,2e-16 |

Greška `gelsd`-a raste kao 1/w², a greška zatvorene forme kao 1/w, što je nepopravljivi pod koji nameće zaokruživanje z. Na samom pragu se rešenje skokovito menja za ceo v3-deo rešenja punog ranga, (u3ᵀz/s3)·v3. Pseudoinverz je neprekidan samo ako se rang ne menja ([I] [arXiv 2306.12177](https://arxiv.org/pdf/2306.12177)). Predviđeni skok (−0,399; 0; 0,133) poklopio se sa izmerenim (−0,386…−0,400; 0; 0,129…0,133) ([M]). Za c = 30 i w = 1e-12 `gelsd` vraća (0,00111; 0,0333; 0,3), a rešenje punog ranga je (−6,68e9; 2,23e8; 0,3) ([M]). Skok, međutim, zavisi od podataka. Na `mixed_scale` kutijama koje je `gelsd` odsekao, obična zatvorena forma odstupa od njega samo 1,3e-2 ([M]), jer z tamo jedva zavisi od x (x̄ ≈ 0, σ''(0) = 0).

Na postojećim tipovima kutija iz `realistic_grids` (200 kutija po tipu, float64, kao `degenerate_y` uzeta transpozicija `degenerate_x`) izmereno je sledeće ([M], `kinds/kinds_check.py`):

| tip kutije | rang po `gelsd`-u | `gelsd` vs tačno (med / max) | zatvorena forma vs tačno (med / max) | max \|CF − `gelsd`\| rel. |
|---|---|---|---|---|
| `normal` | 3 (200) | 8,9e-16 / 7,2e-14 | 1,6e-16 / 2,1e-15 | 7,1e-14 |
| `degenerate_x` | 2 (200) | 4,9e-16 / 3,0e-14 | 1,4e-16 / 1,6e-15 | 3,1e-14 |
| `degenerate_y` (nedostaje u testu) | 2 (200) | 4,9e-16 / 2,7e-14 | 1,3e-16 / 5,0e-16 | 2,8e-14 |
| `point` | 1 (200) | 3,5e-16 / 1,2e-15 | 1,3e-16 / 4,6e-16 | 1,2e-15 |
| `near_degenerate` | 3 (200) | **2,4e-4 / 2,9e-3** | 1,3e-16 / 1,4e-15 | **2,9e-3** |
| `saturated` | 3 (200) | 2,7e-15 / 3,3e-13 | 1,2e-16 / 7,6e-16 | 3,3e-13 |
| `mixed_scale` | 2 (92), 3 (108) | 1,4e-3 / 2,3e-2 | 3,7e-4 / 3,7e-3 | 2,3e-2 |

Kolona „tačno" je min-norm rešenje tačne pseudoinverzije zaokruženih podataka, sa tačnim rangom (`lx == ux`). Kod `mixed_scale` uslovni broj je ogroman, pa obe metode imaju forward grešku, a zatvorena forma je ~6× bolja. Na 65 536 nasumičnih kutija sa širinama ≥ 1e-3 razlika CF−`gelsd` dostiže 5,0e-10. Na 20 najgorih kutija `gelsd` je od tačnog rešenja udaljen 5,0e-10, a zatvorena forma 1,1e-13, **pa razlika potiče od `gelsd`-a** ([M], `kinds/timing.py`). U float32 slika je drugačija ([M], `kinds/kinds_check_f32.py`). `near_degenerate` se svodi na point kutije, jer je `lx + 1e-12 == lx` u f32. `saturated` daje z ≡ 0, jer σ(x) u f32 postaje 1 za x ≳ 17, što objašnjava `max_rel=nan` u audit logu ([P] [batched_lstsq.log](/Users/dimitrije/prakse/UIUC/notes/audit_2026-09-19/batched_lstsq.log)). Na `mixed_scale` `gelsd` odseca rang na 1 u 179 kutija i na 2 u 21, a **zatvorena forma bez pravila ranga daje koeficijente do 1,86e5, naspram 0,25 kod `gelsd`-a, i forward grešku do 7,45e5**. To je tačno κ·u_f32.

### Prag se reprodukuje elementwise, bez SVD-a

Prag zavisi od položaja kutije. [Z, provereno [M]] `gelsd` tretira y-osu kao degenerisanu kada je w_y ≲ 2√2·rcond·√(1+ȳ²)·√(1 + x̄² + ȳ² + σx²), a za x-osu važi simetričan izraz. Granica dakle raste sa ~(1+x̄²) i sa |ȳ|. U f64 iznosi ≈ 1,6e-12 pri x̄ = 10 i ≈ 1,6e-8 pri x̄ = 1e3. U f32 iznosi ≈ 8,4e-4 pri x̄ = 10 i ≈ 8,3e-2 pri x̄ = 100. Pravilo „rank-deficient iff s3(5K) ≤ rcond·s1(5K)" pogodilo je `gelsd`-ov prelaz do rezolucije mreže (~2,5%) u svih 8 sweep-ova, u f64 i f32 ([M]). [Z] Prag oblika `width / (1 + |mean|) < tau` iz R3 ima pogrešno skaliranje (linearno umesto kvadratno u |x̄|) i ignoriše ȳ. Kod `mixed_scale` upravo |ȳ| ~ 1e6 gura x-granicu na ~1e-8.

Invarijante λ_i = s_i² matrice (5K)ᵀ(5K) su sume nenegativnih članova, pa se računaju sa visokom relativnom tačnošću [Z]:

- e1 = 25(1 + x̄² + ȳ² + σx² + σy²)
- e2 = 625[σx²(1+ȳ²) + σy²(1+x̄²) + σx²σy²]
- e3 = 15625·σx²σy²

Cardano na formiranom XᵀX razrešava λ3 tek do ~u·λ1, odnosno s3 do ~√u·s1, pa ne može da razreši prag. Analitičke 3×3 formule su nestabilne za ponovljene sopstvene vrednosti ([I] [Habera & Zilian, arXiv 2511.00292](https://arxiv.org/abs/2511.00292)). Put koji je radio ([M], `kinds/analytic_rank.py`):

1. Najveći koren λ1 dobija se sa 8 Newton koraka od λ = e1. Kubni polinom je konveksan desno od najvećeg korena, pa konvergencija ide monotono.
2. Iz njega slede q = e3/λ1 i p = (e2 − q)/λ1, a zatim λ2 = (p + √(p² − 4q))/2 i λ3 = q/λ2.
3. Rang je 1 + [λ2 > rcond²·λ1] + [λ3 > rcond²·λ1].

Ovaj elementwise test poklopio je `gelsd`-ov rang na **1 400/1 400 kutija svih sedam tipova, i u f64 i u f32**. Na 4 sweep-a × 2 000 širina oko praga imao je 0 promašaja u f64 i 1 u f32, i to tačno na graničnoj širini 1,03e-5. Za poređenje, `torch.linalg.svdvals` na 5K imao je 2 promašaja. Rešenje za odsečene kutije ne mora da ide preko vektora: dovoljno je da se osa označena K-pravilom tretira kao degenerisana i da se primeni tačna min-norm formula sa x0 := x̄. Taj „compat" put dao je **4,8e-9 prema `gelsd`-u na 92 odsečene `mixed_scale` kutije** u f64, naspram 1,3e-2 bez pravila. U f32 je zadržao koeficijente ograničene (max 0,25 umesto 1,86e5) ([M], `kinds/compat.py`). [Z] Odsečeno rešenje ne treba računati kao θ_full − (v3ᵀθ_full)·v3, jer θ_full u tom režimu ima red 1e9 i oduzimanje bi pojelo ~9 cifara. Ostaje nesvodivi pojas od ~1–2% širine, u kome izračunato s3 ima grešku O(u·s1), pa ga nijedno spoljno pravilo ne može bit po bit da prati. Vrlo je verovatno da se i sam `gelsd` tu razlikuje između Accelerate, MKL i OpenBLAS biblioteka.

## Ispravnost ne zavisi od ravni, pod četiri uslova

Rad o Pasadu bira ravan „linear regression at uniformly spaced points" samo kao heuristiku preciznosti. Veličina 5×5 postoji samo u kodu, a soundness dolazi od posebno izračunatog maksimalnog odstupanja D. Teorema 4.1 to dokazuje preko Hesijana i traži A ≠ 0. U radu se δ < 1e-9 dodaje samo kada je A = 0, dok kod uvek dodaje N(0, 1e-4) ([I] [Pasado OOPSLA'23](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)). [Z] Postoji i jednostavniji argument, koji ne traži A ≠ 0:

1. Za fiksno x, E(x, y) = σ'(x)·y − Ax − By − C je afino u y, pa je |E| konveksno u y i maksimum se postiže na y-ivicama.
2. Na ivici y = e funkcija φ_e(x) je glatka, pa je max |φ_e| na krajevima ili u korenu jednačine e·σ''(x) = A.
3. Za e = 0, φ_e je afino i dovoljni su krajevi.
4. Za e ≠ 0 koreni rešavaju 2s³ − 3s² + s − A/e = 0. U (0,1) ih ima najviše dva, a nema ih nijednog kada je |A/e| > 1/(6√3).

Nijedan korak ne koristi osobine (A, B, C), pa je svaka ravan sound. Isti obrazac koristi Prover za σ(x)·y ([I] [Ryou et al.](https://arxiv.org/abs/2005.13300)). Isti princip važi i u afinoj aritmetici, gde je svaka aproksimacija sa ograničenom greškom Δ sound, a izbor (p, q) utiče samo na čvrstinu ([I] [Rump & Kashiwagi](https://www.tuhh.de/ti3/paper/rump/RuKas14.pdf)). LinSyn takođe deli posao na LP ravan i sertifikovani pomak ([I] [LinSyn](https://arxiv.org/abs/2201.13351)). Provera u realnoj aritmetici na 300 kutija × 5 pravila ravni pokazala je da uglovi plus tačni koreni kubne jednačine nikad ne daju manji D od guste pretrage 41×4001 ([M]).

**Odgovor na Q1 je „da, u realnoj aritmetici", ali argument važi samo ako su ispunjena četiri uslova.**

**Prvi uslov: pronađene su sve stacionarne tačke na obe y-ivice.** Upstream put rešava kubnu jednačinu Cardanom u `complex64` sa eksponentom 0,33333. U NumPy emulaciji tog puta stacionarne tačke odstupaju do 9e-6. D je zbog toga potcenjen do 1,1e-8 relativno za LS ravni, ali do 3,7% za ravni koje nisu LS ([M], emulacija, ne pravi torch kod). [Z] Svi testovi novog solvera treba da rade sa `USE_REAL_CUBIC = True`, kao što `test_batched_splits.py` već radi. Zatvorena forma daje LS ravan, pa ostaje u režimu 1e-8.

**Drugi uslov: maksimizuje se |E|, a ne E.** To je ispravka od 22.10. Posle nje, svih 264 redova sweep-a je sound, sa najmanjim odnosom 1,000041. Pre ispravke ispod je bilo 13 CUDA i 6 CPU redova ([P] [soundness_fix README](/Users/dimitrije/prakse/UIUC/Pasado/results/soundness_fix_2026-10-22/README.md)). Dakle, da: upravo |E| ispravka čini graničnu proveru korektnom za proizvoljnu ravan.

**Treći uslov: D se računa iz tačno onih (A, B, C) koji se upisuju u zonotop,** posle šuma i bez ikakvog dtype cast-a između. Redosled „solve → šum → assert → D" mora ostati isti.

**Četvrti uslov: zaokruživanje se eksplicitno odriče ili pokriva.** Pasado i Phocus izričito kažu da nisu floating-point sound ([I] [Phocus OOPSLA'25](https://jsl1994.github.io/papers/OOPSLA2025.pdf)). Jia i Rinard pokazuju da se zanemareno zaokruživanje može sistematski iskoristiti ([I] [Jia & Rinard](https://arxiv.org/abs/2003.03021)). DeepJ-jev FP-sound režim košta 2,8–4,1× vremena i menja rezultate za manje od 1e-10 ([I] [DeepJ POPL'22](https://jsl1994.github.io/papers/POPL2022Final.pdf)).

Postoji i jedna zamka u implementaciji. Assert `((As > 0) | (As < 0)).all()` u `batched_splits.py` odbija NaN u A, jer je svako poređenje sa NaN netačno. Zato bi underflow σx² (videti Q4) srušio run glasno. NaN u B ili C, međutim, prolazi tiho, pa posle solvera treba dodati eksplicitnu `isfinite` proveru za konačne ulaze [Z]. U float32 bez pravila ranga, koeficijenti reda 1e5 koji se međusobno poništavaju ostaju sound u realnoj aritmetici. Ipak, oni razvlače granicu i pojačavaju grešku zaokruživanja u `A·x̂ + B·ŷ + C` [Z]. To je drugi razlog da je K-pravilo obavezno u f32.

Usput, za razgovor sa mentorom, a ne za ovu izmenu: najveći izbegljivi gubitak preciznosti nije solver, već šum N(0, 1e-4) na A. Proksi poluprečnika W pod tim šumom ima medijanu 1,006× i 90. percentil 1,88× u odnosu na čistu LS ravan. Minimax ravan bi dobila najviše ~3% (medijana W 0,999×, 10. percentil 0,972×) ([M], `plane_check.py`). [Z] Za čvrstinu su razlike solvera reda 1e-13 devet redova veličine ispod efekta šuma, pa bit-paritet sa `gelsd`-om ima vrednost samo za test reproduktivnosti, a ne za kvalitet granice. Šum u ovom zadatku ne sme da se menja, jer seeded stream mora ostati identičan.

## Četiri pitanja recenzenta, rizici R1–R10 i dizajni A–D

**Q1 (soundness iz R1)** je odgovoreno u prethodnom odeljku. Argument je tačan u realnoj aritmetici, ne traži A ≠ 0 i zavisi od četiri navedena uslova, koje katalog testova pokriva.

**Q2 (tolerancija ili reprodukcija `gelsd`-a):** tolerancija je dovoljna, a projekat je to već zapisao. Pravila su „numerički ugovor zapiši pre prvog pokretanja … prag se ne olabavljuje naknadno" i „ne tvrditi univerzalnu bit-identičnost" ([P] [TASKOVI_HIBRID.md](/Users/dimitrije/prakse/UIUC/TASKOVI_HIBRID.md)). Postojeći pragovi su 1e-10 po splitu i 1e-9/1e-5 za CPU naspram CUDA-e. [Z] Bit-paritet nije dostižan ni za C: `pinv` je i na CPU-u 1,0e-12 od `gelsd`-a ([M]), Jacobi i bidijagonalni QR drugačije zaokružuju, a u pojasu od ~2% nijedno pravilo ne prati izračunato s3. „Reprodukovati `gelsd`" tik iznad praga znači reprodukovati njegovu grešku od 1e-2 do 1e6. Jedini realan rizik za A je e2e prag od 1e-10, i to samo ako stvarni podaci sadrže kutije koje `gelsd` odseca iako nisu tačno degenerisane. To se meri jeftino (Korak 0, dole).

**Q3 (čistije rcond pravilo):** da. Pravilo se dobija iz invarijanti e1, e2, e3 Newtonom, elementwise i bez sync-a, i poklopilo se sa `gelsd`-ovim rangom na 1 400 kutija i 16 000 sweep tačaka uz jedan promašaj na samoj granici u f32. Za odsečene ose dovoljna je tačna degenerisana formula sa x0 := x̄, koja se slaže sa `gelsd`-om na 4,8e-9.

**Q4 (numerički hazardi unifikovane formule):** gubitak jedinice u 1 + x0² je bezopasan. Relativna greška ostaje ≤ u, a test sa x0 = 1e8 dao je tačno C = 1e-16 i A = 1e-8 ([M]). Stvarni hazardi su ovi:

1. **Tiha nula pri prelivanju x0².** Za |x0| > 1,34e154 (f64), odnosno > 1,84e19 (f32), x0² se preliva i formula tiho vraća C = A = 0. Sa Pasado ciljevima to je bezopasno, jer je tada σ'(x) = 0, pa je z ≡ 0 i k = 0. Izmereno je da i CF i `gelsd` vraćaju nulu za x0 = 1e155 (f64) i 1e20 (f32) ([M], `kinds/hazards.py`).
2. **NaN iz k·y0.** Za |y0| > ~1e154, k·y0 može da se prelije u inf/inf = NaN [Z]. Lek je skaliranje sa m = max(1, |x0|, |y0|) pre kvadriranja.
3. **Underflow σx² za sitne, ali ne-nulte širine.** Tada je `lx != ux`, a sxx = 0. Izmereno je da prototip za x ≈ 1e-30, w ≈ 1e-37 (f32) i x ≈ 1e-200, w ≈ 1e-210 (f64) vraća **[nan, nan, 0,25]**, dok `gelsd` vraća konačno rešenje ranga 2 ([M]). Lek je predikat degeneracije `(lx == ux) | (sxx == 0)` ili K-pravilo.
4. **NaN iz neizabrane `where` grane.** U forward prolazu `where` vraća tačnu vrednost, ali aritmetičko mešanje maskom daje NaN, a autograd kroz `where` daje NaN gradijente ([I] [PyTorch forum](https://discuss.pytorch.org/t/incorrect-gradient-calculation-with-torch-where-and-nans/185367)).
5. **FTZ.** Pod flush-to-zero se subnormalni rezultati (npr. C ≈ 1e-310) gube, a ni Sterbenz više ne važi.
6. **Poništavanje u C = z̄ − A·x̄ − B·ȳ.** Ono je inherentno i `gelsd` ga ima jednako, pa testovi treba da porede i fitovane vrednosti u uglovima kutije, ne samo sirovo C.
7. **Gubitak tačnosti u ciljevima.** 1 − σ(x) gubi relativnu tačnost za x ≳ 37 (f64), odnosno 17 (f32), ali to je zajedničko za oba solvera.

−0,0 i +0,0 se porede kao jednaki, pa idu u degenerisanu granu, kao i kod `gelsd`-a. NaN u jednoj kutiji ostaje u svom redu i kod `gelsd`-a (torch 2.12.1, CPU) i kod zatvorene forme ([M]).

| Rizik | Šta su nalazi pokazali | Šta uraditi |
|---|---|---|
| R1 soundness | tačno u realnoj aritmetici, ne traži A ≠ 0; četiri uslova | `USE_REAL_CUBIC`, \|E\|, D iz emitovanih koeficijenata, `isfinite`, autograd provera |
| R2 centriranje | CGL teorija potvrđuje pivot; prototip je ispravan | x0 = `x[:,0]` za degenerisanu osu, x̄ = x0 + mean(d); mutanti S4, S5, S16 |
| R3 prag | prag ~(1+x̄²)·√(1+ȳ²); `mixed_scale` je već preko praga u 46% kutija | K-pravilo elementwise; Korak 0 meri učestalost na stvarnim podacima |
| R4 float32 | prag 8e-4 pri x̄ = 10; CF bez pravila daje koeficijente 1,86e5 | K-pravilo obavezno u f32; f32 se poredi sa f64 tačnim oraklom |
| R5 A ≠ 0 | A = 0 je moguće kao i kod `gelsd`-a; NaN u A ruši assert, NaN u B/C prolazi | zadržati redosled i broj izvlačenja šuma; `isfinite`; mutant S15 |
| R6 `where` | prototip štiti tačne grane, ali ne i underflow sxx | `sxx == 0` u predikatu; mutant S14 |
| R7 ne-konačno | NaN ostaje u redu kod oba solvera; `lx > ux` je matematički bezopasan | ugovor po elementu (O5) |
| R8 reproduktivnost | ugovor je već toleranca; bit-paritet je nemoguć u pojasu | ugovor iz O3 |
| R9 SVD | isto pravilo, ali sync, ~328 MB za U i neizmerena brzina; na Mac CPU `pinv` 114 ms, `gelsd` 140 ms, CF 4,7 ms | C samo za poređenje, izmeriti na A100 |
| R10 obim | dispatch u `batched_splits.py`; dva testa pala bi kad bi se menjao `lin_reg_tensor_batched` | novi modul + prekidač, ćelija 21 meri novu funkciju |

| Dizajn | Brzina | Vernost `gelsd`-u | Glavni rizik | Ocena |
|---|---|---|---|---|
| **A** zatvorena forma, tačna degeneracija | elementwise, bez sync-a; na Mac CPU ~30× brža; e2e plafon ~14× | tačna za tačnu degeneraciju, tačnija drugde, razlikuje se u pojasu odsecanja | f32 bez pravila ranga | **primarni** |
| **A+K** A sa analitičkim rangom („compat") | i dalje elementwise, nekoliko desetina flopova više po kutiji | rang se poklapa van pojasa od ~1–2%; odsečeno rešenje ~5e-9 | heuristika izbora ose kada su obe ose blizu praga | opcija za f64, **podrazumevano za f32** |
| **B** A + CPU `gelsd` na podskupu | jedan sync i graph break po sloju | `gelsd` tačno na označenom podskupu | složenost; zastavica ima isti pojas | samo ako Korak 0 to traži |
| **C** `pinv` preko SVD-a | `gesvdjBatched`, sync, veliki U | isto pravilo; rang se može razlikovati u pojasu | neizmerena brzina, #75494 u f32 | tačka poređenja |
| **D** paralelni CPU `gelsd` | [Z] 287 + 2201/6 ≈ 654 ms pri S=64 → 4067/654 ≈ **6,2×** i uz idealnih 6 jezgara | bit-tačan | ne dostiže cilj 10×; D2H/H2D ostaju | odbaciti |

## Odluke koje vlasnik mora da donese

Ovo nisu tehnička pitanja sa jednim tačnim odgovorom. Svaka odluka menja šta testovi znače, pa ih treba zapisati pre prvog pokretanja, u duhu postojećeg pravila o numeričkom ugovoru.

**O1. Specifikacija ranga: tačna degeneracija ili `gelsd`-ov rcond prag.** Postoje tri opcije:

- (a) tačna degeneracija, `lx == ux` ili `sxx == 0`, što daje tačnu pseudoinverziju zaokruženih podataka;
- (b) `gelsd`-ovo pravilo preko analitičkog K;
- (c) sopstveni prag tipa `width/(1+|mean|)`.

Dokazi:

- (a) je matematički tačna i tačnija od `gelsd`-a na svim tipovima kutija.
- (b) se poklapa sa `gelsd`-om van pojasa od ~1–2% i jeftina je.
- (c) nema ispravno skaliranje i ne poklapa se ni sa jednom specifikacijom.
- U f32 (a) daje koeficijente 1,86e5.
- Na stvarnim f64 podacima e2e razlika je ≤ 1,2e-13 pri eps_idx 0/8/15 i S=4, što sugeriše da se odsecanje tamo retko ili nikad ne dešava. To još nije izmereno direktno.

**Preporuka:** u f64 podrazumevano (a), sa implementiranim i testiranim (b) iza parametra; u f32 obavezno (b); (c) odbaciti. Konačan izbor za f64 treba doneti posle Koraka 0. Ako nijedna stvarna kutija nije odsečena mimo tačne degeneracije, (a) je na stvarnim podacima nerazlučiva od `gelsd`-a i pobeđuje jednostavnošću. Ako takvih kutija ima, (b) čuva e2e prag od 1e-10 bez CPU fallback-a.

**O2. Orakl za testove.** Kandidati su CPU `gelsd` u f64; tačni racionalni orakl O1 (`fractions.Fraction` na istim float ulazima, sa tačnim rangom); orakl O2 u visokoj preciznosti, koji primenjuje `gelsd` specifikaciju (mpmath sa dps ≥ 50 i eksplicitnim odsecanjem σ ≤ 25·eps·σ1); i f64 tačno rešenje za f32 runove. Dokazi:

- `gelsd` ima forward grešku 2,9e-3 na `near_degenerate`, 2,3e-2 na `mixed_scale` i 1,1e-2 na kutiji c = 1, w = 1e-12 ([M]).
- `mpmath.pinv` sa podrazumevanim `rtol` odseca na `max(m,n)·eps·σ1²` umesto `σ1`, a `if not rtol` tiho pretvara 0 u default ([I] [mpmath linalg.py](https://github.com/mpmath/mpmath/blob/master/mpmath/matrices/linalg.py)).
- Racionalni min-norm preko faktorizacije X = F·G, X⁺ = Gᵀ(GGᵀ)⁻¹(FᵀF)⁻¹Fᵀ, daje Xᵀr = 0 i (c, −1, 0)·b = 0 tačno ([M]).

**Preporuka:** O1 kao primarni orakl za (a), O2 za (b). `gelsd` služi samo kao dijagnostika i kao referenca u e2e testu, gde je sekvencijalni `gelsd` put referenca po definiciji. f32 se uvek poredi sa f64 tačnim rešenjem, nikad sa f32 `gelsd`-om.

**O3. Ugovor tolerancije.** Predlog ima četiri nivoa.

1. **e2e:** ostaje 1e-10 relativno po splitu prema seeded sekvencijalnom putu u f64, uz soundness `lc ≥ sampled max`.
2. **Koeficijenti prema oraklu:** forward greška ≤ 30·u·(κ + κ²·tanθ), sa κ = σ1/σ3 iz orakla. To je oblik LAPACK-ove ocene greške za least squares ([I] [LAPACK Users' Guide](https://www.netlib.org/lapack/lug/node82.html)).
3. **Backward odnos:** važna ispravka u odnosu na LAPACK-ov `DQRT17` sa α = ‖r‖. Taj odnos je ovde neupotrebljiv, jer je izmereno do 1,7e13 i za *tačno rešenje zaokruženo na float64* na `near_degenerate` ([M], `kinds/r17b.py`; [I] [dqrt17.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/dqrt17.f)). Normalizacija `R17b = ‖Xᵀr‖₁ / (‖X‖₁²·‖b‖₁·25·eps)` ostala je ≤ 0,02 za `gelsd`, CF i zaokruženo tačno rešenje na svim tipovima, u f64 i f32 ([M]). Uz nju ide odnos reda R18 za rang-deficijentne rezultate.
4. **Bitwise** samo tamo gde je to po konstrukciji tačno: skaliranje sa 2^k, negacija, transpozicija, invarijantnost na veličinu batch-a i ponavljanje runa na CUDA-i.

Za f32 predlog je 1e-5 relativno na konačnoj granici, kao u postojećem ugovoru za CPU naspram CUDA-e, plus konačni koeficijenti i provera njihove veličine ili fitovanih vrednosti u uglovima. Ta poslednja provera je potrebna jer R17b ne hvata ogromne koeficijente: CF u f32 na `mixed_scale` imao je R17b = 0,0016 uz koeficijente 1,86e5. **Preporuka:** potvrditi sa Sašom da je tvrdnja u radu „granice se poklapaju sa originalom do 1e-10/1e-9 i sound su", a ne „bit-identično".

**O4. Separabilni oblik ili 25 ciljeva.** Pošto je z = σ'(x)·y separabilno, za pun rang važi B = mean_i σ'(x_i), A = ȳ·slope_x(σ') i C = −A·x̄. To je 5 evaluacija σ' umesto 25, bez šumnog poda zaokruženog z [Z]. Taj oblik je od `np.linalg.lstsq` odstupio najviše 2,1e-11 na 300 kutija ([M]). Ipak, i on traži min-norm grane: za degenerisano y `gelsd` deli intercept između B i C, dok separabilni oblik daje B = h̄. Osim toga, menja ulaze, pa se orakl i e2e referenca razlikuju na nivou eps. **Preporuka:** prvo 25-target oblik. On je drop-in sa istim ulazima kao `gelsd` (`batched_splits.py` linije 125–127 i dalje grade mrežu i ciljeve), istim oraklom i istim e2e testom. Separabilni oblik je kasnija optimizacija iza posebnog prekidača, i to samo ako A100 profil pokaže da mreža i ciljevi postaju bitni kada solver nestane.

**O5. Ugovor za ne-konačne ulaze.** NumPy u tom slučaju diže `LinAlgError`, a PyTorch ne garantuje ništa ([I] [numerical_accuracy.md](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.md)). Izmereno je da i `gelsd` i CF zadržavaju NaN u redu u kome je nastao ([M]). **Preporuka:** NaN se propagira po elementu, bez kontaminacije suseda. Posle solvera ide eksplicitna `isfinite` provera za konačne ulaze, a ugovor se dokumentuje u docstring-u na engleskom.

**O6. Gde i kako se bira solver.** **Preporuka:**

- Novi modul pored `batched_splits.py`. Ime je samo predlog, npr. `forward_mode_tensorized_src/grid_lstsq.py` sa `solve_grid_plane(grid, targets, rank_rule=...)`.
- Dispatch u `batched_splits.precise_sigmoid` prema prekidaču iz okruženja, npr. `PASADO_BATCHED_SOLVER` ∈ {`gelsd` (default), `closed_form`, `closed_form_gelsd_rank`, `pinv`}.
- Redosled „solve → šum → assert" i broj izvlačenja šuma ostaju nepromenjeni.
- Testovi postavljaju prekidač eksplicitno, jer `mutation_check.py` briše `PASADO_*`.
- Pre Colab runa novi fajlovi idu na fork ili u `%%writefile` ćelije, inače Colab tiho pokreće stari kod.
- `precise_transformer.py` se ne dira.

**O7. Uloga B, C i D.** B dolazi u obzir samo ako Korak 0 nađe odsečene kutije i mentor traži paritet. C je izmerena tačka poređenja (f64, `driver=None`) i nezavisna GPU unakrsna provera. D treba odbaciti, jer ni u idealnom slučaju ne dostiže 10×.

## Katalog testova i ivičnih slučajeva, po prioritetu

Prioriteti znače sledeće: **P0** mora biti zeleno pre nego što se prekidač uključi u bilo kom eksperimentu, **P1** pre default-on ili brojeva za rad, a **P2** je dodatno ojačavanje. Predlog fajlova:

- `experiments/grid_lstsq_oracle.py` — O1 i O2 orakli;
- `experiments/test_grid_lstsq.py` — tipovi kutija, MR-ovi, odnosi, ivični slučajevi;
- `experiments/log_gelsd_rank.py` — Korak 0 i snimanje stvarnih kutija u `.pt`;
- proširenje `test_batched_splits.py` argumentom `--solver`;
- novi mutanti u `mutation_check.py`.

| # | Slučaj | Kako generisati | Očekivanje i orakl | Prioritet |
|---|---|---|---|---|
| K1 | `normal` (postoji) | `realistic_grids` | O1, forward ≤ 30u(κ+κ²tanθ); R17b < 30 | P0 |
| K2 | `degenerate_x` za c ∈ {0, 1, 30, 1e6} | postoji; dodati c | tačna formula; A = x0·C tačno (R18); c = 0 sam ne hvata grešku u formuli | P0 |
| K3 | **`degenerate_y` (nedostaje)** | transpozicija K2; to je i stvarni sloj 1 | B = y0·C; MR4 sa K2 | P0 |
| K4 | `point`, uključujući cx, cy ≠ 0 i ishodište | postoji; dodati | b = z̄·(1, x0, y0)/(1+x0²+y0²); u ishodištu b = (z̄, 0, 0) | P0 |
| K5 | `near_degenerate` (w = 1e-12, \|c\| ≤ 4) | postoji | samo O1; `gelsd` greši do 2,9e-3 pa je samo dijagnostika; za f32 dodati w ≈ 1e-5·(1+\|c\|), jer 1e-12 kolapsira u point | P0 |
| K6 | sweep oko praga | širina log od 1 ulp do 1, c ∈ {0; 0,3; 1; 2; 10; 30; 1e3; 1e4}, ȳ ∈ {0, 1, 1e3, 1e6} | (b): rang = `gelsd` van pojasa [rcond/8, 8·rcond] (heuristički faktor); u pojasu prihvata se bilo koji rang uz R17b < 30 i R18 | P0 za (b), P1 za (a) |
| K7 | `mixed_scale` (postoji) | 46% odsečeno u f64 | (a): O1; (b): rang i ~5e-9 prema `gelsd`-u; f32: ograničeni koeficijenti | P0 |
| K8 | `saturated` + saturated × near-degenerate (c = 30, w = 1e-12) | postoji; dodati | `gelsd` rang 2 naspram tačnog ranga 3: test razdvaja (a) od (b) | P0 |
| K9 | krajevi udaljeni 1–3 ulp | `nextafter` | konačno, u skladu sa specifikacijom | P1 |
| K10 | **sitna ne-nulta širina (underflow sxx)** | f32: x ≈ 1e-30, w ≈ 1e-37; f64: x ≈ 1e-200, w ≈ 1e-210 | konačno; prototip trenutno daje NaN | P0 |
| K11 | prelivanje: \|x0\|, \|y0\| ∈ {1e8, 1e154, 1e155, 1e200} f64; {4096, 1e19, 1e20} f32 | sintetičko diadsko z (sigmoid ciljevi bi bili 0) | tačno ili dokumentovana nula, nikad NaN | P1 |
| K12 | −0,0/+0,0; `lx > ux` | ručno | degenerisana grana; ista ravan kao kod preuređene kutije | P1 |
| K13 | NaN/inf u jednom z ili u jednoj granici | ručno | NaN u tom redu, susedi bitwise nepromenjeni | P1 |
| K14 | konzistentni podaci z = p + q·x + r·y | diadski p, q, r i diadska mreža | b = (p, q, r) na nekoliko ulp·κ; R16 | P0 |
| K15 | nulti podaci z ≡ 0 (i f32 `saturated`) | ručno | b = 0 tačno, bez NaN | P0 |
| K16 | stvarne snimljene kutije | 4 mreže, svi slojevi, eps_idx {0, 4, 8, 12, 15}, S ≤ 8 | O1 + statistika ranga + `gelsd` dijagnostika | P0 |
| K17 | veličine batch-a i mešani batch | n ∈ {0, 1, 2, 3, 5, 50, 1024, 65 535, **65 536** (= S·N za `big` pri S=64), 65 537} | rezultat po elementu jednak solve-u pojedinačne kutije | P0 (0, 1, mešani), P1 (veliki) |

Metamorfne relacije ne traže orakl, a ubijaju greške u znaku, rasporedu i granama ([I] [Kanewala](https://mountainscholar.org/handle/10217/166910)). U notaciji (C, A, B) = (intercept, nagib po x, nagib po y) one su ovakve:

| MR | Transformacija | Očekivani odnos | Tačno u FP? |
|---|---|---|---|
| MR1 | z → 2^k·z, k ∈ {−60, −1, 1, 60} | b → 2^k·b | bitwise |
| MR2 | z → −z | b → −b | bitwise |
| MR3 | obrnuti x-osu: z'(i,j) = z(4−i, j) | A' = −A, B' = B, C' = C + A·(lx+ux) | tolerancija |
| MR4 | transpozicija x ↔ y (zameniti granice) | A' = B, B' = A, C' = C | bitwise ako je kod simetričan |
| MR5 | z → z + p + q·x + r·y | pun rang: (C+p, A+q, B+r); degenerisano: projekcija na red-prostor | tolerancija |
| MR6 | z tačno planarno (K14) | b = (p, q, r) | skoro tačno |
| MR7 | pomeriti kutiju x → x + s (isto z) | A, B isti; C → C − A·s; **isključiti blizu praga** | tolerancija |
| MR8 | skalirati x → 2^k·x (isto z) | A → A/2^k; B, C isti; **rang nije invarijantan na skaliranje kolona** | skoro bitwise |
| MR9 | z → X·b (fitovane vrednosti) | b nepromenjeno | tolerancija |
| MR10 | linearnost po z | solve(z1+z2) = solve(z1) + solve(z2) | tolerancija |
| MR11 | ista kutija na poziciji 0 i n−1, batch 1 naspram 65 537, NaN susedi | identičan rezultat po elementu | bitwise za elementwise kod |
| MR12 | CPU f64 naspram CUDA f64; MPS f32 naspram f64 | unutar forward granice, nikad `torch.equal` | ne |

Odnosi u stilu LAPACK-a dolaze iz `DDRVLS`-a, koji koristi THRESH = 30 i četiri odnosa nezavisna od skale ([I] [ddrvls.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/ddrvls.f)). LAPACK-ov generator drži singularne vrednosti ili ≥ 0,1 ili tačno 0 i koristi RCOND ≈ √eps/2, pa **nikad ne testira prag 25·eps**, i tu prazninu projekat mora sam da popuni ([I] [dqrt15.f](https://github.com/Reference-LAPACK/lapack/blob/master/TESTING/LIN/dqrt15.f)). Za vaš problem odnosi su sledeći:

- **R16** `‖z − Xb‖₁ / (25·‖X‖₁·‖b‖₁·eps)`, samo za konzistentne podatke (K14).
- **R17b**, opisan u O3, umesto `DQRT17` sa α = ‖r‖.
- **R18** proverava da je rešenje u red-prostoru: |(x0, −1, 0)·b| / (‖(x0, −1, 0)‖·‖b‖·25·eps) < 30 za degenerisano x, simetrično za y i oba vektora za point. Ovo direktno proverava „minimalnu normu", a ne samo „neko LS rešenje", i nad racionalnim oraklom vrednost je tačno 0 ([M]).
- **R15** poredi singularne vrednosti, samo za C, sa `singular_values` koje vraća `gelsd`.
- **ULP** udaljenost (`numpy.testing.assert_array_max_ulp`) samo za MR1, MR2 i MR4 i za konzistentne podatke.

Od biblioteka vredi preuzeti nekoliko obrazaca:

- NumPy-jev `test_rcond`, jedini nađeni test koji namerno stavlja singularnu vrednost između dva praga ([I] [NumPy test_linalg.py](https://github.com/numpy/numpy/blob/main/numpy/linalg/tests/test_linalg.py));
- SciPy-jev batch mešanog ranga ([I] [SciPy test_basic.py](https://github.com/scipy/scipy/blob/main/scipy/linalg/tests/test_basic.py));
- PyTorch-ov test skaliranja sa 1e-8 i regresioni test za stanje koje curi između elemenata batch-a ([I] [PyTorch test_linalg.py](https://github.com/pytorch/pytorch/blob/main/test/test_linalg.py)).

Njihove tolerancije nisu dobar uzor: PyTorch koristi 1e-5 i za f64, pa bi bag od 1e-6 prošao.

Novi mutanti dobijaju prefiks **S**, da se ne bi sudarili sa postojećim M1–M14 u `mutation_check.py`. Ulaze u `MUTANTS` sa ključem `file` koji pokazuje na novi modul, a novi test ulazi u `TESTS`. Traži se 100% ubijenih. Za preživelog mutanta prvo treba proveriti da li je ekvivalentan na celom domenu, kao što je S3 kada su svi generisani c = 0 ([I] [Wu & Mitchell](https://www.cs.toronto.edu/~rwu/papers/wu2019mutation.pdf)).

| ID | Mutant | Ubija ga | Preživljava na |
|---|---|---|---|
| S1 | obrnut znak A ili B (iz briefa: wrong sign) | MR6, MR3 | z konstantno po x |
| S2 | zamenjen redosled izlaza (C, A, B) | MR6, MR4 | simetrične kutije |
| S3 | pogrešna degenerisana formula, npr. C = mean, A = 0 (iz briefa) | R18, O1 sa c ≠ 0 | c = 0 |
| S4 | degeneracija preko fl(mean), `std == 0` ili relativne širine umesto krajeva (iz briefa) | K2/K3 sa c = 30, K6 | širine daleko od praga |
| S5 | bez pivot centriranja, tj. centriranje na fl(mean) ili sirov dizajn (iz briefa) | O1 na c = 30, w ≤ 1e-8: greška skače sa ~1e-15 na ≥ 1e-5 | dobro uslovljene kutije |
| S6 | raspored mreže (x najbrži; `y[:, 0] == y[:, -1]`) | MR4, asimetričan ručni primer | simetrično z |
| S7 | pogrešan imenilac (24 ili 5 članova) | MR6 | konstantno z |
| S8 | y-grana prepisana iz x-grane | K3 `degenerate_y` | skup samo sa `degenerate_x` |
| S9 | point imenilac 1 + cx² | K4 sa cy ≠ 0 | cy = 0 |
| S10 | interni downcast na f32 ili TF32 matmul | O1 u f64 na 30uκ | labave tolerancije |
| S11 | aritmetičko mešanje umesto `where` | `isfinite` na K2–K4 | pun rang |
| S12 | indeksiranje batch-a | MR11, n = 65 537 | homogeni batch |
| S13 | rcond pomeren (eps umesto 25·eps; σ umesto σ²) | K6 | van pojasa |
| S14 | bez `sxx == 0` zaštite | K10 | sve ostalo |
| S15 | šum pre solve-a ili drugačiji broj izvlačenja | seeded `test_batched_splits.py` | unseeded runovi |
| S16 | degenerisana vrednost iz fl(mean) umesto `x[:, 0]` | tačni R18 = 0 nad oraklom | svaka tolerancija (mutant osetljivosti) |

Uz to ide i `mutmut` 3.x na modulu za pokrivenost operatora, uz `# pragma: no mutate` na dtype i device linijama ([I] [mutmut](https://github.com/boxed/mutmut/blob/main/README.rst)).

GPU-specifični testovi su sledeći [Z, iz izvora]:

- **TF32.** Redukcije treba pisati kao elementwise sume, a ne kao `@`. Test zatim prebacuje `torch.backends.cuda.matmul.fp32_precision` između `"tf32"` i `"ieee"` i traži bitwise isti f32 izlaz ([I] [numerical_accuracy.md](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.md), [common_cuda.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_cuda.py)).
- **Determinizam.** Dva CUDA runa pod `torch.use_deterministic_algorithms(True)` moraju biti bitwise jednaka, kao što `colab_verify_gpu_hybrid.py` već traži.
- **Invarijantnost na veličinu batch-a** (MR11).
- **CPU naspram CUDA-e** unutar forward granice, jer FMA menja zaokruživanje ([I] [NVIDIA floating point](https://docs.nvidia.com/cuda/floating-point/)).
- **MPS f32** se poredi sa f64 tačnim rešenjem.
- **FTZ** uključen i isključen (`torch.set_flush_denormal(True)` na CPU-u).

E2e sloj čine:

- `test_batched_splits.py` sa uključenim prekidačem, na CPU-u i `--device cuda`, u f64 sa rtol 1e-10 i u f32 sa unapred zapisanom tolerancijom;
- varijanta `test_soundness.py` koja ravni gradi novim solverom (sada ih gradi `pt.lin_reg_tensor_batched`);
- `check_sweep_soundness.py` na novim sweep CSV-ovima;
- ćelija 19 notebook-a (< 1e-9) kao e2e provera u CUDA razmeri.

## Preporučeni put: prvo izmeriti rang, pa A, pa A100

**Korak 0 traje nekoliko sati na CPU-u i odlučuje O1.** U skripti za merenje, a ne u `precise_transformer.py`, treba monkeypatch-ovati `pt.lin_reg_tensor_batched`, kao što to već radi `measure_hybrid_fallback.py`. Skripta beleži rang koji `gelsd` vraća i analitički odnos s3/s1 za svaki sloj, na 4 mreže × eps_idx {0, 4, 8, 12, 15} × nekoliko slika × S ∈ {1, 4}, i usput snima kutije za K16. Ishod su dva broja: koliko kutija `gelsd` odseca iako nisu tačno degenerisane i koliko ih je u pojasu. Ako su oba nula na celoj matrici, specifikacija (a) je na stvarnim podacima ekvivalentna `gelsd`-u.

**Korak 1 je implementacija dizajna A u novom modulu, sa komentarima na engleskom.** Modul treba da ima:

- pivot centriranje;
- predikat degeneracije `(lx == ux) | (sxx == 0)`;
- x0 iz `x[:, 0]` i x̄ = x0 + mean(d);
- skaliranje degenerisanih formula sa m = max(1, |x0|, |y0|);
- zaštićene imenioce i redukcije kao elementwise sume bez matmul-a;
- provere dtype-a;
- parametar `rank_rule`, sa analitičkim K-pravilom kao opcijom, podrazumevano `exact` u f64 i `gelsd` u f32.

**Korak 2 su orakli, P0 testovi i mutanti S1–S16 na CPU-u** (f64 i f32, i MPS f32 ako torch to podržava). **Korak 3 je e2e na CPU-u:** `test_batched_splits.py` sa prekidačem, soundness varijanta i mali sweep naspram sekvencijalnog puta. **Korak 4 je Colab A100**, uz beleženje GPU modela i verzija torch-a i CUDA-e, koje su u runu od 24.10. nedostajale. **Korak 5:** README sa rezultatima na engleskom u `results/`, a prekidač ostaje isključen dok vlasnik i mentor ne potvrde. C je funkcija od desetak linija (`pinv` u f64, `driver=None`) koja se meri u istom notebook-u. B se radi samo ako ga Korak 0 traži, a D se ne radi.

| Šta meriti na A100 | Kako | Cilj ili prag |
|---|---|---|
| vreme solvera po pozivu, n = 1024·S, S ∈ {1, 8, 32, 64}, f64 (i f32) | `torch.cuda.synchronize` tajmeri, warm-up | zabeležiti; [Z] očekuje se ispod milisekunde do nekoliko ms |
| udeo solvera u batched `big` prolazu | prilagođena ćelija 21 | < 20% (kriterijum prihvatanja) |
| ubrzanje naspram sekvencijalnog na `big`, S = 32/64 | sweep ćelije 13–17 | ≥ 10×; plafon ~14× zbog 287 ms ostatka |
| ubrzanje na `3layer`/`4layer`/`5layer` | isti sweep | bez regresije u odnosu na 7,5×/7,0×/6,7× |
| max relativna razlika po splitu naspram sekvencijalnog | `test_batched_splits.py --device cuda`, ćelija 19 | < 1e-10 (test), < 1e-9 (sweep) |
| bitwise ponovljivost runa | dva identična runa | identično |
| peak VRAM | `torch.cuda.max_memory_allocated` | ne veći nego sa `gelsd` putem |
| C: vreme, broj sync-ova, VRAM (~328 MB U pri 65 536), slaganje sa O1 i `gelsd` | posebna ćelija | informativno |
| cena K-pravila (compat režim) | ista ćelija | mala u odnosu na solver |
| statistika ranga na CUDA naspram CPU | skripta iz Koraka 0 | jednako van pojasa |

Mac CPU brojevi su samo orijentacioni ([M], `kinds/timing.py`, n = 65 536, f64, 5 niti). Batched `gelsd` traje 139,6 ms, zatvorena forma 4,7 ms, `pinv(X) @ z` 114,2 ms, a `svdvals` 3×3 matrice 5K 41,0 ms. To je ~2,1 µs po problemu za `gelsd` na M5 Pro, naspram ~8 µs na Colab CPU-u, što proizlazi iz [P] (2 200,7 ms za 64·4·1024 problema). [Z] Na A100 je zatvorena forma desetak redukcija preko 25 vrednosti po kutiji. Ulaz je ~39 MB u f64 pri 65 536 kutija, pa je posao ograničen memorijom i očekivano daleko ispod vremena koje je trošio `gelsd`, ali to nije izmereno.

Ostaje nepoznato nekoliko stvari:

- brzina bilo koje opcije na A100;
- učestalost odsecanja na stvarnim podacima;
- da li `gelsd` daje isti rang na MKL/OpenBLAS i na Accelerate tik uz prag;
- ponašanje `gesvdjBatched` na tačno rang-deficijentnim f64 ulazima (NVIDIA ga ne dokumentuje);
- da li PyTorch CUDA kernel-i flush-uju subnormalne vrednosti;
- heuristika izbora ose u compat režimu kada su obe ose blizu praga, koja nije testirana;
- formalna granica za 8 Newton koraka kod bliskih korena.

## Zaključak

Zadatak je formulisan kao „prenesi `gelsd` na GPU", ali stvarno pitanje je koju specifikaciju solver treba da ispuni. `gelsd`-ov prag na necentriranom dizajnu je artefakt skaliranja kolona: zavisi od x̄² i ȳ, a ne od toga koliko je kutija stvarno degenerisana. Tik iznad praga `gelsd` vraća odgovor sa greškom do 1e6, a ispod praga regularizovan odgovor koji se od pravog razlikuje za O(1). Pošto faktorizacija X = Q·(5K) čini obe specifikacije jeftinim elementwise operacijama, cena više ne bira između njih. Bira test prag, a njega jeftino odlučuje jedno merenje: koliko stvarnih kutija `gelsd` odseca. Raniji zaključak „closed form: NO" bio je ispravan za svet u kome je `gelsd` uzimao 2% vremena, a bit-paritet bio cilj. U svetu u kome uzima 88% i u kome je ugovor već toleranca, zaključak se obrće.

Najveći rizik zato nije solver, nego testovi. Dva od šest postojećih tipova kutija čine `gelsd` lošim oraklom. LAPACK-ov odnos normalnih jednačina sa ‖r‖ ovde daje 1e13 i za tačno zaokruženo rešenje. Float32 je mesto gde pravilo ranga prestaje da bude stvar kompatibilnosti i postaje uslov stabilnosti. Ako se to uvede u ugovor pre prvog A100 runa (orakl je tačno racionalno rešenje, a ne `gelsd`; pravilo ranga je deo specifikacije; f32 ima svoj prag), integracija u `batched_splits.py` svodi se na jedan prekidač. Cilj od ~10× na `big` tada je ograničen još samo ostatkom od 287 ms koji nije regresija.
