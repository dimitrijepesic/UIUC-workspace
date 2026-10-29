# Handoff: batching po `n_splits` gotov, sledeći zadatak je GPU zamena za `gelsd` — 25.10.2026.

Privatna radna beleška za nastavak u novom razgovoru. Fajl je izvan `Pasado` repoa
i ne sme u `Pasado` commit ili PR. Zamenjuje `HANDOFF_GPU_HYBRID_2026-10-22.md`
(njegovi koraci 1 i 2 su gotovi; korak 3 su kompajleri i dalje otvoreni).

Razgovor sa korisnikom ide na srpskom. Kod, komentari i docstringovi u `Pasado` su na engleskom.

---

## 1. Pravila rada (obavezno)

- **Nikakva AI atribucija** nigde: commit poruke, komentari, docstringovi, dokumenta, logovi.
  Nema `Co-authored-by`, "Generated with", imena AI alata.
- **`git commit` se ne pokreće iz agent okruženja** (može samo da doda trailer). Ako korisnik
  izričito traži da se commituje, koristi se plumbing: `git add`, `git write-tree`,
  `git commit-tree`, `git update-ref`, sa
  `GIT_AUTHOR_*` i `GIT_COMMITTER_*` = `Dimitrije Pešić <dimitrije.pesicc@gmail.com>`,
  pa se poruka pročita nazad i skenira. Globalni git identitet se nikad ne menja.
- Pre svakog push-a skenirati commit-ove za atribuciju (`git log fork/gpu-hybrid..HEAD`).
  Push ide na remote `fork` (`dimitrijepesic/Pasado`), nikad na `origin` (upstream
  `uiuc-arc/Pasado`) bez izričitog odobrenja.
- Poslednji put je korisnik tražio: "nemoj nikakve md fajlove da commitujes". `.md` fajlove
  ne commitovati dok ne kaže drugačije.
- Okruženje: Python je `Pasado/.venv/bin/python` (torch 2.12, samo CPU na Mac-u, MPS nije
  dostupan u sandbox-u). CUDA merenja idu isključivo preko Colab sveske koju korisnik pokreće.
- Sandbox: pisanje je dozvoljeno samo u workspace. `/tmp` ne radi za izlazne fajlove
  eksperimenata; privremene fajlove pravi u workspace-u i obriši ih. Mreža (push, `gh`)
  traži `full_network` i `git_write`. `$VAR` sa razmacima u zsh se ne deli na reči;
  složene komande pokreći kroz `bash -c`.
- Korisnik ne zna unapred da odluči o numeričkim izborima ("ne znam da odgovorim");
  zato se odluke pretvaraju u merenja sa pravilom izbora zapisanim unapred (odeljak 5).

---

## 2. Stanje repoa

- `Pasado`, grana `gpu-hybrid`, HEAD `842e9c9`, **sve pushovano** na `fork`.
  Poslednja tri commita (25.10., u korisnikovo ime):
  - `b144542` feat: run interval splits as one batched forward pass, with per-split equivalence test
  - `b0eb02d` feat: add --split-mode and --split-chunk to the split sweep
  - `842e9c9` docs: archive CUDA results for batched vs sequential split sweep
- Prethodno (22.10.): `b526762` popravka dva upstream baga, `def845d`, `69a7163`.
- Upstream PR za bagove: https://github.com/uiuc-arc/Pasado/pull/3 (grana
  `fix-zonotope-neg-and-boundary`, worktree `Pasado-original`), otvoren. Saša nije ništa
  pisao, korisnik traži da se pretpostavi da je to rešeno.
- Necommitovano u `Pasado`: samo `results/gpu_hybrid_batched_splits_2026-10-24/README.md`
  (namerno izostavljen kao `.md`).
- Spoljašnji repo `UIUC` (nije commitovano ništa od danas): `COLAB_GPU_HYBRID_BATCHED_SPLITS.ipynb`,
  `notes/DAILY_LOG.md` (unos za 24.10. i 25.10.), `notes/PROMPT_GPU_REGRESSION_2026-10-25.md`,
  ovaj fajl, i ranije necommitovane sveske i beleške. `research_notes/GPU zamena za gelsd/`
  (pet fajlova i `scripts/`) napisao je drugi agent; to je beleška po pravcima, objedinjeni
  izveštaj `reports/GPU zamena za gelsd.md` još ne postoji.
- Plan za PR-ove ka upstream-u razgovaran, ništa urađeno: `gpu-hybrid` je 61 commit /
  103 fajla ispred `origin/main`. Za svaki PR nova grana od `origin/main`, fajlovi uzeti sa
  `gpu-hybrid` (`git checkout gpu-hybrid -- fajlovi`), ne cherry-pick celu istoriju.
  Predloženi redosled: (1) bagovi (PR #3), (2) vektorizovani precizni transformer + testovi,
  (3) podrška za uređaje, (4) realni Viète kubni rešavač, (5) interval splitting.
  `results/`, `logs/`, beleške, sveske, profilisanje ne idu u upstream PR-ove.
  Prvo pitati Sašu da li upstream uopšte želi GPU/splitting deo.
- Veličina koda vs `origin/main` (samo `.py`/`.sh`): oko 8 960 dodatih linija; biblioteka
  721, testovi 1 581, eksperimenti 6 659.

---

## 3. Šta je urađeno 24.-25.10.

### 3.1 Batching po `n_splits`

`Pasado/forward_mode_tensorized_src/batched_splits.py`: svih `S` podintervala u jednom
forward prolazu. Centri `[S, N]`, generatori `[S, G, N]`; boksovi (split, neuron) se
spljošte u `[S*N]` za grid, regresiju i proveru greške (koriste se postojeće
`get_linspace_batched`, `lin_reg_tensor_batched`, `check_corners_tensor`,
`check_nonlinear_boundary_tensor`). Parametar `chunk` ograničava memoriju.
Sa seed-om svi delovi dele isti tok šuma (reprodukuje sekvencijalni protokol gde se
`np.random` resetuje pre svakog dela); bez seed-a svaki par (deo, neuron) ima svoj šum.
Nije preuzet "svi lb == ub" prečac iz `SigmoidZonotope` (dokazano bezazlen).
`extract_affine_layers` vraća `detach()`-ovane tenzore (greška iz Colab-a: `net.to(...)`
pomeralo je `Parameter` bias).

`experiments/test_batched_splits.py`: svaka pojedinačna granica batched vs sekvencijalno
(seed 7, `float64`), chunking, nezavisnost splitova, nesigurni (bez seed-a) slučaj naspram
autograd-a (`check_sweep_soundness.sampled_max_derivative`). 7 zasađenih mutanata je uhvaćeno
na CPU-u, ali **mutanti nisu sačuvani** (bila je privremena skripta) i `mutation_check.py`
ne pokriva ovaj test.

`experiments/measure_split_sweep.py`: nove opcije `--split-mode {sequential,batched}` i
`--split-chunk`; CSV ima kolone `split_mode`, `split_chunk`.

### 3.2 CUDA rezultati (Colab, `float64`, GPU model nije zapisan u logu; `--split-chunk 64` implicira >= 64 GiB)

Folder: `Pasado/results/gpu_hybrid_batched_splits_2026-10-24/` (README tamo ima sve tabele).

- Korektnost: svaka granica batched vs sekvencijalno `<= 2.3e-14`; u sweep-u max razlika
  `lc_precise` `1.2e-14` preko 56 ćelija. Test na CUDA-i: `ALL CHECKS PASSED`.
- Ubrzanje (`eps_idx 0`, medijana): `3layer` 4.6x (S=8), 6.8x (S=32), **7.5x (S=64)**;
  `4layer` 7.0x; `5layer` 6.7x; **`big` samo 1.6x** za sve `S >= 8`.
- Uzrok za `big`: CPU `gelsd` je 65% (S=1), 84% (S=8), 86% (S=32), **88% (S=64)** batched
  prolaza; `~34 ms` po splitu bez obzira na `S` (LAPACK petlja po `S*N` malih problema).
  Ostatak prolaza je `287 ms` na S=64, pa bi bez `gelsd` `big` bio oko `4067/287 = 14x`
  brži od sekvencijalnog (procena, nije merenje).
- Memorija batched prolaza raste linearno: `big` `503 MB` (S=1) -> `29.8 GB` (S=64);
  sekvencijalni je konstantan (`480 MB`).

### 3.3 Provere na pravim podacima (CPU, `float64`, `3layer` i `big`)

- **Prvi sloj ima `ly == uy` za 100% neurona** (dual deo ulaza ima nulte generatore), x nikad
  nije tačno degenerisan, a slojevi >= 2 nemaju tačno degenerisanih boksova. Dakle, rešenje
  minimalne norme je svaki boks prvog sloja, nije izuzetak.
- Uski boksovi (širina relativno prema `1 + |x̄| + |ȳ|`): `eps_idx 0` praktično nema ispod 1e-3;
  `eps_idx 8` oko 1% do 25% x-boksova ispod 1e-3; `eps_idx 15` svi x-boksovi ispod 1e-3 i
  0.1% do 0.6% ispod 1e-6, do 50% y-boksova u dubljim slojevima ispod 1e-3.
- Tačnost prema egzaktnom racionalnom rešenju (`fractions.Fraction`, isti float x, y, z),
  `big`, slojevi 2-4, 150 boksova po sloju:
  - `eps_idx 0`: `gelsd` medijana 1e-16 (max 2e-14), zatvorena forma 4e-17 (max 2e-15)
  - `eps_idx 8`: `gelsd` 4e-15..8e-15 (max 6e-13), zatvorena forma 2e-16 (max 9e-15)
  - `eps_idx 15`: `gelsd` 2e-13..3e-13 (max 7.6e-12), zatvorena forma 7e-16 (max 9e-15)
- Granica do kraja: zatvorena forma umesto `gelsd` (bez rezervnog puta), po-split razlika
  `<= 1.2e-13` na `3layer` i `big`, `eps_idx 0/8/15`, S=4, 3 slike.
- Merenje 25.10.: `gelsd` greška na jednom uskom boksu (x̄=0.7) raste otprilike kao `1/w`
  (2e-15 pri w=1e-2, 6e-8 pri w=1e-10). Izveštaj drugog agenta tvrdi `1/w²`; nije reprodukovano.
  Zaključak ostaje isti: `gelsd` nije referenca za uske boksove.

---

## 4. Sledeći zadatak: GPU zamena za `gelsd` (regresija ravni)

Problem u jednoj rečenici: regresija `z = sigmoid'(x)*y` na 5x5 mreži po boksu, `X = [1, x, y]`
(25x3), rešenje `(C, A, B)` minimalne norme kao `gelsd` sa `rcond = eps*max(m,n)`; treba
batched GPU rešenje za oko 65 000 problema; zatim se `A += N(0, 1e-4)` (ostaje kakav je).
Poziva se u `batched_splits.precise_sigmoid` (`pt.lin_reg_tensor_batched`). Detaljan opis,
kod prototipa, rizici R1-R10, kandidati i kriterijumi: `notes/PROMPT_GPU_REGRESSION_2026-10-25.md`
(sadrži radni prototip zatvorene forme).

### 4.1 Matematika (sve proveravano)

- Centrirane kolone su ortogonalne jer je mreža Dekartov proizvod (provereno), pa je
  `A = Σ dx·dz / Σ dx²`, `B = Σ dy·dz / Σ dy²`, `C = z̄ − A·x̄ − B·ȳ`.
- Degenerisano (rešenje minimalne norme): x konstantno (`x0`): `C = k/(1+x0²)`,
  `A = k·x0/(1+x0²)`, `k = z̄ − B·ȳ`; y simetrično; oba: `z̄·(1,x0,y0)/(1+x0²+y0²)`.
- **Zamke koje su stvarno ujele prototip:** (1) srednja vrednost 25 jednakih brojeva nije tačno
  taj broj, pa se degeneracija mora detektovati po krajevima (`lx==ux`, `ly==uy`), ne po
  `Σ(x−x̄)²==0`; (2) centriranje oko izračunate srednje vrednosti kvari tačnost kod uskih boksova
  (greška do 8e-8, a granica je u jednoj varijanti bila 2 do 3.6 puta pogrešna); centrirati
  oko tačke iz podataka (`x - x[:,:1]`), pa oduzeti srednju vrednost malih razlika
  (Chan, Golub, LeVeque 1983); (3) `torch.where` evaluira obe grane, imenilac mora biti
  zaštićen (`where(deg, 1, sxx)`).
- Egzaktni identitet (provereno, `1.3e-13`): `σ(X) = 5·σ(K)`,
  `K = [[1, x̄, ȳ], [0, σx, 0], [0, 0, σy]]` (σx, σy populacione std). Omogućava
  `gelsd`-ov `rcond` test nad 3x3 umesto 25x3.
- Separabilna forma (provereno, `2.3e-12`): `z = h(x)·y`, pa `B = mean(h)`,
  `A = ȳ·cov(x,h)/var(x)`, `C = h̄·ȳ − A·x̄ − B·ȳ`; samo 5 evaluacija `σ'`; nije bit-identična
  sa postojećim `(1−s)*(s*y)` targetima.
- Zdravost (R1): svaka ravan je zdrava ako je greška `D = max|σ'(x)y − (Ax+By+C)|` tačno
  izračunata (uglovi + stacionarne tačke na y-ivicama; E je linearno u y). `A`, `B`, `C`, šum na
  `A` i `A != 0` ne ulaze u dokaz. Implementacija može da pokvari: promašen koren, `max(E)`
  umesto `max|E|` (popravljeno 22.10.), D iz drugih koeficijenata od upisanih (npr. posle
  cast-a u fp32), zaokruživanje. Pasado nije floating-point sound.
- Poznato iz repoa: `gels` na CUDA-i je na rank-deficient boksovima pogrešan do `3.35e34`
  (`results/optimization_section_5_4/GPU_READINESS.md`, faza 12); zatvorena forma je ranije
  odbačena u `LSTSQ_ANALYSIS.md:137-139` ("kvadrira uslovljenost") kada je `gelsd` nosio
  samo oko 4% vremena i kada nije bilo centriranja oko pivota; `pinv`/SVD nikad nije meren.
- Neproveravano (tvrdi drugi agent, bez pristupa mreži): PyTorch nema `gelsd` na CUDA-i ni u
  2.14.1 (issue-i #72711, #117122, PR #176746); cuSOLVER, MAGMA, JAX, CuPy, TF nemaju batched
  least squares sa otkrivanjem ranga; `pinv` na CUDA-i ide preko `cusolverDnDgesvdjBatched`,
  podrazumevani `rtol` isti kao `gelsd`, plus jedan host sync.

### 4.2 Kandidati

| oznaka | solver | pravilo za rang |
|---|---|---|
| G | `gelsd` na CPU-u (današnje) | `rcond` cutoff |
| A1 | zatvorena forma, 25 targeta | tačni krajevi (`lx==ux`) |
| A2 | zatvorena forma, 25 targeta | `gelsd` cutoff preko `σ(K)` |
| A3 | separabilna forma, 5 `σ'` | tačni krajevi |
| P | `torch.linalg.pinv` na GPU-u | `rcond` cutoff (gesvdj) |
| H | zatvorena forma + CPU `gelsd` samo na boksovima koje označi A2 cutoff | kao G na označenim |

H je dodat u razgovoru (nije bio u listi drugog agenta). Napomena: P se na Mac-u ne može
verodostojno izmeriti (CPU `pinv` koristi drugi algoritam), pa nalazi o njegovom kvalitetu
čekaju Colab; na Mac-u samo gruba slika.

---

## 5. Pravilo izbora (zapisati u zaglavlje harness-a i commitovati PRE prvog merenja)

Isti princip kao "numerički ugovor" iz T5. Redosled:

1. **Uslov 1, zdravost.** Konačna granica je uvek >= autograd uzorkovanog maksimuma; pored
   toga `D` po boksu >= pravi `max|E|` za tu ravan, izračunat sa 50 cifara (mpmath, egzaktni
   koreni). Ko padne, ispada. Autograd uslov je samo filter za greške, ne rangira (svaka ravan
   je zdrava); mpmath provera `D` je glavni deo.
2. **Uslov 2, robusnost.** Bez NaN/inf i bez pada `A != 0` asserta, na svim familijama
   boksova (uključujući `degenerate_y`, boksove oko cutoff-a sa obe strane, overflow,
   `saturated`, NaN/inf ulaze), u `float64` i `float32`.
3. **Kriterijum 1, tesnoća.** `lc_precise` po splitu na 4 mreže, više slika, `eps` 0/8/15,
   `S` do 64, isti seed (šum na `A` je tada isti za sve kandidate). Po boksu još širina
   `W = |A|·r_x + |B|·r_y + D`. Očekivanje: izjednačeno, jer je razlika među solverima
   (~1e-13) osam redova manja od šuma na `A` (1e-4); pobednika će odlučiti robusnost.
4. **Kriterijum 2, tačnost.** Rastojanje od egzaktnog `Fraction` rešenja; odlučuje samo ako
   je tesnoća izjednačena.
5. **Brzina na A100** kao poslednji tie-breaker.

Referenca za koeficijente je egzaktni orakl (`Fraction`), ne `gelsd`; `gelsd` se koristi samo za
dobro uslovljene boksove. Krajnji ugovor: granica ostaje u `1e-10` naspram sekvencijalnog puta
(postojeći `test_batched_splits.py`) uz autograd proveru.

### 5.1 Tri nivoa provere i rano odsecanje

- **Nivo boksa (Mac CPU):** boksovi izvučeni iz pravih prolaza svih mreža + sintetički
  adversarijalni (fiksirane familije, broj slučajeva i seed zapisani unapred). Svi uslovi i
  kriterijum 2. Ko ne prođe uslove 1 i 2, ne ide dalje.
- **Nivo mreže (Mac CPU):** `batched_splits` sa prekidačem za solver; kriterijum 1 i autograd
  zdravost, samo za one koji prežive.
- **Nivo GPU-a (Colab):** brzina i ponovljena korektnost na CUDA-i; ovde i P.

Familije boksova koje moraju da uđu (postojeći tipovi u `experiments/test_batched_lin_reg.py`:
`normal`, `degenerate_x`, `point`, `near_degenerate`, `saturated`, `mixed_scale`) plus:
`degenerate_y` (svaki boks prvog sloja), boksovi tačno oko `rcond` praga, ogromne magnitude
(`x0² ` preko overflow-a: preko `1.3e154` u fp64, `1.8e19` u fp32; rešenje reskaliranje `t = 1/x0`),
NaN/inf, `float32`. Metamorfne relacije sa egzaktnim očekivanjem: obrnut x grid daje `B -> −B`,
zamena osa x/y, `z·2^k`, `z + linearna funkcija`. LAPACK-ovi test odnosi: rezidual, normalne
jednačine (`Xᵀr ≈ 0`), rešenje u row space-u (min-norm). GPU specifičnosti: TF32 za `float32`
matmul, FMA, redosled redukcija. Planirani mutanti (cilj ~13): pogrešna formula za degenerisan
boks (promakne kad je centar 0), bez pivot-centriranja (vidi se samo na loše uslovljenim),
srednja vrednost umesto egzaktne detekcije degeneracije, pogrešan znak, nezaštićen imenilac, itd.

---

## 6. Plan rada (redosled)

1. Orakl (`Fraction`, a mpmath za `D`) i harness za nivo boksa u `Pasado/experiments/`,
   necommitovano; `precise_transformer.py` se ne dira. Pravilo izbora iz odeljka 5 upisati u
   zaglavlje harness-a i da korisnik to commituje pre prvog pokretanja.
2. Pet (šest sa H) kandidata u jednom novom modulu pored `batched_splits.py`, prekidač
   podrazumevano isključen; sekvencijalni put netaknut.
3. Mutanti u `mutation_check.py`, uz dodavanje `test_batched_splits.py` na njegovu listu
   (sada ga ne pokriva), uz čuvanje mutanata u repou.
4. Povezati solver sa `batched_splits.py` kroz **jednu zajedničku ulaznu funkciju** (npr.
   `solve_plane`) na koju se kači merač. Razlog: ćelija u Colab sveski za udeo `gelsd`-a
   zamenjuje `pt.lin_reg_tensor_batched` svojim omotačem, pa novi solver ne bi bio izmeren;
   isto važi za merne skripte (`measure_*`) koje menjaju `pt.lin_reg_tensor_batched`.
5. Colab: G, A1/A2/A3, H i P na `big`, `S=32/64`, uz ponovljenu korektnost; pa
   kompajleri (TorchInductor, TVM, XLA) na `big` sa `n_splits=32` (Sašin zahtev iz 22.10.).
   Kompajleri imaju smisla tek kad `gelsd` više nije 88% vremena.

Ko piše kod: u ovoj sesiji je korisnik tražio da kod pišem ja i da commit-ujem u njegovo ime.
Drugi agent je pomenuo da korisnik "od 19.9. sam kuca kod"; pitanje je ostalo bez odgovora.
Pitati na početku sledećeg razgovora ili nastaviti kao u ovoj sesiji.

---

## 7. Korisno za sledeći razgovor

- Pokretanje testova: `Pasado/.venv/bin/python experiments/test_batched_splits.py [--quick] [--device cuda]`.
- Sweep: `python experiments/measure_split_sweep.py --device cuda --split-mode batched --split-chunk 64 --splits 1 2 4 8 16 32 64 --networks big`.
- Colab: `COLAB_GPU_HYBRID_BATCHED_SPLITS.ipynb` upisuje tri fajla preko `%%writefile`; ako se
  izmeni kod, **sveska se mora ponovo generisati i ponovo uploadovati** (stara kopija u Colab-u
  upisuje staru verziju; to se već desilo). Ćelija za kloniranje radi `os.chdir("/content")`
  pre `rmtree` (inače `git` pada pri ponovnom pokretanju).
- Šta izvršava Colab sveska: test, tri sweep-a (`sequential`, `batched` male mreže,
  `batched` big sa chunk-om iz VRAM-a), poređenje, udeo `gelsd`-a, grafik, zip.
- Ekstremni boksovi i `float32` još nisu testirani sa zatvorenom formom; CUDA vreme zatvorene
  forme nije mereno.
- Ostalo otvoreno iz ranije: PR #3 upstream čeka pregled; objedinjeni izveštaj o `gelsd`
  zameni (`reports/GPU zamena za gelsd.md`) još ne postoji; Trainium/TorchNeuron je samo
  napomena za diskusiju.
