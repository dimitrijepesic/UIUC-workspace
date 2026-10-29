# Dnevna evidencija i izveštaji

Plan: [DAILY_PROGRESS_PLAN.md](../DAILY_PROGRESS_PLAN.md).
Početno stanje: [PROJECT_AUDIT_2026-09-19.md](PROJECT_AUDIT_2026-09-19.md).

## Pravilo dana

Jedan glavni task; najviše tri konkretna koraka. Poslednjih 10–15 minuta
sačuvaj dokaz i upiši sledeću radnju. Beleži utrošeno vreme da bismo naredne
procene prilagodili stvarnom tempu.

Statusi: `TODO`, `U TOKU`, `GOTOVO`, `BLOKIRANO`.
„Pročitao sam” se računa kada postoji objašnjenje, izveden primer ili
zatvoreno tehničko pitanje. Benchmark se računa kada postoje protokol,
raw rezultat i tumačenje.

## Šablon dnevnog unosa

```text
Datum / task ID:
Fokusirano vreme / mašinsko vreme:
Cilj dana (jedna rečenica):
Status:
Šta sam završio (do tri stavke):
Dokaz (commit, komanda, CSV/log, beleška):
Šta rezultat znači:
Šta ne mogu još da tvrdim:
Blokada + najmanji eksperiment za sutra:
Sledeći konkretan korak:
```

## Šablon poruke na svaka 2–3 dana

```text
Zdravo Saša,

u prethodna [2–3] dana završio sam [konkretan korak].
Na [mreža, broj slika, epsilon/budžet, hardver] dobio sam [rezultat].
Provera [korektnost/protokol] pokazuje [nalaz]; [ograda ako je pilot].
Tabela/graf/kod: [link].
Sledeće radim [jedan korak], očekujem [okvir].
[Jedno pitanje, samo ako utiče na odluku.]

Dimitrije
```

Ako nema novog ubrzanja, pošalji tehnički zaključak: izolovan uzrok
numeričke razlike, mapa prenosa, minimalan reproducer, nova pouzdana
referenca ili negativno merenje. Ne izmišljati broj da bi poruka izgledala
kao rezultat. Lokalnu putanju pre slanja zameniti dostupnim linkom/prilogom.

## Evidencija izveštaja

| Izveštaj | Planirani sadržaj | Poslat datum | Link / odgovor |
|---|---|---|---|
| U01 | CPU referenca, poznati izuzeci i T1 | — | — |
| U02 | Device put, CPU solver most i GPU okruženje | — | — |
| U03 | CUDA correctness na big i status numerike | — | — |
| U04 | Hibridno vreme, profil i prvi celovit izveštaj | — | — |
| U05 | Fer ablacija sa auto_LiRPA | — | — |
| U06 | Direktne equal-time krive | — | — |
| U07 | Skaliranje i memorija | — | — |
| U08 | Nacrt i reproducibilni paket | — | — |

## 19.9.2026. — početni pregled (Codex)

- Pregledan aktivni istraživački kod, lokalna istorija, rezultati i beleške;
  sačuvani audit i dnevni plan. Razvojni kod nije menjan.
- Ponovljene ciljane CPU provere: uglovi, granica, max-error, selector putanje,
  batched regresija i dve e2e konfiguracije prolaze; grid test ima poznati
  arm64 STEPS=7/9 neuspeh uz pogrešan exit 0. CSV consistency prolazi.
- Identifikovan nesklad `25` naspram `9` solve-ova u equal-time protokolu;
  poreklo GPU CSV-a ostaje da se razjasni.
- Nema novog GPU rezultata. Današnji logovi su u `audit_2026-09-19/`.
- Sledeći korisnikov task: D01; ako već ume da objasni navedeni prolaz,
  zabeležiti dokaz i preći na D02.
- Današnji pregled nije označen kao korisnikov završen D01 niti kao poslat
  izveštaj mentoru. Poslednji sačuvani zapis o slanju je 12.9. u
  `PORUKA_SASI_2026-09-12.md`; komunikacija van workspace-a nije proverena.
- Korisnik je potvrdio 2–3 sata dnevno i dostavio Claude-ovu listu T0–T8.
  Ona ostaje tehnička osnova, uz dopune i raspored D01–D12; dani D13–D24
  su uslovni nastavak koji se revidira posle prvog hibridnog izveštaja.

## 1.10.2026. — test fix, precise-only profil, analiza kubnog lanca

- Detalji: [SESSION_2026-10-01.md](SESSION_2026-10-01.md).
- Commit `d23843d` (Pasado, `gpu-hybrid`): `test_batched_grid.py` ima strogi/
  informativni režim i pravi exit kod; mutacioni test daje `exit=1`.
- Novi `profile_precise_only.py` (3layer, 1,5 ms po forward-u): `lstsq` 18%,
  kubni lanac 17%, matmul 5%, grid 3%. 3layer ne važi za odluke o `big`.
- Analiza kubnog lanca: spajanje `ly`/`uy` grana je bit-identično, x1,18–1,20 e2e
  na 3layer; na `big` verovatno ~1% (procena). Nije implementirano u pravom kodu.
- Skripte i logovi su necommitovani. Nema novog GPU rezultata.

## 3.10.2026. — D02 / T0b zatvoren

- Šum seed-a (3layer, 3 neseedovana runa): do 1.55e-5 relativno u `lc_precise`;
  isti seed daje identičan rezultat, drugi seed do 7.8e-5. Razlog zašto
  `lc_precise_big.pth` ne može da bude referenca „≤1e-10“: neseedovan
  `np.random.normal(0, 1e-4)` u koeficijentu A. Detalji u `review/REVIEW_NOTES.md`.
- Varijante u `benchmark_lipschitz.py` i `profile_variants.py` zadaju svih pet flagova;
  novi `test_benchmark_variants.py` dokazuje da izvezeni `PASADO_*` ne prolaze
  (stari mapping: 6 od 6 varijanti kontaminirano).
- Sledeći korak: D03 / T1 (`--device`, `--num-images`, seed, precise-only).

## 4.10.2026. — D03 / T1: kontrolisani ulaz

- `get_lipschitz.py`: `--seed`, `--num-images`, `--precise-only`, `--device` (ili `PASADO_DEVICE`),
  ispis konfiguracije i ID-eva slika, tajmer sa `cuda.synchronize()`, sufiksi u imenima sačuvanih fajlova.
- CPU regresija: `lc_precise` i `lc_zonos` bit-identični pre i posle (3layer, 3 epsilona, 30 slika, seed 7).
- `experiments/test_get_lipschitz_cli.py`: 11 provera; hvata zasađene greške (seed, warning, precise-only).
- Sledeće: D04 / T2 (device higijena: `Duals.py`, `SimpleZono.py`, konstante u `inverse_poly_tensor`). Izveštaj U01 Saši.

## 4.10.2026. — D04 / T2: device higijena

- `traceify`, `Zonotope.expand`, `HyperDualIntervalToDualZonotope`, `DualIntervalTensor` nasleđuju uređaj;
  konstante u `inverse_poly_tensor` ostaju 0-dim CPU (dozvoljeno pored GPU tenzora).
- CPU: bit-identično pre i posle (3layer seed 7; 5layer prema starom kodu); svi testovi prolaze.
- `experiments/check_device_plumbing.py`: ceo precizni prolaz na MPS (float32, `lstsq` privremeno na CPU),
  svi izlazi na uređaju, razlika prema CPU 1.5e-7. Stari kod pada na `Duals.py:19`; sva četiri mesta
  pojedinačno vraćena na staro bivaju uhvaćena (dva ruše proces na MPS-u, exit 139).
- Sledeće: D05 / T3 (most `lstsq` GPU→CPU→GPU u `lin_reg_tensor_batched`, test sa uređajem kao parametrom).

## 5.10.2026. — prvi CUDA rezultat (Colab Pro, A100)

- Plumbing (T2) radi na CUDA-i: svi tenzori na `cuda:0`. Numerika sa kompleksnim kubnim rešavačem se
  razlikuje 3.9% od CPU-a; dijagnostika (`diagnose_device_diff.py`) pokazuje uzrok: imaginarni ostaci do
  3.7e-8 na CUDA-i, `torch.isreal` baca korene (39/100 neurona u sloju 0).
- Sa `PASADO_REAL_CUBIC=1` CPU i CUDA se poklapaju na 1.4e-16 (jedan slučaj; širi T5 tek sledi).
- Otvoreno za odluku: realni rešavač na GPU-u naspram kubne na CPU-u. Sledeće: D05 / T3 (most `lstsq`), pa
  `get_lipschitz.py --device cuda`. U01 Saši sada ima pravi rezultat i pravi problem.

## 5.10.2026. — D05 / T3: most za lstsq

- `lin_reg_tensor_batched` rešava `gelsd` na CPU-u kad su ulazi na GPU-u i vraća rezultat na uređaj.
  CPU bit-identično; test sa uređajem (MPS: 13/13 identično CPU rešenju); probne skripte bez privremene zamene.
- Merenje na `big` (CPU, precizni prolaz, 6 prolaza): matmul `AffineZonotope` 66.5%, `lstsq` 3.3%. Amdahl: gornja
  granica ubrzanja ~3x, realno 2 do 3x; `gels`/CUDA `gelsd` nisu potrebni. Procena, ne rezultat.
- Sledeće: `get_lipschitz.py --device cuda` na Colab-u (CPU naspram CUDA preko CLI-ja), pa odluka o realnom kubnom
  rešavaču i T5 (širi skup slučajeva).

## 5.10.2026. (kasno) — D08 stigao ranije: CPU naspram CUDA preko CLI-ja

- A100 80 GB, realni kubni rešavač: razlika CPU/CUDA 8.4e-16 (3layer) i 8.9e-16 (big). Plan je D08 imao za 9.10.
- Realni naspram kompleksnog rešavača (CPU): pomeraj `lc_precise` do 8.3e-6; obe varijante validne prema autograd-u.
- Grubo vreme na big: 0.514 s (Colab CPU) naspram 0.063 s (A100) po slici; nije merenje (T6).
- Sledeće: odluka o realnom rešavaču (preporuka: da); T5 širi skup slučajeva; T6 kontrolisano merenje; U01 Saši.

## 8.10.2026. — T5 završen: puna A100 korektnosna matrica i mutacioni testovi

- Detalji: [SESSION_GPU_HYBRID_2026-10-08.md](SESSION_GPU_HYBRID_2026-10-08.md) i [HANDOFF_GPU_HYBRID_2026-10-10.md](HANDOFF_GPU_HYBRID_2026-10-10.md).
- Commitovi `f51f977`, `c2a2c65`, `d765ab3`, `125ddf2` (`Pasado`, `gpu-hybrid`).
- Mutation harness (`experiments/mutation_check.py`): svih 9 testova zeleni u osnovi, svih 10 zasađenih mutanata (M1–M10) uhvaćeno.
- Puna verifikacija na `NVIDIA A100-SXM4-80GB` (`results/gpu_hybrid_verification_2026-10-08/`): 4 mreže (`3layer`, `4layer`, `5layer`, `big`), 30 slika, svih 16 epsilona — `28/28 PASS`.
- Maksimalna relativna razlika sa realnim (Viète) rešavačem u `float64`: `1.786e-15` (`precise`), `4.792e-16` (`zonotope`), `5.517e-16` (`interval`); u `float32`: `8.318e-7` (prag `1e-5`). Ponovljen CUDA run i `full` naspram `--precise-only`: `0`.
- Negativna kontrola (stari kompleksni Cardano rešavač): divergencija CPU/CUDA od `14.39%` (`3layer`) do `63.78%` (`big`).

## 16.10.2026. — Task A: merenje CPU fallback-a (`lin_reg_tensor_batched`) na CUDA-i

- Commitovi `a4dd045` i `03bd011`; rezultati u `Pasado/results/gpu_hybrid_fallback_2026-10-16/`.
- Skripte `experiments/measure_hybrid_fallback.py` i `experiments/measure_sync_free_hybrid.py` (3 slike × 3 epsilona, 5 warm-up + 10 merenih ponavljanja = 90 prolaza po konfiguraciji, `float64`).
- Na `big`: medijana `precise` prolaza je `532.31 ms` na Colab CPU-u (6 niti) naspram `63.34 ms` na CUDA hibridu (`58.62 ms` bez nepotrebnih host-sync provera u petlji slojeva).
- Udeo CPU fallback-a (`D2H + design + gelsd + H2D`) u ukupnom CUDA `precise` vremenu: `18.00%` (`3layer`), `18.75%` (`4layer`), `19.11%` (`5layer`), i **`54.54%`** na `big` (`34.57 ms` od `63.34 ms`, od čega sam `gelsd` nosi `32.11 ms`, a D2H/H2D prenosi `1.40 ms`).
- Odluka prema kriterijumu iz handoff-a (> 15%): na `big` je CPU `gelsd` glavno usko grlo, pa ima smisla ili GPU SVD/minimum-norm solver ili paralelizacija nezavisnih podproblema (npr. batching po splitovima).

## 17.10.2026. — Task B i Task C: layerwise divergencija i `--n-splits` sweep (1..16)

- Commitovi `3b7b0a2`, `3b35688` (layerwise) i `e894470`, `600ba2c` (split sweep).
- Layerwise analiza (`Pasado/results/gpu_hybrid_layerwise_2026-10-17/`):
  - Sa kompleksnim rešavačem na `big` (`image_id=0, eps_idx=0`), `torch.isreal` na CUDA-i odbacuje `2348–2812` od `6144` kandidata po sloju; relativna razlika bounda raste kroz slojeve (`0.77% -> 23.6% -> 19.8% -> 44.3% -> 65.2%` na izlazu).
  - Dodatni nalaz: i na samom CPU-u kompleksni put gubi deo validnih korena blizu `s ≈ 0` ili `1` zbog `float32/complex64` preciznosti (npr. u sloju 2 na `big`, `2048` korena u kutiji naspram `2348` sa realnim Viète rešavačem).
  - Sa realnim Viète rešavačem, CPU i CUDA zadržavaju identičan broj korena u svakom sloju, a razlika izlaznog bounda ostaje na nivou mašinske preciznosti (`<= 1.41e-15` preko 25 parova slika × epsilon).
- Split sweep `n_splits = 1, 2, 4, 8, 16` (`Pasado/results/gpu_hybrid_split_sweep_2026-10-17/`):
  - Vreme raste linearno `O(n)` jer se podintervali obilaze sekvencijalno; u postojećem kodu `lc_precise` opada sa brojem splitova.

## 19.10.2026. — Prošireni split sweep do `n_splits = 64`, VRAM i kompleksni kubni fix-evi

- Commitovi `603a48c`, `25e6092`, `0eb14cd`.
- Split sweep proširen na `n_splits = 1, 2, 4, 8, 16, 32, 64` uz praćenje vršne CUDA memorije (`peak_vram_mb`) i dual-axis grafike (`experiments/plot_split_sweep.py`).
- Peak VRAM ostaje konstantan `O(1)` (`10.78 MB` na `3layer`, `480.52 MB` na `big` za sve `n_splits = 1..64`).
- `experiments/verify_complex_cubic_fixes.py`: upoređeni fiksni prag (`|Im(x)| <= 1e-6` uz proveru kubnog ostatka) i analitička procena zaokruživanja inspirisana Miné (2004), uz popravku eksponenta `0.33333 -> 1.0/3.0`. Na CUDA-i oba pristupa zadržavaju svih `3000/3000` validnih korena (naspram `1895/3000` kod `torch.isreal`) i poklapaju se sa Viète rešavačem.

## 20.10.2026. — Otkrivena dva upstream baga u originalnom Pasado kodu

- Autograd (`jvp`) provera nad izlazima split sweep-a otkrila je dva baga koja postoje u originalnom (`origin/main`) Pasado kodu:
  1. **`Zonotope.__neg__` i `DualZonotope.__neg__` (`SimpleZono.py:115-117, 369-370`) ne negiraju generatore:** vraćaju `Zonotope(-self.centers, self.generators)` umesto `-self.generators`. Zbog toga izraz `(one - zono_eps) * img + zono_eps` u `get_lipschitz.py` pogrešno postavlja smer generatora (`1 + img` umesto `1 - img`), pa za podintervale `[l, r]` sa `l > 0` (`n_splits > 1`) realni zonotop više ne obuhvata stvarnu putanju zamagljenja i `lc_precise` može da padne ispod stvarnog izvoda mreže (npr. `4layer`, slika 0, `eps_idx=8`, `n_splits=16`: `lc_precise = 177.1394 < true_max = 177.1797`).
  2. **`check_nonlinear_boundary` i `_max_objective_over_x_candidates` (`precise_transformer.py:417, 440, 453`) uzimaju samo `max(E)` umesto `max(|E|)`:** za razliku od `check_corners` koji proverava obe strane, nelinearna granica ignoriše najnegativniju grešku `E < 0` u unutrašnjim kritičnim tačkama.
- Sa obe ispravke u memoriji, sva narušavanja `lc_precise < autograd_max` nestaju (`0/99`), a na `big` (`n_splits=1`) bound je čak uži (`9820.98` umesto `14466.40`).
- Sledeći korak: napraviti čist reproducer skript za oba upstream baga, obavestiti Sašu, i nakon dogovora primeniti ispravke na novoj grani i ponoviti split sweep (uz GPU paralelizaciju po splitovima).

## 22.10.2026. — Ispravljena oba upstream baga, verifikovan reproducer i pripremljen PR + handoff

- Detalji: [HANDOFF_GPU_HYBRID_2026-10-22.md](HANDOFF_GPU_HYBRID_2026-10-22.md).
- U `Pasado` (`gpu-hybrid`):
  - Ispravljeni `Zonotope.__neg__` i `DualZonotope.__neg__` u `SimpleZono.py` (`-self.generators`) i dodato `.abs()` u `check_nonlinear_boundary`, `check_nonlinear_boundary_softplus` i `_max_objective_over_x_candidates` u `precise_transformer.py`.
  - Dodat `experiments/verify_upstream_bugs.py` (algebarska provera + autograd provera na `4layer` i `big`). Svih 9 testova i `mutation_check.py` prolaze.
  - Očišćen duplikat foldera `gpu_hybrid_to64_and_cubic_fixes/` u root-u i dopunjene napomene u README fajlovima rezultata.
- U `Pasado-original`: pripremljena čista grana `fix-zonotope-neg-and-boundary` direktno od `origin/main` (`e857413`) sa 6 linija ispravke za PR ka `uiuc-arc/Pasado`.
- Sledeći koraci:
  1. Komitovati izmene i otvoriti upstream PR.
  2. Implementirati GPU paralelizaciju (batching) po `n_splits` i ponoviti split sweep sa ispravljenim bagovima.
  3. Uraditi kompajlersku evaluaciju na `big` (`n_splits = 32`) kroz **TorchInductor**, **TVM** i **XLA** (uporediti vreme i logove fuzije).



## 24.10.2026. — GPU paralelizacija po `n_splits` (batching), CPU verifikacija

- Novi modul `Pasado/forward_mode_tensorized_src/batched_splits.py`: svih `S` podintervala prolazi kroz precizni forward odjednom (`[S, N]` centri, `[S, G, N]` generatori); `(split, neuron)` boksovi se spljošte u `[S*N]` za grid, `lstsq` i proveru greške. Parametar `chunk` ograničava memoriju (raste linearno sa `S`).
- `experiments/test_batched_splits.py`: poređenje svake pojedinačne granice sa sekvencijalnom petljom (seed 7, `float64`, CPU): `3layer`/`4layer`/`5layer`/`big`, `n_splits` 1–8, max relativna razlika `2.5e-14`; chunking, nezavisnost splitova i nesigurni (bez seed-a) slučaj naspram autograd-a prolaze. 7 zasađenih mutanata uhvaćeno.
- `measure_split_sweep.py` dobio `--split-mode {sequential,batched}` i `--split-chunk`; CSV ima kolone `split_mode`, `split_chunk`. Na CPU (`3layer`, `n=16`) isti `lc_precise`, `31.2 ms` -> `10.8 ms`; GPU merenje tek sledi (Colab).
- Napomena: `gelsd` ostaje na CPU-u i radi nad `S*N` malih problema, pa se očekuje da on postane usko grlo; to treba izmeriti na `big`.

## 25.10.2026. — CUDA rezultati batchinga po `n_splits`: GPU deo ubrzan, `gelsd` postaje usko grlo

- Rezultati: `Pasado/results/gpu_hybrid_batched_splits_2026-10-24/` (README sa tabelama).
- Korektnost na CUDA-i: svaka pojedinačna granica batched naspram sekvencijalnog `<= 2.3e-14`; u sweep-u najveća razlika `lc_precise` `1.2e-14` preko 56 ćelija.
- Ubrzanje (`eps_idx 0`, `n_splits = 64`): `3layer` 7.5x, `4layer` 7.0x, `5layer` 6.7x, ali `big` samo 1.6x.
- Uzrok: CPU `gelsd` u batched prolazu na `big` nosi `65%` (`n=1`) do `88%` (`n=64`) vremena, `~34 ms` po splitu bez obzira na veličinu batch-a. Bez njega bi `big` bio oko 14x brži (procena iz merenja).
- Memorija batched prolaza raste linearno sa brojem splitova: `big` `503 MB` -> `29.8 GB` pri `n=64`.
- Prototip zatvorene formule za regresiju (mreža 5x5 je Dekartov proizvod, normalne jednačine su dijagonalne; degenerisani slučajevi preko minimalne norme): na realnim podacima razlika granica prema `gelsd` `<= 1.2e-13`, a tačnost prema egzaktnom racionalnom rešenju bolja od `gelsd`. Prvi sloj ima `ly == uy` za sve neurone. Nije u repou; opis i rizici u `PROMPT_GPU_REGRESSION_2026-10-25.md`.
- Sledeće: GPU rešavač za regresiju (opcija A/B iz beleške), pa Colab sweep; zatim kompajleri na `big`.
