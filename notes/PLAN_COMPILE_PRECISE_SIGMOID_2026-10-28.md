# Plan: kompajliranje celog `precise_sigmoid` (sledeći task, ne počinjati pre auto-dispatch-a)

Privatna beleška, 28.10.2026. Ne ide u Pasado commit.

## Zašto

Posle kompajliranog A2 male mreže su vezane za ostatak preciznog transformera: batched
prolaz na 3layer traje oko 9 ms bez obzira na broj splitova, a solver nosi samo oko
2 × 0,24 ms (`results/v3_compiled_cuda_2026-10-28/`). Isti launch overhead postoji u
proveri uglova, granice i kubnoj jednačini.

## Stanje, izmereno lokalno (CPU, torch 2.12.1, 3layer, 2 splita)

- `torch._dynamo.explain(batched_splits.precise_sigmoid)`:
  - sa `planar_fit`: **2 grafa (360 + 168 operacija), 1 prekid**:
    `batched_splits.py:155`, `assert bool(((As > 0) | (As < 0)).all())` (`Tensor.item()`,
    host sync);
  - sa `gelsd`: 5 grafova, 4 prekida (`lstsq`, dinamički oblik u
    `precise_transformer.py:162`, i isti assert).
  - Boolean maske u `inf_to_nan` / `filter_range` (in-place `b[guard] = nan`) nisu
    prekinule graf, ali treba proveriti kako ih Inductor spušta.
- **Šum na A se menja pod kompilacijom.** Dynamo ne prekida graf na `np.random.normal`
  u `_perturbation`, nego ga prevodi. Posle `np.random.seed(7)`: stanje NumPy generatora
  posle poziva 236 (eager) naspram 624 (kompajlirano); dualni centri se razlikuju oko
  1e-5 relativno; dva uzastopna kompajlirana poziva sa istim seed-om daju različite
  rezultate. Seed-ovani protokol (isti šum kao sekvencijalni kod) bi se time pokvario.

## Koraci

1. **Šum kao tensor ulaz.** `_forward_chunk` vuče šum za svaki sloj pre poziva
   (`_perturbation`, isti redosled NumPy poziva kao sada) i prosleđuje ga u
   `precise_sigmoid(z, noise)`. Kompajliran deo nema NumPy poziv. Test: seed-ovani
   per-split boundovi kompajliranog puta jednaki eager A2 putu do tolerancije, i stanje
   NumPy generatora posle prolaza isto kao eager.
2. **Assert van grafa.** `A != 0` računati u grafu kao maska, vratiti jedan bool tensor
   po sloju, proveriti ga jednom po prolazu (jedan sync umesto jednog po sloju), ili iza
   debug prekidača. Semantika provere ostaje ista; mutant koji je uklanja mora da padne.
3. **In-place i maske.** Proveriti `filter_range`, `inf_to_nan`, `diagonal().copy_()` u
   `new_rg` / `new_dg` pod funkcionalizacijom; ako treba, batched verzije sa
   `torch.where` u novom modulu (sekvencijalni put i `precise_transformer.py` ostaju
   netaknuti kao referenca).
4. **Koreni i granica.** Realni (Viète) koren, `acos`/`cos`/`clamp`, `nan_to_num`: proveriti
   da kompajliran D nije manji od eager D više od G1 granice (harness, mpmath), jer fuzija
   menja zaokruživanja baš u delu koji nosi zdravost.
5. **Granica kompilacije.** Prvo samo `precise_sigmoid`, zatim ceo `_forward_chunk`
   (affine + sigmoid slojevi), ako prvi korak prođe.

## Merenje

- eager naspram kompajliranog `precise_sigmoid` i `_forward_chunk`: vreme po sloju i po
  prolazu, broj CUDA kernela (profiler), fuzije (`TORCH_LOGS="fusion,output_code"`);
- 4 mreže, `n_splits` 1, 8, 32, 64, eps 0 i 8;
- hladan i topao start (zaseban `TORCHINDUCTOR_CACHE_DIR`) i break-even kao u
  `COLAB_V3_COMPILED.ipynb`.

## Korektnost (pre bilo kakvog broja brzine)

- `test_batched_splits.py` sa novim režimom: svaki per-split bound naspram sekvencijalnog
  ≤ 1e-10, autograd zdravost, provera da je kompajlirani put stvarno korišćen (broj
  grafova, bez fallback-a);
- seed-ovani boundovi kompajliranog i eager A2 puta: tolerancija zapisana unapred;
- mutation check: šum ponovo unutar grafa, uklonjen assert, pogrešan redosled šuma.

## Posle TorchInductor-a

TVM i XLA na istoj funkciji i istim oblicima (Sašin predlog: `big`, `n_splits = 32`):
vreme, broj kernela i šta je fuzionisano. Ako Inductor već fuzioniše skoro sve, upisati
to kao rezultat; ako ne, identifikovati šta ostaje nefuzionisano.

## Urađeno (28.10.2026)

- Koraci 1 i 2 iz liste: šum na A je tensor ulaz `precise_sigmoid(z, noise)`, vuče ga
  `_forward_chunk` istim redosledom; `A != 0` se proverava jednom po chunk-u
  (`8e3d0df`). Testovi celog stanja NumPy generatora (sa seed-om, sa chunk-ovima, bez
  seed-a), istih bitova za isti seed i pada na A = 0, u sva tri batched režima, plus
  mutanti M27-M29 (`a57b560`). Na neuspešnom prolazu chunk staje posle poslednjeg sloja,
  pa stanje generatora tada nije isto kao ranije; za uspešne prolaze semantika je ista.
- Sledeći compiler eksperiment se fiksira na `a57b560`.

## Korak 4, lokalna analiza (28.10.2026, CPU, torch 2.12.1, `a57b560`)

Skripta i izlaz: `research_notes/compile_precise_sigmoid/analyze.py`, `analysis_cpu.json`.
Eager A2 unutra (`USE_CLOSED_FORM_FIT=True`, `USE_COMPILED_FIT=False`,
`USE_AUTO_PLANE_FIT=False`), eksplicitan šum, S = 1.

| ulaz | grafova | prekida | operacija | fullgraph | Inductor kernela (CPU) | max rel. razlika izlaza |
|---|---:|---:|---:|---|---:|---:|
| 3layer sloj 1 (100 boxova, G0 2) | 1 | 0 | 526 | ok | 17 | 4,5e-16 |
| 3layer sloj 2 (100, G0 202) | 1 | 0 | 526 | ok | 22 | 2,1e-15 |
| big sloj 1 (1024, G0 2) | 1 | 0 | 526 | ok | 17 | 8,6e-16 |
| big sloj 2 (1024, G0 2050) | 1 | 0 | 526 | ok | 22 | 2,4e-15 |

- Kompajliran poziv ne dira stanje NumPy generatora; zastavica A != 0 ista kao eager.
- Izlazi nisu bit-identični (fuzija), razlika do 2,4e-15 relativno.
- Ceo batched prolaz sa kompajliranim `precise_sigmoid`: konačne granice do 1,9e-16 od
  eager puta (3layer S=1 bit-identično), stanje generatora posle prolaza isto kao eager.
- `BatchedDualZonotope` nije prepreka za `fullgraph`: tensor-only omotač ni refaktor klase
  nisu potrebni.
- Kompilacija na CPU-u 2,3-5,4 s po obliku (svaki slučaj posle `reset`); slojevi imaju
  različit G0, pa na A100 treba izmeriti koliko rekompilacija ima ceo prolaz.

Sledeće: A100 sveska sa vremenima (eager naspram kompajliranog `precise_sigmoid` u celom
batched prolazu), brojem CUDA kernela, `TORCH_LOGS` za fuzije, rekompilacijama, hladnim i
toplim startom, break-even, i proverama korektnosti (per-split naspram sekvencijalnog,
seed protokol).

## A100 sveska (28.10.2026, pripremljena i pre-registrovana, nije pokrenuta)

- `COLAB_COMPILED_PRECISE_SIGMOID.ipynb` (koren workspace-a), generisana iz
  `research_notes/compile_precise_sigmoid/build_notebook.py`; harness
  `research_notes/compile_precise_sigmoid/colab_harness.py` ide u svesku preko `%%writefile`.
  Pasado pinovan na `a57b560`, bez izmene koda biblioteke.
- Pre-registracija je u zaglavlju harness-a; sveska proverava sha256
  `df0e88f5f43486d061c990aa6b6a0e70ad4b14e0edf821560e269f1742fdd21c` (normalizovan završni
  novi red) i upisuje ga u `colab_environment.json`. Commitovano u workspace pre A100 runa:
  `91e02e4` (samo harness, builder i sveska).
- Posle korisnikovog pregleda: C1 obuhvata i provere steady procesa i start procesa (bound
  naspram sekvencijalnog, kompajlirani regioni, `lstsq`, fallback, bar jedan graf); pad daje
  ishod `stop` u `criteria.json`. Pre toga je `summarize` ignorisao `steady_checks.csv`.
- Dopune specifikacije, korisnik ih je potvrdio pre runa: G5 tolerancija 1e-12
  normwise; "prihvatljiv break-even" = najviše 480 prolaza (30 slika × 16 eps) naspram
  najboljeg puta bez kompilacije, male mreže, S = 1; ishodi opt-in / auto candidate
  (topli keš) / default candidate (i hladni keš).
- Lokalno (CPU, `--smoke`: 3layer, S = 1 i 2): ceo tok sveske simuliran, gate 52/52, zip
  potpun. Pet mutanata uhvaćeno: tolerancija 0 (G5), ugnježdena kompilacija (G2;
  Dynamo je ne odbija sam), "kompajliran" režim koji je eager (G2), isti u steady procesu
  (ishod `stop`), compiled fit na fallback-u u start procesu (ishod `stop`). Rekompilacije po
  (mreža, S): 2 grafa (G0 2 → 202, G0 postaje dinamičan); deljeni callable: S postaje
  dinamičan na drugom S.
- Dijagnostika (nije gate): kompajliran D manji od eager D do 5e-15 apsolutno
  (1,3e-11 relativno, D je mali).

## v1 A100 run: zaustavljen gate-om (28.10.2026)

- A100-80GB, chunk 64, harness `df0e88f5…`: 868/876 provera. G4 pao 4× (prvi prolaz
  konfiguracije: sloj 1 radi na statičkom grafu 0/0, kasnije na dinamičkom 0/1; lokalno
  potvrđeno preko profiler oznaka), G5 pao 4× (samo `dg`, S = 64, eps 8, 1,06-1,72e-12).
  G1/G2/G3/G6/G7/G8 prolaze; per-split naspram eager A2 ≤ 9,8e-15, finalni naspram
  sekvencijalnog ≤ 2,1e-13. Timing nije rađen.
- Arhivirano nepromenjeno: Pasado `results/compiled_sigmoid_cuda_2026-10-28_gate_stop/`,
  commit `1ac62f9` (samo rezultati; nije pushovano).
- Ispravka mog izveštaja: najveći apsolutni manjak D je 1,71e-13 (gde je D ≈ 290, tj.
  1-2 ulp-a), ne 2e-14.

## v2 (pre-registrovan, workspace `e18278c`, harness sha256 `9e8535ff…`)

- G4: `torch._dynamo.mark_dynamic(z.rg, 1)` i `(z.dg, 1)` pre svakog kompajliranog poziva;
  G2 dodaje "tačno jedan graf po (mreža, S)". Lokalno (`check_mark_dynamic.py/.json`, CPU,
  sve 4 mreže, S = 1 i 8): 1 graf, prolazi 1-3 bit-identični, izlazi ≤ 9,3e-14 od eager-a,
  dinamički naspram statičkog grafa 0,78-1,15 (jedan izuzetak 2,74 bio je šum: naizmenično
  0,99-1,01).
- G5: `rc`, `rg`, `dc`, `dg` generatori i nule ≤ 1e-12 normwise; D po G1 ugovoru
  (mpmath, manjak/W ≤ 10 × max(1e-12, eager)) na uzorku pri S = 64, eps 8 (32 po |ΔD|,
  32 po |ΔD|/W, 64 slučajna, seed [20261028, mreža, slika, sloj]; do 4992 boksa).
  (A, B, C) i boks iz dijagnostičkog grafa (isti `precise_sigmoid` pod drugim imenom, kuka
  na `pt.check_corners_tensor`), koji mora biti bit-identičan proizvodnom; D iz
  proizvodnog izlaza.
- Lokalno: smoke 56/56, simulacija sveske prolazi, 8 mutanata uhvaćeno (tolerancija 0,
  ugnježdeno, eager "kompajliran", bez mark_dynamic, D umanjen 1e-6, eager dijagnostički
  graf, steady eager, start fallback).
- Rizik za v2 (nije menjan): `rg` je u v1 dostigao 5,8e-13 (58% praga) na S = 64, eps 8;
  blok `diag(mu2)` u `rg` je iste prirode kao D (greška relaksacije sa kancelacijom).

## Push pre v2 run-a (28.10.2026)

- Pasado: `1ac62f9` pushovan na `fork/gpu-hybrid` (skeniranje čisto).
- Workspace: po korisnikovom zahtevu iz tri nepushovana commita izbačeno sve što pominje
  "claude" (izmene u `claude-memory/`, red u README-u, tri reda u `UCENJE.md`; premeštanje
  `MAC_SETUP.md` izostavljeno jer fajl ima takvu sekciju; poruka prvog commita skraćena).
  Sadržaj pre-registracije je bajt po bajt isti (isti blobovi, SHA harness-a `9e8535ff…`),
  autor i datumi sačuvani. Novi hash-evi, pushovano na `origin/main` (UIUC-workspace,
  privatan):
  - `7c40772` → `a7e2a58` (reorganizacija)
  - `91e02e4` → `788589d` (pre-registracija v1)
  - `e18278c` → `7c9a8ef` (pre-registracija v2)
- Stari lanac je na lokalnoj grani `backup/main-e18278c` (ne pushuje se). Radno stablo
  netaknuto: izbačeni delovi su i dalje lokalno, kao necommitovane izmene.
- `91e02e4` se i dalje pominje u zaglavlju v2 harness-a (ne menja se zbog SHA) i u
  README-u v1 arhive u Pasado (`1ac62f9`, javno): mapiranje upisati u README v2 rezultata.
- Na remote-u od ranije (`caab335`) ostaju `claude-memory/` i `.claude/settings.local.json`;
  uklanjanje iz istorije bi tražilo force push.

## v2 A100 run: zaustavljen gate-om (29.10.2026)

- A100-80GB, `9e8535ff…`: 892/908. `mark_dynamic` uspeo (G4 96/96, jedan graf u svih 16
  konfiguracija, bez rekompilacija); G5 po blokovima prolazi (`rg` 5,8e-13, generatori
  3,9e-13, nule 0; stara metrika nad celim `dg` bi bila 1,79e-12). G1-G3, G6-G8 prolaze.
- Pao samo D deo (16 provera): dijagnostički graf (dodatni izlazi preko kuke) nije bio
  bit-identičan proizvodnom ni u jednom od 39 poziva, pa mpmath nije dobio nijedan boks.
  Ništa se ne zna o zdravosti D. Harness je beležio samo da/ne.
- Arhivirano: Pasado `results/compiled_sigmoid_v2_cuda_2026-10-29_gate_stop/`, `aff244b`
  (nije pushovano).

## v3 (pre-registrovan, workspace `85de132`, harness `898980d9…`)

- Korisnik izabrao opciju A: formalni kandidat je `precise_sigmoid_instrumented` (kopija
  `a57b560` red po red, vraća A, B, C, lx, ux, ly, uy, D kao tenzorske izlaze kompajlirane
  funkcije); isti graf ide kroz gate, D ugovor i merenje. Gate dodaje: kopija eager
  bit-identična originalu; D koji kandidat vraća = D koji propagira. Kompajliran graf bez
  dijagnostike je ablacija (razlika izlaza, brzina, kerneli; `formal=False`, van ishoda).
- Lokalno: smoke 63/63, simulacija sveske prolazi (zip `compiled_sigmoid_v3_cuda.zip`),
  10 mutanata se ponaša kako treba (7 obara gate, steady i start daju `stop`, pokvarena
  ablacija ne menja ishod).
- Granica: podmetanje samo eager A/B/C (bez D) ne bi uhvatila nijedna provera na nivou
  izlaza (razlika je na nivou zaokruživanja); štiti ga struktura koda (dijagnostika iz
  istog poziva kandidata).
