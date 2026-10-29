# Handoff: Pasado GPU hybrid, 10.10.2026.

Ovo je privatna radna beleška za nastavak u novom razgovoru. Fajl je izvan
Pasado repoa i ne treba ga dodavati ni u jedan Pasado commit ili PR.

## 1. Lokacije i Git stanje

- Workspace: `/Users/dimitrije/prakse/UIUC`
- Pasado repo: `/Users/dimitrije/prakse/UIUC/Pasado`
- Aktivna grana: `gpu-hybrid`
- Fork remote: `https://github.com/dimitrijepesic/Pasado`
- Upstream remote: `https://github.com/uiuc-arc/Pasado`
- Poslednji pushovani commit: `125ddf210c9bbbe7f6370751853b7eafa198f862`
- Grana je čista i lokalni `HEAD` je jednak `fork/gpu-hybrid`.

Poslednji relevantni commitovi:

- `f51f977` - `fix: harden GPU verification paths`
- `c2a2c65` - `docs: normalize verification artifacts`
- `d765ab3` - `fix: use dtype-specific CUDA tolerances`
- `125ddf2` - `docs: archive full A100 verification results`

`gpu-hybrid` je istraživačka/arhivska grana sa mnogo starih logova, profila i
beležaka. Ne otvarati upstream PR direktno iz nje. Za review treba kasnije
napraviti čistu granu od `origin/main` i preneti samo proizvodni kod i relevantne
testove.

Privatni Colab notebook:

`/Users/dimitrije/prakse/UIUC/COLAB_GPU_HYBRID_VERIFICATION.ipynb`

## 2. Šta je implementirano

### CPU optimizacije

- Vektorizovani corner i nonlinear-boundary checkovi.
- Batched least-squares regresija.
- Batched sampling grid.
- Seedovana CPU numerika ostala je bit-identična kroz proverene stare i nove
  commit-e za 3layer, 4layer, 5layer i big.
- Ranija kontrolisana CPU merenja:
  - 3/4/5layer približno 3,2-3,6x;
  - big približno 1,65x.

### Device/dtype put

- Tenzori u dualnim i zonotopskim strukturama nasleđuju uređaj i dtype.
- Popravljeni su `traceify`, `Zonotope.expand`, dual/hyper-dual intervali,
  legacy corner/boundary putanje i regresione matrice.
- MPS koristi float32.
- CPU i CUDA podrazumevano koriste float64.
- Sve selector kombinacije prolaze na MPS-u.

### Hibrid CPU/GPU

Glavni hibrid je u:

`Pasado/forward_mode_tensorized_src/precise_transformer.py`

Funkcija:

`lin_reg_tensor_batched`

Tok:

1. ulazni batched grid i targeti su na originalnom uređaju;
2. ako uređaj nije CPU, prebacuju se na CPU;
3. `torch.linalg.lstsq(..., driver="gelsd")` se izvršava na CPU-u;
4. rezultat se vraća na originalni uređaj.

Veliki matmul u `SimpleZono.AffineZonotope` ostaje na GPU-u.

CUDA `gels` nije zamena za CPU `gelsd`. Na rank-deficient sistemima ranije je
izmerena razlika od 15 do 34 reda veličine. Zato je minimum-norm CPU fallback
namerno zadržan.

### Realni kubni rešavač

Originalni tensorized Pasado koristi complex64 Cardano put i strogi
`torch.isreal`. Na CUDA-i mali imaginarni ostaci dovode do odbacivanja
matematički realnih korenova.

Dodat je realni Viète solver:

- `inverse_poly_real_tensor`
- selector `PASADO_REAL_CUBIC`

Realni solver uklanja device-dependent filtriranje korenova. Ne tvrditi da su
originalni CPU rezultati automatski pogrešni; precizna tvrdnja je da originalni
kompleksni put nije numerički prenosiv između CPU-a i CUDA-e.

## 3. Testovi i mutation provera

Glavni testovi:

1. `test_batched_grid.py`
2. `test_selector_paths.py`
3. `test_check_corners_tensor.py`
4. `test_check_nonlinear_boundary_tensor.py`
5. `test_compute_max_error_tensor.py`
6. `test_batched_lin_reg.py`
7. `test_benchmark_variants.py`
8. `test_get_lipschitz_cli.py`
9. `correctness_e2e.py`

Svih devet prolazi.

Mutation harness:

`Pasado/experiments/mutation_check.py`

Radi u izolovanoj privremenoj kopiji. Prvo zahteva zelen baseline, zatim
pojedinačno ubacuje deset realističnih kvarova i posle svakog ponovo pokreće
svih devet testova. Originalni fajl se vraća u `finally` bloku. Skripta vraća
nonzero ako baseline nije zelen, tekst mutacije nije pronađen tačno jednom ili
bilo koji mutant preživi.

Mutanti:

- M1: pogrešan korak u sampling gridu;
- M2: propušten negativni znak greške;
- M3: propušten ugao;
- M4: `ly` korišćen umesto `uy`;
- M5: filtrirani koren postaje 0 umesto `-inf`;
- M6: gubitak dtype promocije korena;
- M7: izbačen `uy` boundary maksimum;
- M8: obrnut implementation selector;
- M9: pogrešan redosled kolona regresione matrice;
- M10: `gelsy` umesto `gelsd`.

Svih deset mutanata je uhvaćeno.

## 4. Finalna A100 korektnost

Hardver:

`NVIDIA A100-SXM4-80GB`

Raw rezultati i logovi:

`Pasado/results/gpu_hybrid_verification_2026-10-08/`

Quick:

- 5 slika;
- epsilon indeksi 0, 8 i 15;
- 28/28 PASS.

Full:

- 3layer, 4layer, 5layer i big;
- 30 tačno klasifikovanih slika;
- svih 16 epsilona;
- 28/28 PASS;
- log završava sa `ALL PASS`.

Full maksimumi:

| Provera | Maksimalna relativna razlika |
|---|---:|
| real float64 precise CPU/CUDA | `1,786e-15` |
| real float64 zonotope CPU/CUDA | `4,792e-16` |
| real float64 interval CPU/CUDA | `5,517e-16` |
| real float32 precise CPU/CUDA | `8,318e-7` |
| ponovljen CUDA run | `0` |
| full naspram precise-only CUDA | `0` |

Pragovi:

- float64: `1e-9`;
- float32: `1e-5`.

Prva verzija protokola pogrešno je koristila `1e-9` i za float32. Taj neuspeh
je sačuvan kao nalaz, protokol je ispravljen na dtype-specifičan prag, a quick
i full matrice su ponovljene.

Kompleksni negativni control, maksimalna relativna razlika finalnog prosečnog
`lc_precise`:

- 3layer: 14,39%;
- 4layer: 33,42%;
- 5layer: 48,72%;
- big: 63,78%.

PASS za ove redove znači da je očekivana divergencija otkrivena. Ne znači da
je kompleksni solver prihvatljiv.

## 5. Važne ograde

Još nije dokazano ubrzanje finalnog hibrida. A100 matrica je correctness
eksperiment, ne benchmark.

Full matrica poredi proseke preko 30 slika po epsilonu. Nema per-image
maksimalnu CPU/GPU razliku.

Nije izmereno kako se kompleksna CPU/GPU divergencija razvija posle svakog
pojedinačnog sigmoidnog sloja.

Nije izmeren udeo finalnog A100 vremena koji odlazi na:

- GPU -> CPU transfer;
- CPU `gelsd`;
- CPU -> GPU transfer;
- GPU čekanje izazvano fallbackom.

Ne implementirati sopstveni GPU `gelsd` pre ovog merenja. `gelsd` je robustan
SVD/divide-and-conquer algoritam, ne jednostavan kernel. Ako fallback nije
značajan deo vremena, čak ni savršen GPU solver ne može dati važno end-to-end
ubrzanje.

## 6. Sledeći task A: merenje CPU fallbacka

### Cilj

Odgovoriti na pitanje:

> Koliki procenat finalnog A100 precise vremena odlazi na CPU fallback i gde
> GPU trenutno čeka?

### Šta se meri

Za svaki poziv `lin_reg_tensor_batched`, odvojeno:

1. GPU -> CPU transfer `x_batch`;
2. GPU -> CPU transfer `zs_batch`;
3. CPU konstrukcija design matrice ako je relevantna;
4. CPU `torch.linalg.lstsq(..., driver="gelsd")`;
5. CPU -> GPU transfer rešenja;
6. ukupno vreme fallbacka.

Za svaki precise forward:

1. ukupno wall-clock vreme;
2. vreme svakog sigmoidnog sloja;
3. zbir fallback vremena;
4. fallback procenat.

Formula:

```text
fallback_share_precise =
    (GPU->CPU + gelsd + CPU->GPU) / forward_zono_precise_time * 100
```

Poželjno prijaviti i procenat ukupnog `get_lipschitz` runa, ali primarna
metrika je procenat precise forwarda.

### Korektno CUDA merenje

- Uraditi warm-up pre merenja.
- Koristiti `torch.cuda.synchronize()` pre početka i posle kraja segmenta.
- Za CPU segmente koristiti `time.perf_counter()`.
- Ne koristiti cProfile vremena kao finalni benchmark.
- Više ponavljanja, prijaviti median, min i max.
- Varijante pokretati naizmenično da sistemski drift ne favorizuje jednu.
- Isti seed, slike i epsiloni za svaki par.

Napomena: `.cpu()` već nameće čekanje na podatke, ali eksplicitne
sinhronizacije čine granice merenja jasnim. H2D transfer može biti asinhron,
zato treba sinhronizovati pre uzimanja završnog timestamp-a.

### Prvi pilot

- mreža: `big`;
- uređaj: A100;
- dtype: float64;
- real-cubic: uključen;
- `--precise-only`;
- 1-3 slike;
- epsilon indeksi 0, 8 i 15;
- seed 7;
- najmanje 5 warm-up prolaza;
- najmanje 10 merenih ponavljanja.

### Predložena implementacija

Ne menjati production kod pre pilota. Napraviti novu eksperimentalnu skriptu,
na primer:

`Pasado/experiments/measure_hybrid_fallback.py`

Opcije:

- monkeypatch/wrapper oko `lin_reg_tensor_batched`;
- ili mali opcioni timing hook koji je aktivan samo u eksperimentalnoj skripti.

Skripta treba da čuva CSV, ne samo da štampa.

Predložene kolone:

```text
network
image_id
epsilon_index
repeat
layer
n
m
dtype
d2h_x_ms
d2h_z_ms
design_ms
gelsd_ms
h2d_ms
fallback_total_ms
layer_total_ms
precise_total_ms
fallback_share_layer
fallback_share_precise
```

### Odluka posle merenja

- fallback ispod 5%: zadržati CPU `gelsd`; GPU solver verovatno nije vredan;
- fallback 5-15%: prvo probati pipeline nezavisnih poslova;
- fallback iznad 15%: prototip GPU SVD/minimum-norm solvera;
- sopstveni kernel tek ako postojeće CUDA SVD/cuSOLVER opcije ne daju potrebnu
  semantiku ili brzinu.

### Paralelizam

Za isti sloj iste slike sledeći GPU korak zavisi od `gelsd` rezultata, pa nema
lakog preklapanja.

Moguće nezavisne jedinice za pipeline:

- različite slike;
- različiti epsiloni;
- različiti splitovi;
- različiti BaB podproblemi.

## 7. Sledeći task B: rast CPU/GPU razlike po slojevima

### Cilj

Odgovoriti na pitanje:

> Gde kompleksna CPU/GPU divergencija prvi put nastaje i kako se akumulira kroz
> sigmoidne slojeve?

### Trenutna metrika

Finalna A100 matrica koristi:

```text
relative_difference = abs(CPU - CUDA) / max(abs(CPU), 1e-300)
```

Prijavljuje najveću razliku preko epsilon vrednosti finalnog prosečnog
`lc_precise` rezultata.

To nije layerwise metrika i ne daje per-image maksimum.

### Šta treba instrumentirati

Posle svakog `PreciseSigmoidDualZonotope` sloja sačuvati:

- layer indeks;
- real centers;
- real generators;
- dual centers;
- dual generators;
- lower bound i upper bound;
- širinu intervala `ub - lb`;
- privremeni bound `max(abs(dual_lb), abs(dual_ub))`;
- broj ulaznih cubic kandidata;
- broj kandidata sa nenultim imaginarnim delom;
- broj kandidata odbačenih posle `torch.isreal`;
- broj korenova koji ostanu posle `[lx, ux]` filtera.

Porediti CPU i CUDA sa:

- max absolute difference;
- max relative difference nad konačnim elementima;
- broj NaN/Inf pattern mismatch-eva;
- razliku širine intervala;
- razliku privremenog bounda.

### Dve obavezne varijante

1. kompleksni CPU naspram kompleksnog CUDA;
2. realni CPU naspram realnog CUDA kao kontrola.

Očekivanje:

- kompleksni put: razlika nastaje u root filtriranju i akumulira se;
- realni put: ostaje blizu float64 mašinske preciznosti kroz sve slojeve.

### Prvi pilot

- mreže: 3layer, 4layer i 5layer;
- slika: prva tačno klasifikovana;
- epsilon indeks: 0, najveći epsilon;
- seed: 7;
- dtype: float64;
- CPU i A100.

Big dodati posle pilota jer menja i širinu, ne samo dubinu.

### Predložena implementacija

Proširiti:

`Pasado/experiments/diagnose_device_diff.py`

ili napraviti novu skriptu:

`Pasado/experiments/measure_layerwise_device_diff.py`

Nova skripta je verovatno čistija jer postojeća dijagnostika koristi samo
3layer i zaustavlja se na prvom različitom boundary pozivu.

Predložene CSV kolone:

```text
network
image_id
epsilon_index
solver
layer
tensor
max_abs_diff
max_rel_diff
finite_pattern_mismatches
cpu_width_max
cuda_width_max
width_rel_diff
cpu_bound
cuda_bound
bound_rel_diff
nonreal_roots_cpu
nonreal_roots_cuda
roots_kept_cpu
roots_kept_cuda
```

### Važna interpretacija

Ako se validan kandidat odbaci, lokalni maksimum aproksimacione greške može
samo ostati isti ili se smanjiti. Neposredni error generator i interval zato
postaju isti ili uži, što može biti unsound under-approximation.

Posle narednih nelinearnih slojeva konačan odnos ne mora biti jednostavno
monoton, zato treba meriti svaki sloj umesto zaključivati samo iz finalnog
bounda.

### Proširenje posle pilota

Ako pilot pokaže smislen trend:

- svih 30 slika;
- svih 16 epsilona;
- sve četiri mreže;
- čuvati per-image rezultate, ne samo proseke;
- prijaviti median, p95 i maksimum po sloju.

## 8. Dodatni task koji je Saša tražio: split sweep

Povećati `--n-splits` i izmeriti odnos preciznosti i vremena.

Prvi sweep:

```text
n_splits = 1, 2, 4, 8, 16
```

Držati fiksno:

- mrežu;
- slike;
- epsilon indekse;
- seed;
- uređaj;
- dtype;
- real-cubic put.

Meriti:

- `lc_precise`;
- prosečno precise vreme;
- ukupno wall-clock vreme;
- relativno poboljšanje bounda u odnosu na `n_splits=1`;
- multiplicativni trošak vremena.

Više splitova deli `[0, epsilon]` na manje podintervale i očekivano daje uži
ili isti bound, uz veće vreme. To treba proveriti, ne pretpostaviti.

Pilot:

- `3layer` i `big`;
- 3 slike;
- epsilon indeksi 0, 8 i 15;
- CPU i A100;
- seed 7.

## 8b. Task D (istraživački): floating-point soundness

Izvor ideje: Saša, 11.10.2026. Status: ideja, još ništa nije implementirano ni
pročitano. Prioritet posle Taska A i B.

### Problem

Pasado računa intervale i zonotope u floating-point aritmetici sa zaokruživanjem
na najbliže i ne proširuje granice za grešku zaokruživanja. Dobijeni
`lc_precise` je zato sound samo u odnosu na idealnu realnu aritmetiku
operacija koje Pasado modeluje, ne u odnosu na stvarno izvršavanje.

### Predloženi pristup

- Miné, A. (2004). Relational abstract domains for the detection of
  floating-point run-time errors. ESOP, 3-17.
  https://doi.org/10.1007/978-3-540-24725-8_3
- DeepJ (POPL'22) je ovo koristio u evaluaciji:
  https://misailo.cs.illinois.edu/papers/DeepJ-popl22.pdf

Obe reference tek treba pročitati. Dole navedeno je moje razumevanje iz
pamćenja, ne iz čitanja radova, i treba ga proveriti pre implementacije.

### Kandidati za varijante (za proveru, ne odluku)

1. Bez korekcije (trenutno stanje), referenca.
2. Spoljno zaokruživanje: posle svake operacije `lb = nextafter(lb, -inf)` i
   `ub = nextafter(ub, +inf)`. Važi samo za korektno zaokružene operacije
   (+, -, *, /, sqrt). Za sigmoid/exp biblioteke nisu korektno zaokružene,
   pa treba poznata gornja granica greške u ulp-ovima, posebno za CPU i CUDA.
3. Minéov pristup: greška zaokruživanja svake operacije ulazi kao dodatni
   član (relativna greška `u * |x|` plus apsolutni deo za denormale), uključena
   u afinu formu kao nov šum simbol ili pripojena postojećem članu.
4. Standardna granica za sumiranje (tipa `gamma_n * (|W| |x|)`), koja ne zavisi
   od redosleda sabiranja, što je bitno za matmul na GPU-u.

### Otvorena pitanja koja treba proveriti

- Matmul na A100: TF32 mora biti isključen, a FMA i redosled sabiranja
  menjaju zaokruživanje. Granica mora da pokrije sve te slučajeve.
- Rast broja generatora ako se svaka operacija doda kao novi šum simbol.
  Treba meriti i varijantu u kojoj se greška sažima u jedan član po neuronu.
- Regresija (A, B, C) i izbor korena kubnog solvera mogu ostati heuristike,
  ali računanje maksimalne greške nad kutijom (corners, boundary) mora samo
  biti sound, uključujući zaokruživanje.
- Sound pristup zahteva i da nijedan validan kubni koren ne bude odbačen
  (vidi interpretaciju u sekciji 7), što je nezavisno od floating-pointa.

### Pitanja za merenje

1. Koliko dodavanje ovih operacija usporava izvršavanje (po varijanti, na CPU-u
   i A100)?
2. Da li kompajlerske optimizacije (`torch.compile`, fuzija operacija) smanjuju
   taj trošak? Postoji `Pasado/experiments/colab_dynamo_explain.py` sa
   graph-break analizom `sigmoid_prime_product_tensor`.
3. Koliko se bound `lc_precise` širi u odnosu na varijantu bez korekcije
   (po mreži i epsilonu)?
4. Sweep po dtype: greška zaokruživanja raste sa jediničnom greškom `u`, pa
   float32 (MPS) i float64 treba porediti.

### Istraživačko pitanje (potencijalno novi rezultat)

Da li se može tvrditi da je interval greške zaokruživanja minimalan u oba
smisla: po preciznosti (najuži bound koji ostaje sound) i po vremenu izvršavanja
(najjeftinija korekcija)? Verovatno se radi o Pareto frontu između varijanti
gore, ne o jednoj najboljoj varijanti. Precizna formulacija "minimalnosti"
mora da se dogovori sa Sašom, uključujući šta je referentni bound (npr.
izračunat u višoj preciznosti ili sa MPFR intervalima na maloj mreži).

### Ograde

- Ne tvrditi soundness ni za jednu varijantu dok se ne proveri na malom
  primeru protiv referentne višestruke preciznosti.
- Ne mešati ovo sa production putem na `gpu-hybrid` dok ne postoji pilot.

## 9. Redosled sledeće sesije

1. Proveriti da je `Pasado/gpu-hybrid` čist i remote jednak lokalnom HEAD-u.
2. Ne menjati production kod.
3. Implementirati pilot `measure_hybrid_fallback.py`.
4. Pokrenuti mali lokalni/MPS smoke test.
5. Pokrenuti A100 pilot i pregledati breakdown pre šireg sweep-a.
6. Implementirati pilot `measure_layerwise_device_diff.py`.
7. Pokrenuti kompleksnu i realnu kontrolu na 3/4/5layer.
8. Tek nakon pregleda rezultata proširiti na 30 slika i 16 epsilona.
9. Split sweep raditi posle prva dva pilota ili paralelno ako je A100 sesija
   dostupna.
10. Task D (floating-point soundness): prvo čitanje Minéa (2004) i DeepJ
    evaluacije, pa mali prototip na intervalnom delu, tek onda zonotopi.
    Koristi infrastrukturu merenja iz Taska A.

## 10. Šta ne treba raditi

- Ne pisati sopstveni `gelsd` kernel pre fallback merenja.
- Ne tvrditi GPU ubrzanje iz correctness notebooka.
- Ne otvarati upstream PR direktno iz `gpu-hybrid`.
- Ne koristiti stare CPU profile kao procenat finalnog A100 fallbacka.
- Ne zaključivati layerwise rast samo iz 3layer/4layer/5layer finalnih brojeva.
- Ne menjati float32 prag nazad na `1e-9`.
- Ne spajati timing instrumentation u production put pre nego što pilot bude
  pregledan.
- Ne pushovati ovaj privatni handoff fajl.

## 11. Kratak status za novi razgovor

Korektnost hibridnog CPU/GPU Pasado FFNN puta je završena i arhivirana.
Real-cubic float64 CPU/A100 slaganje je oko `1,8e-15`; CUDA je deterministična;
full i precise-only se poklapaju. Stari kompleksni solver divergira 14-64%.

Sledeća dva glavna eksperimenta su:

1. izmeriti kompletan CPU fallback i njegov procenat A100 precise vremena;
2. izmeriti nastanak i rast kompleksne CPU/GPU divergencije po sigmoidnim
   slojevima.

Treći zahtev je split precision/time sweep.
