# Taskovi: hibrid CPU(gelsd)/GPU port precise puta za `big`

Napisano 19.9.2026. Stanje na početku: `Pasado/` na `sprint-optimizations`
(čisto, poslednji commit `515c864`), grana `gpu-hybrid` još ne postoji, `modal`
nije instaliran u `.venv` prema početnoj belešci. Rok za MLSys je 30.10.
Za potvrđenih **2–3 h dnevno**, planiraj približno **12 radnih sesija**
(24–36 h), uz još 2–3 sesije rezerve za CUDA/numerički debugging. Ranijih
3–4 dana podrazumevalo je celodnevni rad i nije dobar dnevni raspored.

[Dnevni raspored](DAILY_PROGRESS_PLAN.md) razlaže T0–T7 i uvodi izveštaje
posle svaka 2–3 dana. Ovaj fajl ostaje tehnička lista taskova; dnevni plan
je kalendar iste liste. [Audit 19.9.](notes/PROJECT_AUDIT_2026-09-19.md)
beleži provereno početno stanje i rezultate testova.

Način rada: ti kucaš, ja objašnjavam i pregledam. Kad zapneš, traži prvo
nagoveštaj, pa tek onda rešenje. Posle svakog taska: relevantne provere prolaze,
poznati izuzeci su jasno zabeleženi → mali smislen commit → pregled diff-a.
Ako task traje više dana, izveštaj ne čeka poslednji commit.

Redosled je namerno drugačiji od starog plana: prvo se sređuje samo **vrući
put** (desetak mesta), pa se što pre ide na GPU po prvi broj. Mehaničko
čišćenje ostalih ~45 factory poziva je na kraju (T8), jer ne utiče na rezultat.

---

## T0 — Grana i zaključana CPU referenca (~1 h)

- [x] U repou `Pasado/`, `git switch -c gpu-hybrid` sa `sprint-optimizations`.
- [x] Pre izmene zabeleži stvarno stanje: `test_batched_grid.py`,
      `test_selector_paths.py` (iz `forward_mode_tensorized_src/`),
      `experiments/test_batched_lin_reg.py`, i dimni test iz `README.md`.
      Audit je već potvrdio: na Mac-u grid test ispisuje FAIL za STEPS=7/9
      (do 8,882e-16) uz exit 0, dok produkcioni STEPS=5 prolazi. Ispravi
      prijavljivanje neuspeha i odvoji poznatu platformsku bit-egzaktnost od
      produkcione regresije; ne menjaj formulu samo da test pozeleni.
- [x] Osnovna varijanta za ceo port je **`vec_batched_grid`**:
      `PASADO_BATCHED_LSTSQ=1 PASADO_BATCHED_GRID=1` (ostala dva flaga su već
      podrazumevano 1, vidi `precise_transformer.py:24-46`). Za merenja
      eksplicitno zapiši sva četiri flaga na 1 i `PASADO_REAL_CUBIC=0`;
      ne oslanjaj se na nasleđeno okruženje. Bez ova dva ide
      stari `get_linspace` (`:64-65`) sa `.item()` po neuronu.
- [x] **Izmeri šum seed-a.** `get_lipschitz.py` nigde ne seed-uje `np.random`,
      a precise put perturbuje `A` sa `np.random.normal(0, 1e-4)`
      (`precise_transformer.py:745-746`). Pusti 3layer dva puta na CPU i
      zapiši koliko se `lc_precise` razlikuje između dva pokretanja. To je
      ilustracija nedeterminističnosti tog uzorka, ne dokazana donja/gornja
      granica greške i ne tolerancija koja se prenosi sa 3layer na big.
      Glavna referenca za port mora biti svež CPU run sa istim seed-om,
      mrežom, image IDs, epsilonima i flagovima kao GPU run.

Gotovo kad: imaš granu, pouzdan status testova, zabeležen poznati arm64
izuzetak, zaključane flagove i zapisanu ilustraciju run-to-run šuma.

Pitanje: zašto zbog ovoga poređenje GPU rezultata sa
`Section_5_4/results/lc_precise_big.pth` ne može da bude „≤1e-10"?

## T1 — Uređaj na ulazu: `get_lipschitz.py` (~2–3 h)

Uređaj treba da živi samo na ulaznoj tački; biblioteka ga izvodi iz tenzora
koje dobije.

- [x] Argument `--device`, podrazumevano `os.environ.get("PASADO_DEVICE", "cpu")`.
- [x] Argument `--num-images` (sada je tvrdo 30 na `:37`); treba za kratke
      GPU iteracije. Ako nije 30, ispiši upozorenje da se prosek ne poredi sa
      sačuvanim vrednostima.
- [x] Eksplicitan seed za NumPy/torch i zapis image IDs/config-a. Za poređenje
      re-seed pre odgovarajućeg forward-a, ne samo jednom pre učitavanja mreže.
- [x] Omogući merenje samo `precise` puta, da prvi GPU rezultat ne zavisi od
      portovanja intervalnog i običnog zonotope baseline-a. Podrazumevani
      postojeći CLI režim može ostati sva tri puta.
- [x] Na uređaj: mreža (`:60-61`), `img_f` (`:167`), `real_l`/`real_u` za
      `DualIntervalTensor` (`:158-159`), i `torch.tensor([1])` na `:174` i
      `:189`.
- [x] `torch.cuda.synchronize()` pre svakog čitanja tajmera kad je uređaj cuda
      (`:170-208`).

Zamke:
- `Duals.py:15,17` pravi `e1_l/e1_u` sa `torch.zeros(real_l.shape)` bez
  uređaja, a `:19` ima `assert real_l.device == e1_l.device`. To će biti prvi
  pad na GPU-u.
- `torch.tensor([1])` ima oblik `[1]`, nije 0-dim skalar, pa mešanje sa CUDA
  tenzorom nije dozvoljeno (0-dim CPU tenzori jesu).

Gotovo kad: `--device cpu` daje isto što i pre izmene (uporedi sa seed-om,
vidi T5), a svi testovi iz T0 i dalje prolaze.

Pitanje: merenje precise vremena na `:185-196` je i bez `synchronize()` već
pošteno. Zašto? (Pogledaj `:191-192`.)

Urađeno 4.10.2026: `--seed` (samo `np.random`, pre svakog preciznog forward-a), `--num-images`
(upozorenje kad nije 30), `--precise-only`, `--device` / `PASADO_DEVICE`, ispis konfiguracije i
`image_ids`, tajmer `now()` sa `cuda.synchronize()`. Pri čuvanju svaki run koji se razlikuje od
podrazumevanog dobija sufiks u imenu fajla. Dokaz: `experiments/test_get_lipschitz_cli.py` i bit-identičan
`lc_precise`/`lc_zonos` na CPU pre i posle izmene (3layer, 3 epsilona, 30 slika, seed 7). Prvi pad na GPU-u
je `Duals.py:15,17` (T2).

## T2 — Device higijena vrućeg puta (~2–3 h)

Samo mesta kroz koja `big` precise stvarno prolazi sa `vec_batched_grid`:

- [x] `SimpleZono.py:19` `traceify` — `torch.eye(leng)`
- [x] `SimpleZono.py:75` `Zonotope.expand` — `torch.zeros((n, cols))`
- [x] `SimpleZono.py:328` `HyperDualIntervalToDualZonotope` — `torch.zeros(...)`
- [x] `Duals.py:15,17` — `DualIntervalTensor.__init__`
- [x] `precise_transformer.py:216,221,232` — konstante u `inverse_poly_tensor`

Uređaj i dtype uzimaj od tenzora koji je već tu (`x.new_zeros(...)`,
`device=x.device, dtype=x.dtype`), ne iz globalne promenljive.

Napomene:
- `zeros_like` na `SimpleZono.py:192,244,304` nasleđuje uređaj; to ne diraj.
- Konstante na `:216-232` su 0-dim, pa verovatno ne bi ni pukle na GPU-u
  (`GPU_READINESS.md` #2 tvrdi da bi; to nije izmereno). PyTorch podržava
  ovaj izuzetak za CPU skalare; ne tvrditi da svaka konstanta pravi skup
  H2D prenos/sync bez profila. Ako ih menjaš, sačuvaj postojeće dtype-ove,
  naročito kompleksni `1j`, i proveri numeriku.
- Put za non-precise zonotope i intervale (`forward_zono`, `forward_interval`)
  ide kroz iste funkcije plus `Duals.py:96`. Ako zapne, za GPU run ih
  privremeno preskoči flagom, ne troši vreme na njih.

Gotovo kad: CPU regresija prolazi prema zaključanom kriterijumu; za delove
koji su bit-egzaktni traži bit-egzaktnost. Da li radi na GPU-u
saznaješ tek u T5; grep nije dokaz.

Opciono, ograniči na 30 min, neprovereno: `--device mps` sa float32 samo kao
lokalni detektor cross-device grešaka. MPS nema fp64 i moguće je da mu fale
complex/linalg operacije; ako zapne na tome, odustani odmah.

Urađeno 4.10.2026: `traceify`, `Zonotope.expand`, `HyperDualIntervalToDualZonotope` i `DualIntervalTensor`
sada nasleđuju uređaj (i tip) od tenzora koji već postoji (`eye(..., device=, dtype=)`, `new_zeros`,
`zeros_like`). Konstante u `inverse_poly_tensor` nisu menjane: to su 0-dim CPU tenzori, a PyTorch ih
dozvoljava pored GPU tenzora (potvrđeno na MPS-u); dodat je komentar zašto. CPU: `lc_precise` i
`lc_zonos` bit-identični (3layer, seed 7, i 5layer prema starom kodu), svi testovi prolaze.
Provera uređaja: `experiments/check_device_plumbing.py --device mps` (float32) prolazi, razlika prema CPU-u
1.5e-7; stari kod pada na `Duals.py:19`. Nije diran `Duals.py:96` (`__truediv__`, nije na FFNN putu) ni
stari per-neuron putevi (T8). Na CUDA-i ovo još nije pokrenuto.
## T3 — Hibrid: lstsq na CPU (~1–2 h)

Jedina funkcija koja se menja: `lin_reg_tensor_batched`
(`precise_transformer.py:199-206`).

- [x] Ako ulaz nije na CPU: prebaci na CPU → `lstsq(driver='gelsd')` → vrati
      rešenje na uređaj ulaza. Ako jeste na CPU: put ostaje doslovno isti
      (bit-identičnost CPU rezultata ne sme da se pomeri).
- [x] Proširi `experiments/test_batched_lin_reg.py`: test sa uređajem kao
      parametrom, koji se preskače kad nema cuda. Obavezno uključi degenerisane
      kutije (`lx==ux`), jer su one razlog zašto `gels` na GPU-u ne dolazi u
      obzir (`GPU_READINESS.md`, Faza 12: razlika 15–34 reda veličine).
- [x] Ništa vezano za dimenzije ne hardkoduj (`[n,25,3]` je samo današnji
      oblik); Saša hoće mreže veće od `big`.

Gotovo kad: postojeći test prolazi nepromenjen, novi test postoji.

Pitanje: gde je bolje uraditi `.cpu()`, pre ili posle pravljenja `xplusone`,
i da li je to uopšte merljivo za `[1024, 25, 3]`?

Urađeno 5.10.2026: `lin_reg_tensor_batched` prebacuje ulaze sa GPU-a na CPU pre pravljenja `xplusone`
(prenos 3 kolone umesto 4), rešava `gelsd` i vraća rešenje na uređaj ulaza; na CPU-u se ništa ne menja
(`lc_precise`/`lc_zonos` bit-identični pre i posle, 3layer i big, seed 7). Odgovor na pitanje iz zadatka:
`.cpu()` ide PRE pravljenja `xplusone`, jer se prenosi 25% manje podataka, a `cat` i `gelsd` rade isto kao na CPU-u.
`experiments/test_batched_lin_reg.py` ima uređaj kao parametar (13 slučajeva na MPS-u identično CPU rešenju;
bez GPU-a SKIP); zasađene greške (nema `.cpu()`, nema `.to(device)`) test hvata. CUDA test: Colab.

## T4 — Modal okruženje (~2–3 h)

- [ ] `pip install modal` u `Pasado/.venv`, pa `modal setup` u terminalu
      (interaktivni login). `!` je prefiks za notebook, nije deo shell komande.
- [ ] `experiments/modal_run.py`: image sa istim torch-om kao lokalno (2.12.1,
      CUDA build ako dostupan/kompatibilan) + torchvision + tqdm + numpy +
      scikit-learn (uvozi ga precise_transformer); zaključaj stvarno testirane
      verzije. Lokalni `Pasado/` bez `.venv`
      (1,1 GB) i `profiles/` (65 MB); `gpu="A100"`, `cpu=8`, timeout.
      `Section_5_4/MNIST_Data` (63 MB) ili pošalji, ili skidaj u image-u.
- [ ] Prvi run: `experiments/colab_gpu_profile.py`. Očekuješ fp64 matmul oko
      red veličine istorijskog A100 merenja iz `GPU_READINESS.md`, ali
      15 TFLOP/s nije uslov prolaza. Ako je sporije, proveri device/build,
      oblik, warm-up i opterećenje; samo vreme ne identifikuje hardver.
      Ova skripta potvrđuje okruženje i mikrokernele, ne ceo novi hibrid.

Budžet: prema [Modal cenovniku](https://modal.com/pricing) proverenom 19.9,
Starter ima $30 mesečnih kredita. Broj sati zavisi i od GPU/CPU/RAM
konfiguracije i drugih runova; ne računati garantovanih 8 besplatnih sati.
Prvo kratki runovi (`--num-images`, `--eps-indices`) i čuvanje logova lokalno.

## T5 — Korektnost na GPU (~3–4 h)

- [x] `experiments/correctness_device.py` po uzoru na
      `experiments/correctness_e2e.py`: isti parovi (slika, epsilon), isti
      re-seed `np.random` i `torch` pre svakog forward-a, jednom `cpu`, jednom
      `cuda`. Izveštaj: maks. relativna razlika konačnog LC-a i granica po
      sloju.
- [x] **Numerički ugovor zapiši pre prvog pokretanja** (predlog: ≤1e-9
      relativno na konačnom LC-u sa seed-om). Ako ne prođe, istražuješ sloj po
      sloj; prag se ne olabavljuje naknadno bez objašnjenja.
- [x] Redosled: 3layer dimni test → `big` na 2–3 para → veći `big` run
      protiv sveže seeded CPU reference sa istog udaljenog okruženja.
      Stari `lc_precise_big.pth` služi kao dodatna istorijska provera,
      ne kao glavni GPU oracle; 3layer šum iz T0 ne određuje big toleranciju.
- [x] Revizija sync-ova: jedan run sa
      `torch.cuda.set_sync_debug_mode("warn")`. Očekivani sync-ovi po sloju:
      `.cpu()` u lstsq-u, `assert bool(...)` na `precise_transformer.py:751`,
      `torch.all(lb == ub)` u `SigmoidZonotope` (`SimpleZono.py:302`), i
      `.item()` na kraju forward-a. Dodatni sync zahteva objašnjenje,
      nije automatski bug. Režim `warn` ne otkriva sve sync-ove; dopuni
      profilerom ako treba i isključi dijagnostiku za finalni tajming.

Gotovo kad: ugovor prolazi na definisanom skupu `big` slučajeva i uočeni
sync-ovi su objašnjeni. Ne tvrditi univerzalnu bit-identičnost ili formalnu
floating-point soundness na osnovu ovog testa.

Urađeno 8.10.2026 (`Pasado/results/gpu_hybrid_verification_2026-10-08/` i `17.10.2026` layerwise):
- Puna A100 matrica (`3layer`, `4layer`, `5layer`, `big`, 30 slika, svih 16 epsilona): `28/28 PASS`.
  U `float64` sa `PASADO_REAL_CUBIC=1`, maks. relativna razlika CPU naspram CUDA je `1.786e-15` (`precise`),
  `4.792e-16` (`zonotope`) i `5.517e-16` (`interval`); u `float32` je `8.318e-7` (prag `1e-5`).
- Layerwise analiza (`measure_layerwise_device_diff.py`, `Pasado/results/gpu_hybrid_layerwise_2026-10-17/`)
  prati divergenciju po svakom sigmoid sloju na CPU i CUDA za kompleksni i realni rešavač.

Preliminarni rezultat 5.10.2026 (Colab Pro, A100 80 GB; 3layer, slika 0, eps 0.2, float64, `lstsq` na CPU):
- Kompleksni (Cardano, `complex64`) rešavač: CUDA `lc` 401.73 naspram CPU 418.04, razlika **3.9%** (float32:
  387.64, 7.3%). Uzrok: na CUDA-i korenovi u dozvoljenom opsegu dobiju imaginarni ostatak do 3.7e-8
  (na CPU-u tačno 0), `torch.isreal` ih proglasi kompleksnim i `bool_to_nan` ih baci. U sloju 0 je 39 od
  100 neurona izgubilo sva tri korena (76 mesta u izlazu sa drugačijom NaN/inf šemom).
- Realni (Viete) rešavač (`PASADO_REAL_CUBIC=1`): CPU i CUDA `lc` = 418.041836, razlika **1.4e-16**; sve
  provere po pozivu (`lstsq`, uglovi, granica) iste.
- Obe CUDA vrednosti iz kompleksnog režima su iznad pravog maksimuma 198.7 (autograd, 2001 uzorak), ali
  odbacivanje korena može da naruši obuhvatanje, pa kompleksni put na GPU-u nije prihvatljiv.
- Odluka koja čeka: realni rešavač na GPU-u (CPU referenca takođe sa njim) ili kubna na CPU-u. Pre toga
  izmeriti pomeraj granica (kompleksni naspram realnog) na 3/4/5layer i `big` i proveriti `lc` >= maksimum iz autograd-a.
- Kriterijum koji važi odsad: `lc` mora biti >= maksimum |dF/dt| izračunat autograd-om.
- Alati: `experiments/check_device_plumbing.py`, `experiments/diagnose_device_diff.py` (opcija `--real-cubic`).

Rezultat 5.10.2026 (Colab Pro, A100 80 GB), preko `get_lipschitz.py --device cuda`, `--precise-only`, seed 7,
`PASADO_REAL_CUBIC=1` na obe strane: 3layer (5 slika, 3 eps) najveća relativna razlika CPU naspram CUDA
**8.4e-16**; big (2 slike, 2 eps) **8.9e-16**. Numerički ugovor (<= 1e-9) je ispunjen sa velikom rezervom na ovom
uzorku; širi skup slučajeva (`correctness_device.py`) tek sledi.
Realni naspram kompleksnog rešavača, CPU, 16 epsilona, 30 slika (`measure_real_cubic_shift.py`): najveća relativna
razlika `lc_precise` 1.5e-6 (3layer), 3.6e-6 (4layer), 8.3e-6 (5layer), 9.8e-8 (big); obe varijante ostaju >= pravog
maksimuma iz autograd-a. Zaključak: realni rešavač pomera granice za najviše ~1e-5 relativno i uklanja problem
sa GPU-om; to je osnova za odluku A (realni rešavač na GPU-u).
Vreme (NIJE benčmark; 2 slike, jedan zagrejan epsilon, nekontrolisan Colab CPU): big 0.514 s po slici na CPU
naspram 0.063 s na A100 (~8x); naspram CPU-a na Mac-u (~0.16 s) ~2.6x. Veličina ubrzanja zavisi od CPU-a.

## T6 — Merenje (~0,5–1 dan)

- [x] Tri konfiguracije, `big`, ista varijanta flagova: (a) CPU-only na istoj
      udaljenoj mašini, (b) hibrid na istoj mašini, (c) CPU-only na Mac-u. Ubrzanje se
      računa samo (a) naspram (b); (c) se prijavljuje odvojeno. (a) i (b)
      izvrši u istoj instanci, sa istim CPU/thread podešavanjima,
      da odvojeni pozivi ne dobiju različite hostove.
- [x] Warm-up pre merenja (prvi CUDA poziv plaća kontekst i cuBLAS handle),
      više ponavljanja, medijana + rasipanje, prijavljen broj CPU niti.
- [x] Razlaganje po sloju: GPU deo, CPU lstsq, prenosi. Uporedi sa Amdahl
      procenom iz novog precise-only CPU profila. Starih 2–3× i ~70% su
      hipoteza iz ranije varijante/profila, ne garantovan plafon. Broj noise simbola
      po sloju zapiši uz vremena; to je ulaz za Fazu 1 iz `PLAN_MLSYS.md`.
- [ ] Rezultat u `Pasado/results/optimization_section_5_4/HYBRID_RESULTS.md`
      (sirovi rezultati i analiza su arhivirani u `Pasado/results/gpu_hybrid_fallback_2026-10-16/`
      i `Pasado/results/gpu_hybrid_split_sweep_2026-10-17/`).

Urađeno 16–19.10.2026 (`Pasado/results/gpu_hybrid_fallback_2026-10-16/` i `gpu_hybrid_split_sweep_2026-10-17/`):
- `measure_hybrid_fallback.py` i `measure_sync_free_hybrid.py` (3 slike × 3 eps, 5 warm-up + 10 merenih ponavljanja, 6 CPU niti):
  na `big` (`float64`), medijana `precise` prolaza je `532.31 ms` (CPU) naspram `63.34 ms` (CUDA hibrid baseline) i `58.62 ms`
  (bez suvišnih host-sync provera u petlji slojeva). CPU fallback (`D2H + design + gelsd + H2D`) nosi `34.57 ms` (`54.54%` vremena na `big`,
  od čega `gelsd` `32.11 ms`, a D2H/H2D `1.40 ms`), dok na `3/4/5layer` nosi `18.0–19.1%`.
- `measure_split_sweep.py` (`n_splits = 1..64`): linearno skaliranje vremena `O(n)` i konstantan `peak_vram_mb` `O(1)` (`480.52 MB` na `big`).
- Napomena (20.10.2026): pre pisanja finalnog `HYBRID_RESULTS.md` otkrivena su dva upstream baga u originalnom Pasado kodu
  (`Zonotope.__neg__` ne negira generatore i `check_nonlinear_boundary` gleda samo `+E` umesto `|E|`), zbog kojih će se
  apsolutne vrednosti `lc_precise` u split sweep-u ponoviti na grani sa ispravkama.

U ukupno vreme uključi oba prenosa i CPU solve. Razlaganje po sloju meri
odvojeno: dodatne sinhronizacije profiler-a mogu promeniti e2e vreme.

Gotovo kad: postoji jedna tabela koju bi pokazao Saši bez dodatnog objašnjavanja.

## T7 — Završni izveštaj o hibridu (~1 h)

Kratki izveštaji počinju ranije, posle svaka 2–3 radna dana; ne čekaju T7.

- [ ] Rezultat hibrida (ovo je poruka koju čeka od 23.8).
- [ ] Pitanje o GPU-u grupe / NCSA Delta, koje 12.9 nije otišlo.
- [ ] Da li „fokus na big" sužava obim ispod Faze 3.

## T8 — Ostatak higijene (odloženo, ~2 h)

- [ ] Mehanički sweep preostalih factory poziva bez `device=`:
      `precise_transformer.py` stari per-neuron putevi (`:172`, `:378-452`,
      `:583-651`, `:771-773`, `:816-818`), `HyperDuals.py`, `HyperDualZono.py`,
      `SimpleZono.py:212-227,455-460,547`.
- [x] `conv.py:38` već koristi `np.prod` u zatečenom kodu; nema nove izmene.

Radi se samo ako ostane vremena ili ako CNN/softplus put uđe u rad.

## Reference za dopune 19.9.

- [PyTorch tensor devices i izuzetak za 0-dim CPU skalare](https://docs.pytorch.org/docs/main/tensor_attributes.html)
- [Ograničenja set_sync_debug_mode](https://docs.pytorch.org/docs/main/generated/torch.cuda.set_sync_debug_mode.html)
- [Lokalni audit i današnje provere](notes/PROJECT_AUDIT_2026-09-19.md)
