# Plan: heterogeni CPU–GPU runtime za preciznu NN analizu (MLSys pravac)

> Nastalo iz spoljnog feedbacka (12.8.2026) na pitanje "je li ovo dovoljno za
> MLSys". Teza feedbacka: ne "portovali smo Pasado na GPU", nego heterogeni
> runtime koji odlučuje gde i kako se izvršavaju delovi analize, sa
> equal-time evaluacijom kao metodološkim doprinosom. Ovaj fajl mapira tu
> viziju na ono što je već izmereno i deli je na faze od kojih je svaka
> samostalno završiv rezultat.

## Zašto je vizija izvodljiva: pola "budućeg rada" već postoji

Feedback traži → mi već imamo:

| Traži se | Stanje |
|---|---|
| CPU optimizacija | ✅ 3.1–3.3× / 1.65×, bit-identično (`FORMAL_SUMMARY.md`) |
| Moderan baseline | ✅ auto_LiRPA poređenje, 1.3–2.8× (`LIRPA_COMPARISON.md`) |
| equal-K vs equal-time uvid | ✅ izmereno, 15/15 ćelija (`EQUAL_TIME_COMPARISON.md`) |
| "CPU vs GPU po operatoru" — *zašto* | ⚙️ pola urađeno: `big` profil daje podelu (matmul 33% GPU-friendly; regresija CPU-vezana zbog `gelsd`; `get_linspace` sa 385k `.item()` sync-hostile) — `BIG_PROFILE_ANALYSIS.md`, `GPU_READINESS.md` |
| Više hardvera | ⚙️ pola: A100 fp64 ≈ fp32 (~65× CPU), T4 fp64 ≈ 1/17 fp32 → na T4 scheduler treba skoro sve da drži na CPU — **to je dokaz da scheduler nije A100 hack, već feature** |
| Crossover po obliku mreže | ⚙️ nagovešteno: uske mreže → Python/dispatch režija (CPU teren), `big` → gust BLAS (GPU teren) |
| varying depth/width/eps/splits | ✅ tabele već postoje za tesnoću; fale za performanse GPU strane |

Ključni uvid koji feedback nema, a mi imamo izmeren: **broj šumnih simbola
raste kroz slojeve**, pa veličina matmul-a raste *tokom jednog prolaza* —
optimalno mesto izvršavanja može da se menja usred mreže, ne samo između
mreža. To je pravi "dynamic scheduling" argument, ne konstruisan.

## Faze (svaka je završiv rezultat sama za sebe)

### Faza 1 — Karakterizacija operatora CPU vs GPU (~1–2 nedelje)
Cilj: crossover tabela po (operator, širina, batch, uređaj) — "zašto" za scheduler.
1. **Batch `get_linspace`** — ionako sledeće usko grlo CPU puta (385k `.item()`).
   Dekartov proizvod je bit-identičan; batched `linspace` je ~1 ULP u 14%
   slučajeva → dokumentovana odluka o preciznosti, ne tiha zamena.
2. **Device/dtype higijena** — spisak već postoji u `GPU_READINESS.md`
   (`new_ones`, `device=`, konstante u `inverse_poly_tensor`, `traceify`...).
3. **Rang-provera za lstsq dispatch**: degenerisane kutije → CPU `gelsd`,
   ostatak → GPU `gels` (izmereno: `gels` na degenerisanim greši 15–34 reda
   veličine — motivacija je već dokumentovana; test harness
   `colab_rank_deficient_lstsq_test.py` postoji). Ovo je mikro-primer
   heterogenosti: dispatch po *numeričkoj* karakteristici ulaza.
4. Mikrobenčmark po operatoru na A100 + T4 + lokalni CPU: `AffineZonotope`
   matmul, abs/sum redukcije granica, corner/boundary kerneli, batched lstsq,
   grid generacija — kao funkcija širine (100 vs 1024) i batch dimenzije.
5. **`torch.compile` revizit na Linux+GPU** — na Windows CPU bio blokiran
   (nema MSVC); Colab skripte već pripremljene (`colab_compile_boundary.py`,
   `colab_dynamo_explain.py`).
Izlaz: tabela preseka + Amdahl procena plafona za Fazu 2. Gruba računica iz
`big` profila: ~51% vremena je GPU-friendly (matmul+redukcije), ~24% regresija
(delimično portabilna uz rang-proveru) → realan plafon end-to-end na `big`
možda ~2.5–3.5× preko optimizovanog CPU-a, tj. **~4–5× preko originala**
kumulativno. Na uskim mrežama GPU neće pomoći — i to JE poenta scheduler-a.

### Faza 2 — GPU-only end-to-end + statički hibrid (~2 nedelje)
- Ceo precizni prolaz na GPU za `big` (lstsq preko rang-dispatcha).
- End-to-end CPU vs GPU po mreži → crossover po obliku (očekivano: 3layer
  CPU pobeđuje, `big` GPU pobeđuje; gde je tačka preloma po širini?).
- Statički placement po operatoru (iz Faze 1) = prvi "hybrid" broj.
- Definisati numerički ugovor unapred: bit-identičnost umire na GPU
  (redosled redukcija, `gels` vs `gelsd`) → prag prihvatanja i verifikacija
  po uzoru na postojeći correctness harness.

### Faza 3 — Konkurentni CPU+GPU scheduling podproblema (~2–3 nedelje)
- n_splits podproblemi su nezavisni (equal-time sweep ih je već koristio do
  k=256) → makespan minimizacija: veliki/gusti na GPU stream, mali/iregularni
  na CPU pool, **istovremeno**.
- Jeftin među-korak koji jača i postojeći rezultat: subproblem paralelizam
  samo po CPU jezgrima (multiprocessing) — equal-time dominacija raste i bez
  GPU-a.
- Obavezna ablacija: oracle placement vs naš scheduler vs CPU-only vs
  GPU-only. Cost model neka bude jednostavan (predikcija iz oblika: operator,
  širina, batch — fitovan na mikrobenčmark), oracle drži poštenje.
- Rizik koji treba priznati: ako je crossover čist, "dinamički" scheduler
  degeneriše u statičko pravilo. Odbrana: rast broja šumnih simbola kroz
  dubinu menja veličine usred prolaza; i T4 vs A100 menja sve odluke.

### Faza 4 — Generalnost (paralelno sa 3, po Sašinom prioritetu)
- **Dublje mreže (6–8 sigmoid slojeva)**: jeftino (trening je trivijalan),
  produžava već izmeren trend 1.32 → 1.93 → 2.83.
- **CNN put Sekcije 5.4** (uz postojeći `np.product` fix) kao drugi workload.
- **Sekcije 5.2/5.3 — oprez**: koriste *ne-tenzorizovanu* implementaciju;
  "samo dodaj workload" tu znači prvo tenzorizovati → potencijalna eksplozija
  obima. Pitanje za Sašu da li to ulazi u rad ili ostaje future work.
- Hardver: A100, T4, lokalni CPU (sve već dostupno); eventualno L4/V100.

## Go/no-go gate-ovi (dogovoreno posle drugog kruga feedbacka, 12.8.)

- **Gate 1 (posle Faze 1):** *Da li je crossover dovoljno nestabilan da
  runtime odluka ima smisla?* Tražimo: isti operator prelazi CPU→GPU sa
  rastom broja šumnih simbola; A100 i T4 imaju različite tačke preseka;
  lstsq odluka zavisi od ranka a ne samo veličine; `.item()` pattern pravi
  merljiv sync problem. **Ako je crossover čist i statičan (`n < X → CPU`),
  NE gradimo scheduler** — minimalan statički dispatcher, a priča ide ka
  heterogeneous execution + workload characterization.
- **Gate 2 (posle Faze 2):** statički hibrid mora da tuče i najbolji
  CPU-only i najbolji GPU-only end-to-end, dvocifrenim procentom. Ako je
  2–3%, concurrency nema ROI.
- **Gate 3 (posle Faze 3):** `CPU-only | GPU-only | static hybrid | dynamic
  hybrid | oracle` + **scheduler efficiency = achieved/oracle speedup**.
  Blizina oracle-u je lepši rezultat od još jednog X×.
- **Hard stop: Faza 1 → rezultati → Saša → tek onda arhitektura
  scheduler-a.** Nikakav runtime framework pre nego što merenja pokažu da
  problem postoji.

## Mikrobenčmark harness: šta se beleži (šema za budući cost model)

Po merenju: operator, sloj, mreža, oblik ulaza, **broj šumnih simbola**,
batch, dtype, uređaj, ukupno vreme, transfer vreme, kernel/compute vreme,
rank/degeneracija gde je relevantno. Cost model kasnije: interpretabilan
piecewise `T_cpu(op, shape, state)` vs `T_gpu(op, shape, state) + transfer`,
dispatch = minimum. **Ne ML-based scheduler** — jednostavan model kome
reviewer veruje.

## Metodološko zaključavanje equal-time rezultata (pre teksta)

Headline "Pasado stiže tešnje od LiRPA maksimuma pre nego što LiRPA završi
jedan solve" će reviewer pokušati da razbije. Finalna tabela mora imati:
identičan problem/property; identičnu preciznost gde je moguće; jasno
uključeno/isključeno setup/compile vreme; CUDA sync pre merenja; warm-up +
više ponavljanja; median + disperzija; prijavljen broj CPU threadova; isti
wall-clock budžet; A100 i T4 odvojeno; bez cherry-pickovanja (sve ćelije).

## Terminologija

CPU **nije fallback** nego first-class execution resource. Priča: "Pasado
exposes heterogeneous numerical workloads whose optimal execution device
changes across operators, architectures, hardware, and even layers within a
single analysis as symbolic state grows."

## Šta NE raditi (anti-scope-creep)
- Ne graditi opšti scheduler framework pre nego što Faza 1 pokaže da
  crossover-i uopšte postoje i gde su.
- Ne JAX/Triton — eksplicitno odloženo i dalje stoji.
- Ne tenzorizacija 5.2/5.3 bez Sašinog eksplicitnog da.
- Ne juriti "killer broj" iz feedbacka (4.8×) — broj mora da izađe iz
  Amdahl-a Faze 1, ne iz želje.

## Redosled tvrdnji za abstract (feedbackova struktura, naši brojevi)
1. CPU optimizacija: do 3.3×, bit-identično. ✅
2. Protiv modernog baseline-a prednost je 1.3–2.8× po prolazu, ne 182–2750×. ✅
3. Equal-K evaluacija suštinski pogrešno karakteriše tradeoff; pri jednakom
   vremenu Pasado konzistentno tešnji (15/15, 14–50× manje vremena). ✅
4. [Sistem]: heterogeni CPU–GPU runtime; hybrid vs CPU-only vs GPU-only vs
   oracle; end-to-end ubrzanje + tešnje granice od CROWN familije pri
   jednakom compute budžetu. ⬅ ovo se gradi (Faze 1–3)
5. Design principle: za preciznu NN analizu ni GPU-only izvršavanje ni
   fixed-work evaluacija nisu adekvatni — heterogeno izvršavanje i
   time-budgeted poređenje menjaju precision/performance frontier.

## Za razgovor sa Sašom
- Predstaviti punu viziju (ne samo a/b/c pitanje), ali sa fazama i jasnim
  "šta je dovoljno za submission" pragom (feedbackov nivo C).
- Pitati: MLSys deadline koji ciljamo (proveriti CFP za sledeći ciklus);
  da li 5.2/5.3 ulaze u scope; da li ima pristup stabilnijem GPU hardveru
  od Colaba za finalna merenja.
- Politička nota ostaje: poređenje sa Pasado radom formulisati kao
  "fer evaluacija + nov runtime", ne kao kritiku.

## Prva konkretna nedelja (počinje odmah, ne čeka call)
1. ✅ **Batch `get_linspace` — GOTOVO (12.8.)**: implementacija je već
   postojala od 10.8. (`PASADO_BATCHED_GRID`), bit-egzaktna (symmetric-halves;
   „1 ULP" priča bila artefakt naivne formule). Validirano kroz 12 kombinacija
   selektora (1.4e-17), izmereno: +1,31–1,35× naspram prethodne finalne;
   **direktno original→finalna: 3,24× / 3,64× / 3,37×** (pinovano,
   naizmenično). `FORMAL_SUMMARY.md` ažuriran. Usput: `PASADO_REAL_CUBIC`
   postoji, bez CPU dobitka, ne-bit-identičan (≤5,7e-5) → parkiran kao
   GPU/compile enabler.
2. Device higijena po spisku iz `GPU_READINESS.md`.
3. Rang-provera lstsq dispatch + validacija na degenerisanim kutijama.
4. Mikrobenčmark skripta po operatoru (CPU odmah; A100/T4 čim bude Colab).
5. Pinovan timing run equal-time tabele (ionako dug, ide preko noći).
