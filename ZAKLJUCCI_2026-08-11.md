# Zaključci — 11. avgust 2026 (posle GPU BaB sweep-a)

> Context fajl: stanje razmišljanja posle spajanja `pasado_splits.csv` (Pasado
> n_splits, CPU) i `bab_sweep.csv` (auto_LiRPA BaB, A100). Detaljni brojevi su u
> `Pasado/results/optimization_section_5_4/LIRPA_COMPARISON.md`; ovde je slika
> odozgo i odluka šta dalje.

## Šta smo utvrdili (stanje dokaza)

1. **Merenja su čvrsta.** A100 brojevi za 5layer/eps=0.2 identični su ranijem
   CPU BaB eksperimentu (3641 / 1533 / 483.4 / 250.7) — tri nezavisna puta kroz
   kod daju iste vrednosti.

2. **Headline brojevi Pasado rada (182–2750×) su artefakt slabog baseline-a.**
   Protiv stvarnog SOTA (auto_LiRPA, CROWN-Optimized) jednoprolazna prednost je
   **1.3–2.8×**. Raste sa brojem sigmoid slojeva (2→1.32, 3→1.93, 4→2.83), ne
   sa širinom (`big`: ista dubina kao 5layer, 10× širi, 2.70 — neznatno manje).

3. **Pri jednakom broju podproblema presek je na K≈7 — i to je pravilo, ne
   izuzetak** (pojas 6.6–8.6 kroz sve tri mreže i srednje epsilone). Ali:
   - na malim epsilonima (≤0.11) presek se pomera **iznad K=9** — suprotno
     ranijem očekivanju; Pasado ostaje marginalno tešnji kroz ceo mereni opseg;
   - odnos **nije monoton**: vrh Pasadove prednosti je na K=3 (npr. 5layer/
     eps=0.63: 1.75 → 3.67 → 3.11 → 1.42 → 0.61), jer uniformna podela na malom
     budžetu steže više od best-first bisekcije (posle 2 deljenja BaB-ov najgori
     interval je eps/2 naspram Pasadovih eps/3).

4. **Cena po podproblemu, sada izmerena i protiv A100:** Pasado na laptop CPU-u
   ~0.02–0.04 s po podproblemu naspram ~0.57–1.49 s auto_LiRPA-i na A100 —
   **~30–36× jeftinije pri jednakoj tesnoći (K=7)**. Stara CPU-CPU procena
   (~45×) je povučena i zamenjena ovom. Uz to, auto_LiRPA-in Jacobian put ne
   skalira na široke mreže (~10.7 GB po solve-u na `big`).

5. Inženjerski rezultat sprinta — ažurirano 12.8. posle validacije batched
   grid-a: **3.24× / 3.64× / 3.37×** (3/4/5layer, direktno mereno, pinovano,
   naizmenično) i 1.65× (`big`, još bez grida), bit-identično, iza selektora.
   Grid usput uklanja sve `.item()` sync tačke preciznog puta → preduslov za
   GPU. Napomena: lančanje odnosa iz različitih sesija precenjuje (~4.2–4.4×)
   — merodavni su samo direktni parovi iz iste sesije.

## Procena za rad

- Rad **"Pasado je bolji od SOTA" ne postoji** — postoji pošteniji rad:
  *kada i zašto precizna AD apstraktna interpretacija pobeđuje bound
  propagation* (dubina → gomilanje relaksacione greške; mali split budžeti;
  cena po podproblemu; memorijsko skaliranje).
- Kao samostalna evaluacija: workshop nivo. Za jači rad fali bar jedno od:
  **(a)** equal-time rezultat + eventualno hibrid, **(b)** proširenje na
  Sekcije 5.2/5.3 (ODE, Black-Scholes — tamo CROWN ne igra, konkurencije
  nema), **(c)** sistemski pravac (GPU port sa rang-proverom).
- Slabosti kojih smo svesni: sve važi za jedan usko definisan zadatak (MNIST,
  sigmoid FCN, Lipšic po **skalarnom** haze parametru); Shi et al. rešavaju
  opštiji problem. Politički: nalaz ide grupi koja je pisala Pasado — ton mora
  biti "evo šta fer poređenje pokazuje", ne "razbili smo rad".
- **Izbor pravca je Sašino pitanje #6** iz Week1 statusa — poruka njemu je
  sledeći korak, posle equal-time merenja.

## Equal-time poređenje — IZMERENO, hipoteza potvrđena

> **Ishod (isti dan):** pri jednakom vremenu Pasado je tešnji u **svih 15
> ćelija** (1.02–2.52×), bez ijednog preseka. Dovoljno je k ≤ 16 (≤ 0.9 s po
> slici, CPU) da se pobedi sve što je auto_LiRPA izmerila do 13.4 s po slici
> na A100 — vremenska prednost 14–50×. Pasado k=256 je praktično konvergirao
> (5layer: ~141 kroz sve epsilone); auto_LiRPA K=9 pri istom vremenu ostaje
> do 2.5× iznad. Presek na K≈7 je artefakt equal-budget metrike. Detalji:
> `Pasado/results/optimization_section_5_4/EQUAL_TIME_COMPARISON.md`.
> Preostala ograda: vremena potvrditi pinovanim runom pre teksta.

### Originalna hipoteza i plan (radi istorije)

Presek na K≈7 važi za jednak **broj podproblema** — metrika koja favorizuje
skupljeg. Pri jednakom **vremenu** Pasado može da priušti ~30× više podela.
Pošto obe metode konvergiraju sa deljenjem, očekivanje je da **pri jednakom
wall-clock budžetu Pasado dominira svuda u merenom režimu**. Ako je tako,
narativ se okreće iz "prednost nestaje na k≥7" u "pri jednakom vremenu Pasado
pobeđuje" — mnogo jača teza.

Plan eksperimenta (jeftin, lokalno, samo Pasado strana):
- `n_splits ∈ {16, 64, 256}` × mreže {3,4,5}layer × eps indeksi 0–4 × istih 30
  slika; postojeći `pasado_splits.csv` pokriva {1,3,5,7,9} za donji kraj krive.
- k=256 vremenski odgovara auto_LiRPA-inom K≈7–9 budžetu na A100 po mreži
  (~4.9 s naspram 5.1 s na 3layer; ~7.2 / 8.7 na 4layer; ~10.5 / 13.4 na
  5layer) — ako Pasado na k=256 tuče LiRPA-in K=9, dominacija važi za ceo
  mereni opseg.
- Odlučujuće su eps indeksi 0–2 (tamo LiRPA vodi na K=7/9); indeksi 3–4 su
  gotovo automatski (odnos ≥1 već pri jednakom K).
- Skripta: `lipschitz-comparison/pasado_equal_time_sweep.py` (adaptacija
  `pasado_splits_sweep.py`), izlaz `pasado_equal_time.csv`; rezultati idu u
  `Pasado/results/optimization_section_5_4/EQUAL_TIME_COMPARISON.md`.
- Ograda: Pasado vremena su sa nepinovane mašine — tesnoća je deterministična,
  vremena okvirna; za tekst rada vremena treba potvrditi pinovanim runom.

## Otvoreno posle ovoga

- Poruka Saši (tri stuba: jednoprolazna tesnoća, presek pri jednakom budžetu,
  equal-time; pitanje pravca a/b/c). Pitati i za 5.2/5.3.
- `bab_sweep.csv` skinuti sa Colaba (lokalno postoji samo rekonstrukcija
  30-slikovnih proseka u `bab_sweep_gpu.csv`).
- Eps indeksi 5–6 i `big` nisu u BaB sweep-u.
- Sprint kod i dalje necommitovan.
