# Review notes

Format: `fajl:linija` — pitanje / sporno / potvrđeno. Vidi `../UCENJE.md`.

## Dan 1 — rad + netaknuti kod

## Dan 2 — precise_transformer diff

## Dan 3 — rezultati i skripte

## Dan 4 — ostali radovi + ceo lanac

## T0 — 26.9.2026 — početno stanje pre porta (grana `gpu-hybrid`)

Kontekst: `Pasado/` HEAD `515c864` (`sprint-optimizations`), grana
`gpu-hybrid` napravljena od nje. Python 3.13.15, torch 2.12.1 (CPU),
arm64 Mac. Testovi pokrenuti iz `.venv`, bez `PASADO_*` flagova u okruženju.

### Snimak testova pre bilo kakve izmene

| Test | Rezultat | Napomena |
|---|---|---|
| `forward_mode_tensorized_src/test_batched_grid.py` | ispisuje `FAIL`, **exit 0** | STEPS=3/5 i svi degenerisani slučajevi (point box, lx==ux, width 1e-12, magnitude 1e6; svi STEPS=5) bit-egzaktni; STEPS=7/9 `max|diff|=8.882e-16` (1 ULP). Selector deo: 12/12 kombinacija OK, najgore odstupanje 1.388e-17. |
| `forward_mode_tensorized_src/test_selector_paths.py` | PASS, exit 0 | 12/12 OK; n=100 `max|dg|=1.665e-16` (vektorizovani check-ovi smeju da se razlikuju u poslednjim bitovima). |
| `experiments/test_batched_lin_reg.py` | PASS, exit 0 | 13 slučajeva × {float64, float32}, bit-identično svuda (uklj. degenerate_x, point, mixed healthy+degenerate). |
| `Section_5_4/get_lipschitz.py --network 3layer --no-save` | radi, ~4 s | BEZ flagova = mešana varijanta (vec=1, bnd=1, lstsq=0, grid=0). `lc_precise[0]=526.172…`, `[15]=115.341…`; `time_precise` ≈ 5–6 ms/slika. Upozorenje `pin_memory ... not supported on MPS` je bezopasno (DataLoader na Mac-u). |

Prvi import `sklearn`/`scipy` u svežoj sesiji traje 10–30 s (Gatekeeper,
hladan keš); nije greška. `precise_transformer.py:4` uvlači `sklearn` samo
zbog starog `lin_reg` (`:158-160`) koji tenzorizovani put ne zove → T8.

### Tumačenje grid testa (STEPS=7/9)

- Nije regresija: produkcija koristi samo STEPS=5 (podrazumevano u
  `get_linspace_batched`), a to je bit-egzaktno.
- Uzrok (radna hipoteza, arm64): u `_linspace_rows`
  (`precise_transformer.py:83-92`) množioci `t*step` / `(STEPS-1-t)*step`
  su za STEPS=3 i 5 samo 0, 1, 2 (množenje egzaktno), a za STEPS=7 i 9 prvi
  put se javlja 3. Tada `torch.linspace` kernel (`start + step*idx` u jednom
  C++ izrazu, na arm64 kontrahovano u FMA = jedno zaokruživanje) i naša dva
  odvojena op-a (`mul` pa `add` = dva zaokruživanja) mogu da se razlikuju za
  1 ULP. Na x86 build-u bez FMA kontrakcije poklapaju se (odatle stara
  beleška „bit-egzaktno za STEPS ∈ {3,5,7,9}“).
- Dva defekta testa koja se ispravljaju u T0: (a) `print("PASS"/"FAIL")`
  bez exit koda; (b) `all_ok &= ok` za sve STEPS, pa poznati 1-ULP slučaj
  obara ceo test. Formula u `_linspace_rows` se **ne menja**.
- Cilj posle izmene: STEPS=5 + degenerisani slučajevi odlučuju o exit kodu
  (bit-egzaktno ili pad); STEPS=7/9 ostaju u izlazu sa labelom „known
  non-bit-exact on arm64“ i gornjom granicom od nekoliko ULP.

### Zaključani flagovi (od sada uvek eksplicitno svih pet)

```
original:  PASADO_VECTORIZED_PRECISE=0 PASADO_VEC_BOUNDARY=0 PASADO_BATCHED_LSTSQ=0 PASADO_BATCHED_GRID=0 PASADO_REAL_CUBIC=0
najbolji:  PASADO_VECTORIZED_PRECISE=1 PASADO_VEC_BOUNDARY=1 PASADO_BATCHED_LSTSQ=1 PASADO_BATCHED_GRID=1 PASADO_REAL_CUBIC=0
```

Podrazumevane vrednosti (blok sa prekidačima na vrhu `precise_transformer.py`): prva dva „1“,
ostala tri „0“ → run bez flagova NIJE „najbolji“ nego mešana varijanta.
Osnova za port = `najbolji` (`vec_batched_grid`).

### Ostalo u T0 (zatvoreno 3.10.2026, D02)

- [x] upisati rezultate preostala tri testa u tabelu
- [x] popraviti `test_batched_grid.py`: STEPS 3/5 i zero-width kutije su strogi,
      STEPS 7/9 su informativni (granica 4 ULP), exit kod prati rezultat.
      Mutanti (1e-9 relativno, +1 ULP) daju `exit=1`.
- [x] šum seed-a (`experiments/measure_seed_noise.py`, log
      `Pasado/logs/seed_noise_2026-10-03.txt`), 3layer, svih pet flagova eksplicitno:
      - 3 neseedovana `get_lipschitz.py` runa (prosek preko 30 slika): maks. relativna
        razlika `lc_precise` 1.55e-5, medijana po epsilonu ~5e-6
        (prvi epsilon: 526.1725, 526.1740, 526.1748);
      - jedan forward sa istim seed-om dvaput: tačno isti rezultat;
        sa drugim seed-om: do 7.8e-5 relativno (6 slučajeva).
      Ovo je ilustracija, ne tolerancija za `big`.
- [x] zašto poređenje GPU vs `lc_precise_big.pth` ne može da bude „≤1e-10“:
      `sigmoid_prime_product_tensor` dodaje koeficijentu A slučajan šum
      `np.random.normal(0, 1e-4)`, a `get_lipschitz.py` nigde ne seed-uje
      `np.random`. Sačuvani fajl je zato prosek jednog neseedovanog runa i ne može se
      reprodukovati; šum između runova je reda 1e-5 do 1e-4, šest redova veličine
      iznad 1e-10. Za poređenje CPU/GPU treba sveža CPU referenca sa istim seed-om,
      mrežom, slikama, epsilonima i flagovima (T1 uvodi seed).
- [x] flagovi: svaka varijanta u `benchmark_lipschitz.py` i `profile_variants.py`
      zadaje svih pet; `experiments/test_benchmark_variants.py` dokazuje da izvezeni
      `PASADO_*` više ne prolaze (stari mapping: 6 od 6 varijanti je bilo kontaminirano).
- [x] commit izmena testa (pushovano na `fork/gpu-hybrid`)
