# Mapa čitanja: papers/OOPSLA2023Final.pdf → pitanja iz UCENJE.md Dan 1.1

Strane su PDF strane (= `291:N` u zaglavlju). Rad ima 29 strana; nosivo je 4–25.
Napravljeno 12.9.2026 kao vodič, ne zamena za čitanje.

## Redosled koji preporučujem (≈ pola dana)

| # | strane | šta | pitanje iz UCENJE.md |
|---|---|---|---|
| 1 | 1–3 | apstrakt, uvod, doprinosi | kontekst: šta Pasado radi (sinteza transformera za chain/product/quotient pattern) |
| 2 | 4–10 | Sekcija 2, running example (klima ODE, Euler, Fig 1–3) | **najbolji uvod**: dual brojevi u kodu (Fig 1), rast šumnih simbola ε1…ε12 kroz jedan korak (Fig 2, str. 8), chain rule sinteza korak po korak (str. 7) |
| 3 | 10–11 | Sekcija 3.1–3.2 | dual brojevi, forward vs reverse; formalna definicija apstraktnog stanja: afina forma + interval, `Range`, γ (str. 11) |
| 4 | 12–13 | Sekcija 4.1, Fig 4, **Algoritam 1**, jednačine 3–5 | **ovo je tačno funkcija koju si vektorizovao** (vidi tabelu ispod) |
| 5 | 18 | Teorema 4.1 (dokaz) | zašto su kandidati za max greške samo 4 ugla + koreni f''(x)=A/l_y i A/u_y |
| 6 | 20 | Sekcija 5.1 Methodology | hardver rada, "not floating-point sound" (str. 20) |
| 7 | 24–25 | Sekcija 5.4 | šta se meri, koje mreže, sa čim se poredi, tabela vremena |
| 8 | 14–17, 19 | 4.2–4.3 product/quotient, 4.5–4.6 | prelet: nije u tenzorizovanom kodu za 5.4, ali objašnjava `inverse_poly` ideju (kubna u x2) |
| 9 | 20–24, 25–27 | 5.2, 5.3, 5.5 | prelet |

## Algoritam 1 (str. 13) ↔ `forward_mode_tensorized_src/precise_transformer.py`

Pattern: g(x, y) = f'(x)·y, kod nas f = σ, pa g = σ(x)(1−σ(x))·y (Fig 4, str. 12).

| korak u Algoritmu 1 | kod (linija) | napomena |
|---|---|---|
| `l_x,u_x ← Range(a[x])`, isto za y | pozivalac, `sigmoid_prime_product_tensor` (716) | granice iz reduced product-a (afina forma ∩ interval) |
| `grid ← GridSample([l_x,u_x]×[l_y,u_y])` | `get_linspace` (63) / `get_linspace_batched` (96) | STEPS=5 → 25 tačaka po neuronu; rad: "we limit the number of sampled grid points (trades off precision)" str. 11 |
| `pts ← {f'(x)·y}` | `sigmoid_prime_times_y` (117) | |
| `A,B,C ← LinearRegression` | `lin_reg_tensor` (170) / `lin_reg_tensor_batched` (199) | lstsq gelsd; oblik [n,25,3] |
| `if A = 0 then A ← A+δ` | potraži u `lin_reg_tensor` | rad: δ < 1e-9, uslov A≠0 za dokaz (str. 13) — **proveri da li kod to radi** |
| `D ← max |f'(x)·y − (Ax+By+C)|` | `compute_max_error` (664) = `check_corners` (375/403) ∪ `check_nonlinear_boundary` (449/532) | Teorema 4.1: uglovi + tačke (x*, l_y), (x**, u_y) gde f''(x*)=A/l_y |
| koreni f''(x)=c | `inverse_sigmoid_2nd_deriv` (311) → `inverse_poly_tensor` (215) / `inverse_poly_real_tensor` (246, Viete) | σ''=s(1−s)(1−2s) sa s=σ(x) → **kubna po s**, pa x=logit(s) (`inv_sigmoid`, 146). Rad str. 11: "for many functions f''^{-1} has an analytical formula" |
| `l,u ← min/max f'(x)·y` | isti kandidati | jedn. 4–5, Teorema 4.4: optimalno za intervalni domen |
| `return A·x̂ + B·ŷ + C + D·ε_new, [l,u]` | kraj `sigmoid_prime_product_tensor`, `PreciseSigmoidDualZonotope` (841) | **jedan** novi šumni simbol po neuronu (str. 7) |

Ispravka u UCENJE.md: fit je **linearan** (hiperravan A·x+B·y+C), ne kvadratni.
"Kubna" se odnosi samo na jednačinu za kritične tačke.

## Tri stvari iz rada koje direktno nose naš MLSys narativ

1. **Rast matmul-a kroz slojeve** (str. 6–7, Fig 2): svaka nelinearna operacija dodaje
   šumni simbol; Pasado dodaje 1 po patternu umesto 3. Na FFNN-u to znači +width kolona
   generatora po sigmoid sloju → `AffineZonotope` matmul raste tokom prolaza.
   Ovo je "dynamic scheduling" argument iz `PLAN_MLSYS.md`, sa citatom u radu.
2. **Anomalija "baseline sporiji od precise na big"** (naša memorija: 382 s vs 343 s).
   Rad str. 25, poslednji pasus 5.4: na ConvBig je Pasado (191 s) *brži* od zonotope AD
   (212 s) upravo zbog manje šumnih simbola. Ali za **FFNNBig rad daje suprotno**:
   interval 0.0542 s, zonotope 0.361 s, Pasado 1.49 s po konfiguraciji (str. 25),
   tj. Pasado 4× sporiji. Naš `big` = FFNNBig (Singh 2019b, 95.8%). Dakle naša
   anomalija nije nemoguća, ali se ne slaže sa tabelom rada za tu mrežu → pitanje
   za REVIEW_NOTES: koji baseline mi pokrećemo (`forward_zono`, get_lipschitz.py:64/91)
   naspram "implementation of Laurel 2022b directly" (str. 20)? Hardver rada:
   32-jezgarni Threadripper 3975WX + RTX A5000 (str. 20), mi: laptop CPU.
3. **"Pasado's implementation assumes ideal real arithmetic and is thus not
   floating-point sound"** (str. 20). Argument za numerički ugovor hibrida:
   tolerancija umesto bit-identičnosti je u duhu originala, ne popuštanje.

## Sekcija 5.4, brojevi koje treba znati napamet (str. 24–25)

- Zadatak: lokalna Lipschitz konstanta izlaza mreže po haze parametru α,
  p_α(x_i; α) = (1−α)·x_i + α, MNIST, 1000 tačno klasifikovanih slika (FFNN), 30 (CNN).
- Mreže: 3/4/5-layer FFNN (acc 92.7/89.2/86.7%), FFNNBig (95.8%), ConvSmall/Med/Big.
- Baseline: interval AD i zonotope AD iz Laurel 2022a/b, **ne CROWN/auto_LiRPA**.
- Preciznost: do 182× (FFNNBig), do 2750× (ConvBig) naspram zonotope AD.
- Vremena po konfiguraciji (interval / zonotope / Pasado):
  3layer 0.00338/0.00210/0.0636; 4layer 0.00383/0.00352/0.0969;
  5layer 0.00469/0.00533/0.135; FFNNBig 0.0542/0.361/1.49 s.

## Napomena o HyperDuals

Rad pokriva samo prve izvode (dual brojevi); kaže da su njegove konkretne semantike
"first-order version of Laurel et al. 2022b" (str. 5). `HyperDuals.py`/`HyperDualZono.py`
potiču iz tog ranijeg rada. U `get_lipschitz.py` hiper-dual se pojavljuje samo kroz
`HyperDualIntervalToDualZonotope` (linije 173, 188) pre zonotope puteva → na Dan 1.2
proveri šta se tačno računa na intervalnom putu i zašto je konverzija potrebna.
