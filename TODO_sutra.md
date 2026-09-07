# TODO — sutra

Realno za jedan dan (~6-7h). Poređano po vrednosti, ne po redosledu izvršavanja.

## Spremno pre nego što sedneš

- `lipschitz-comparison/pasado_splits.csv` — Pasado kolona pri budžetima
  k ∈ {1,3,5,7,9} za 3/4/5layer (sweep pušten sinoć, ~25 min, trebalo bi da je
  gotov).
- `experiments/colab_bab_sweep.py` — GPU skripta, pushovana na `colab-gpu-wip`,
  netestirana na Colabu.
- Dva otvorena PR-a na `uiuc-arc/Pasado` (verzije + kompatibilnost).

---

## 1. Pusti GPU sweep prvo (5 min rada, ~2-3h mašinskog vremena)

Uradi ovo **pre svega ostalog** da radi u pozadini dok ti radiš tačku 2.

```
!git clone https://github.com/uiuc-arc/Pasado.git
!mkdir -p Pasado/experiments
!pip install -q --ignore-requires-python git+https://github.com/Verified-Intelligence/auto_LiRPA.git
!wget -q -O Pasado/experiments/sweep.py \
  https://raw.githubusercontent.com/dimitrijepesic/Pasado/colab-gpu-wip/experiments/colab_bab_sweep.py
!python Pasado/experiments/sweep.py --networks 3layer 4layer 5layer --eps-indices 0 1 2 3 4 5 6
```

Runtime: A100 + High RAM. Skripta je resumabilna — ako sesija umre, isti poziv
nastavlja gde je stao.

**Prvo proveri na jednom paru** da ne izgubiš 3h na grešku:
`--networks 3layer --eps-indices 2 --num-images 3` (~2 min). Ignoriši upozorenje
o `num-images != 30`, za dimni test je u redu.

## 2. Poruka Saši (1-2h) — ovo je najvrednija stavka dana

Iz `UIUC_Research_Status_Week1.md` je ostalo **šest neodgovorenih pitanja**, a
najvažnije je #6: *"What level of result would be useful for a potential research
writeup: profiling report, compiler-friendliness analysis, or an actual
fused-kernel speedup?"* Sve od tada radimo pogađajući odgovor.

Sad imaš konkretno šta da pokažeš, pa poruka nije "šta da radim" nego "evo šta
sam našao, koji smer hoćeš".

Sadržaj, kratko:
- **Optimizacija**: 3.1-3.3× na 3/4/5layer, 1.65× na `big`, bit-identično,
  iza selektora. (Izvor: `FORMAL_SUMMARY.md`.)
- **Novi baseline koji rad nema**: Pasado se u 5.4 poredi samo sa svojim ranijim
  radom (interval/zonotope AD). Protiv stvarnog SOTA (auto_LiRPA, Shi et al.
  2022 — koga rad citira ali samo za 5.5) prednost je **1.3-2.8×, ne 182-2750×**.
- **Presek pri jednakom budžetu**: sa BaB-om na obe strane, Pasado vodi do ~7
  podproblema, pa auto_LiRPA prestiže. (Izvor: `LIRPA_COMPARISON.md`.)
- **GPU**: fp64 na A100 je besplatan (~65× CPU), ali `gelsd` ne postoji na CUDA
  i `gels` je izmereno nebezbedan na degenerisanim kutijama (15-34 reda
  veličine). (Izvor: `GPU_READINESS.md`.)
- **Pitanje za njega**: da li ga zanima (a) sistemski smer — GPU port sa
  rang-proverom, (b) ovaj poredbeni/karakterizacioni smer za rad, ili (c) nešto
  treće što je imao na umu od početka.

**Pitaj ga i za Sekcije 5.2/5.3.** Tamo Pasado nema konkurenciju (CROWN ne radi
ODE ni Black-Scholes), a niko ih nije ni profilisao. Moguće da je to bolji teren
od 5.4, gde smo mi ušli u najtežu bitku koju Pasado ima.

## 3. Spoji dve kolone u jednu tabelu (1h, kad GPU sweep završi)

`pasado_splits.csv` + `bab_sweep.csv` → jedna tabela `odnos(mreža, eps, budžet)`
i, po (mreža, eps), **budžet na kom se krive seku**.

Pitanje na koje odgovaraš: da li je presek na ~7 tipičan ili je 5layer/eps=0.2
bio izuzetak. Očekivanje: na drugim epsilonima presek je na **manjem** budžetu,
jer je eps=0.2 bila najpovoljnija tačka za Pasado.

Rezultat upiši u `LIRPA_COMPARISON.md` (sekcija o BaB-u već postoji, treba samo
zameniti jednu tačku punom tabelom).

## 4. Ako ostane vremena (bira se jedno, ne sve)

- **`big` GPU sweep** — `--networks big`, odvojeno jer je solve 4.7 s naspram
  ~1 s. Još ~2h mašinskog vremena.
- **GPU vremena za uske mreže** — jedina preostala nemerena Pasado prednost je
  cena po podproblemu, a sadašnje poređenje (0.44 s naspram 20 s) je CPU-CPU i
  nepošteno prema GPU alatu. Bez ovoga tvrdnja o ceni ne sme u tekst.
- **Commit-uj sprint rad.** Sve stoji necommit-ovano u radnom stablu već danima.
  Bar lokalna grana, da ne visi.

---

## Namerno NE sutra

- GPU port Pasado-a (rang-provera + CPU fallback) — dobro definisano, ali je to
  višednevni posao, ne popodne.
- Triton kerneli — preuranjeno dok `get_linspace` nije batch-ovan.
- JAX/vmap frontend — velik, i ima smisla tek posle Sašinog odgovora o smeru.
- Bilo šta na Sekcijama 5.2/5.3 — prvo pitaj, pa radi.
