---
name: pasado-code-review-plan
description: "Plan i izmereni obim za pregled celog Dimitrijevog koda u Pasado repou (avgust 2026), sa procenom 7-10h"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1489c91b-649b-4983-bc93-7bc416043762
  modified: 2026-08-21T23:13:40.587Z
---

Zadatak (dogovoren 2026-08-22): proći kroz svaku liniju koda koju je korisnik
uradio u `C:\everything\prakse\UIUC\Pasado` od početka prakse.

**Izmereni obim (stanje 2026-08-22):**
- 11 commit-ova (~1.150 dodatih linija) na granama `update-dependency-versions`,
  `fix-compat-bugs`, `colab-gpu-wip`; fork remote je `dimitrijepesic/Pasado`,
  upstream je `uiuc-arc/Pasado` (2 otvorena PR-a). Tri velika commit-a:
  `d6ec21a` Viete cubic + compile benchmark (576), `94cf917` GPU BaB sweep (249),
  `08f3bd0` Colab poredbeni skript (207); ostalo sitne popravke.
- Necommit-ovano u radnom stablu: ~400 linija koda, od toga 382 u
  `forward_mode_tensorized_src/precise_transformer.py` (jezgro optimizacije).
- Untracked: ~3.300 linija u `experiments/` (14 skripti) + 5 test fajlova u
  `forward_mode_tensorized_src/`.
- Ukupno ~4.900 linija.

**Dogovoreni metod — tri nivoa strogosti (ne hronološki po commit-ovima):**
1. Liniju po liniju (~1.200 l., 4-6h): diff `precise_transformer.py`, commit
   `d6ec21a`, izmene `Section_5_4/get_lipschitz.py` + `lipschitz.sh` — kod koji
   nosi tvrdnje "bit-identično" i "3.1-3.3×".
2. Metodološki pregled (~700 l., ~2h): `colab_bab_sweep.py`,
   `colab_compare_pasado_lirpa.py`, `colab_compile_boundary.py` — skripte koje
   su proizvele brojeve za Sašu.
3. Prelet (~3.000 l., 1-2h): jednokratne dijagnostičke/profiling skripte;
   testove pustiti umesto čitati.

Priprema: prvo commit-ovati sprint rad (stoji necommit-ovan danima); praviti
diff fajlove po celini (`git diff origin/main...fork/colab-gpu-wip`,
`git diff origin/main -- forward_mode_tensorized_src/ Section_5_4/`); beleške
voditi u `REVIEW_NOTES.md` (fajl:linija + šta je sporno).

Ukupna procena: 7-10h slojevito (preporučeno); 15-20h ako baš sve istom
strogošću. Ponuđeno (neiskorišćeno): Claude pusti /code-review preko diff-ova
ili sam prođe Nivo 1, pa ručni prolaz bude druga provera.

Dnevni plan za ostatak rada stoji u `C:\everything\prakse\UIUC\TODO_sutra.md`
(GPU sweep na Colabu, poruka Saši sa 6 pitanja, spajanje tabela). Videti i
[[known_issues_tracking]].
