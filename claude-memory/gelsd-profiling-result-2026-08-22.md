---
name: gelsd-profiling-result-2026-08-22
description: Odgovor na Sašino pitanje #3 (22.8.2026) — gelsd je 1.7% na big, matmul u AffineZonotope 45%; hibrid CPU/GPU je pravo rešenje, gelsd/MAGMA spor otpada
metadata:
  type: project
---

Profil `big` mreže u aktuelnoj batched varijanti urađen 2026-08-22 uveče
(`experiments/profile_variants.py --network big --variants vec_both_batched`,
13.6 min; rezultat `profiles/prof_big_vec_both_batched.prof`, log
`logs/profile_big_vec_both_batched.log`).

Udeo u ukupnom vremenu `get_lipschitz` runa (3layer / 5layer / big):
- ukupno: 21 s / 38 s / 816 s (stara ne-batched big varijanta: 1249 s)
- `AffineZonotope` (= `generators @ layer`, veliki matmul): 2.3% / 2.8% / **45.3%**
- `gelsd` (`linalg_lstsq`): 3.9% / 5.0% / **1.7%**
- `torch.cat`: 1.4% / 2.4% / 8.4%
- `abs`+`sum` (zonotope redukcije): 2.6% / 4.3% / 13.8%
- `get_linspace` (kum., nebatch-ovana mreža uzoraka): 16.8% / 26.9% / 10.4%

Zaključci za Sašu (odgovor na njegovu tačku 3 iz [[misailo-feedback-2026-08-22]]):
1. `gelsd` je nebitan (≤5% male, 1.7% big) → spor gelsd/gels/CUDA/MAGMA
   (njegove tačke 4–5) nije prioritet; Jacobov odgovor ne menja plan.
2. Sašina hipoteza o velikom matmul-u je tačna na `big`: ~70% vremena je
   čist tenzorski posao (matmul+cat+abs+sum) → hibrid CPU(gelsd)/GPU(ostalo)
   je pravo i verovatno jedino potrebno rešenje.
3. Na malim mrežama GPU ne pomaže; tamo je cilj batching `get_linspace`.

Zamka: cProfile knjiži `@` operator kao self-time funkcije koja ga zove
(zato `torch.matmul` builtin pokazuje ≤2% — to su samo eksplicitni pozivi).
cProfile potcenjuje velike tenzorske op., pa je 45% donja granica.

Sledeće: poslati ovu tabelu Saši; zatim hibrid GPU port (rang-provera +
CPU fallback za gelsd) je višednevni posao — videti `TODO_sutra.md` sekciju
"Namerno NE sutra". Videti [[pasado-code-review-plan]].

**Dopuna (22.8. kasno uveče):** rezultat poslat Saši na Slack — tekst: gelsd
1.7% ukupnog runa (~4% precise puta), matmul+cat/abs/sum ~70%, predlog hibrid
CPU(gelsd)/GPU(ostalo), male mreže get_linspace bottleneck. Čeka se odgovor.

Nijanse utvrđene naknadno:
- `get_lipschitz` run = interval + baseline zonotope (`forward_zono`) + Pasado
  precise (`forward_zono_precise`); gornji procenti su nad celim runom. Samo
  precise put (343 s od 816 s na big): gelsd ~4% big / ~10% male, matmul ~54%,
  `get_linspace` **48–52%** na malim mrežama.
- Oba zonotope puta zovu isti `AffineDualZonotope` → polovina 370 s matmul-a
  pripada baseline-u.
- **Čudno, za pregled koda:** baseline `forward_zono` traje 382 s, duže od
  Pasado precise 343 s na big; `SigmoidDualZonotope.__mul__` = 119 s. Može da
  menja odnose u tabelama rada — proveriti u [[pasado-code-review-plan]] Nivou 1.
