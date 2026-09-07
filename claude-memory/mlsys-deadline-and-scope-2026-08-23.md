---
name: mlsys-deadline-and-scope-2026-08-23
description: MLSys rok 30.10.2026, iskrena procena catch-up vremena za razumevanje Pasado koda i procena da li je putanja MLSys-worthy; tenzija Faza 1-2 vs Faza 3 iz PLAN_MLSYS.md
metadata:
  type: project
---

**Rok: MLSys 30.10.2026** (korisnik potvrdio 23.8; ranije u `PLAN_MLSYS.md`
stajalo kao otvoreno pitanje "proveriti CFP"). Danas 23.8, prvi korisnikov
commit u Pasado 5.8 → 18 dana prošlo, **68 dana / ~9.7 nedelja ostaje**.

**Kontekst pitanja:** korisnik je otvoreno rekao da nije sam radio kod od
skidanja Pasada nadalje (uz Claude-a) i pitao koliko treba da nadoknadi
razumevanje čitajući kod, paralelno producirajući rezultate.

**Procena catch-up vremena** (data 23.8): jezgro za razumevanje ~2.800–3.000
gustih linija (`precise_transformer.py` 884, `SimpleZono.py` 583,
`HyperDuals.py` 576, `Duals.py` 297, `HyperDualZono.py` 191,
`get_lipschitz.py` 253; testove ne treba čitati liniju-po-liniju). Blokirano
puno vreme: 3-4 radna dana (24-32h) — 1.5 dan fundacija (OOPSLA2023 rad +
netaknuti delovi SimpleZono/Duals), 1.5-2 dana precise_transformer + sopstvene
izmene (poklapa se sa [[pasado-code-review-plan]] Nivo 1 — isti zadatak, ne
raditi dvaput). Preporuka: NE blokirati GPU port na ovome — prvih 1-1.5 dan
čisto čitanje, ostatak se pokriva kroz [[gpu-hybrid-port-plan]] i Nivo 1
pregled dok se radi, razvučeno na ~1.5-2 nedelje kalendarski.

**Procena MLSys-worthiness:** uslovno da. `PLAN_MLSYS.md` Faze 1+2
(karakterizacija + statički hibrid, ~3-4 nedelje procenjeno) su dovoljne za
solidan rad i realno izvodljive u preostalom vremenu (68d − 3-4d catch-up −
2-3 nedelje pisanja ≈ 5-6 nedelja za istraživanje). Puna Faza 3 (dinamički
scheduler + oracle, još 2-3 nedelje) je optimistički scenario, nije
garantovana u roku, pogotovo uz GPU numeriku (correctness debugging na Colabu
nije besplatan). Claims 1-3 iz plana (equal-time metodologija, pošten
baseline vs auto_LiRPA, equal-time dominacija 15/15) su već gotovi i
samostalno vredni, nezavisno od GPU dela.

**Neraščišćena tenzija — treba pitati Sašu uskoro:** Sašina poruka 23.8
("optimizuj primarno za big, male mreže ne zanimaju") sužava scope u odnosu
na PLAN_MLSYS.md Gate 1, koji za dinamički scheduler traži zanimljiv crossover
*između* veličina mreža. Ako Sašu zanima samo big, to prirodno gura rad ka
Fazi 1+2 (karakterizacija + statički hibrid), što je uži ali izvodljiviji rad
od pune Faze 3 vizije. Treba eksplicitno pitati da li je Faza 1+2 dovoljna za
ovaj ciklus pre nego što se troše nedelje na nešto što možda nije scope.
Videti [[misailo-feedback-2026-08-22]].
