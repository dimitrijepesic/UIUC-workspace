---
name: misailo-feedback-2026-08-22
description: "Sašin (Misailović) odgovor na status poruku, 22.8.2026 — poslednja prepiska: rezultat odobren, prvo hibrid CPU/GPU + profiling gelsd-a; pita Jacoba za gels; narativ 'sa optimizacijama brži od alternative, bez njih sporiji'"
metadata: 
  node_type: memory
  type: project
  originSessionId: 612801d8-3028-4418-9dd9-259d72b296a4
  modified: 2026-08-21T23:17:45.140Z
---

Status poruka Saši (stavka #2 iz `TODO_sutra.md`) poslata je preko Slacka;
ovo je **poslednja prepiska sa profesorom, stanje 22.8.2026**.

Šta je javljeno: optimizacije 3.2–3.6× (male mreže) / 1.65× (`big`),
bit-identično; protiv auto_LiRPA one-pass prednost 1.3–2.8× (naspram
182–2750× iz rada, koji poredi samo sa interval/zonotope AD); equal-time
Pasado tešnji u svih 15 postavki, laptop CPU protiv A100.

Sašine reakcije i odluke:

1. **"Odličan novi rezultat!"** — posebno ako se pokaže da je Pasado **sa**
   optimizacijama brži od alternative, a **bez** njih sporiji. → To je narativ
   za rad; traži novu ablaciju: neoptimizovan Pasado vs auto_LiRPA.
2. **Hibrid CPU/GPU kao prvo rešenje** (degenerisani `gelsd` slučajevi na
   CPU, ostatak — pre svega veliki matmul na `big` — na GPU): "to radi kao
   prvo rešenje".
3. **Prvo profilirati koliko vremena odlazi na `gelsd`** — ako je malo (male
   matrice), njegova optimizacija je sekundarna; ako je veliki matmul
   dominantan, izmeriti koliko njegovo ubrzanje pomaže.
4. Saša **pita Jacoba** (Laurel, autor Pasada) da li je `gelsd` neophodan za
   te matrice ili bi `gels` (postoji CUDA implementacija) bio dovoljan —
   čeka se njegov odgovor.
5. Fallback ako je hibrid prespor: implementirati `gelsd` u CUDA (uz pomoć
   Claude-a) ili probati **MAGMA** (github.com/icl-utk-edu/magma); cuSolver
   nema `gelsd`.

Ovim je de facto odgovoreno na pitanje pravca (#6 iz Week1 statusa):
sistemski smer (GPU/hibrid) + poredbeni narativ — poklapa se sa Fazama 1–2
u `PLAN_MLSYS.md`. Videti [[pasado-code-review-plan]].

**Dopuna 23.8.2026, 00:27 (odgovor na profiling poruku):** Saša prihvatio
rezultat ("sasvim ima smisla"). Odluka: **optimizovati primarno za `big`
mrežu** — cilj je analiza mreža *većih* od nje, pa matmul + redukcije na GPU
imaju najviše smisla. Ako neka druga operacija postane bottleneck, rešavaće se
tad. → `get_linspace` batching za male mreže je deprioritizovan; hibrid
CPU(gelsd)/GPU port za `big` je sada glavni zadatak. Videti
[[gelsd-profiling-result-2026-08-22]].

**23.8. ~00:30, zaključeno:** korisnik javio Saši da kreće sa hibridom (zonotope
tenzori na GPU, gelsd na CPU, prvo `big`, meri ubrzanje + bit-identičnost);
Korisnikova poruka Saši: "Dogovoreno! Javljam kad vidim rezultate hibridnog pristupa." Sledeća poruka Saši = rezultati hibridnog pristupa.
Nema otvorenih pitanja prema njemu.
