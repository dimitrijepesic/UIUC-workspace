# Nacrt poruke Saši, 12.9.2026

Zdravo Saša,

pre nego što krenem u hibridni CPU/GPU port, hteo sam da ti pošaljem sve na jednom
mestu: gde smo, šta je sledeće i sa datumima, i nekoliko pravaca za rad, da mi kažeš
šta ti je od toga zanimljivo.

**Gde smo (Pasado, Sekcija 5.4)**

| | rezultat |
|---|---|
| CPU optimizacije preciznog transformera | 3,2–3,6× (3/4/5layer), 1,65× (big), bit-identično, sve iza selektora |
| Novi baseline koji rad nema (auto_LiRPA, CROWN-Optimized, fp64) | Pasado tešnji 1,3–2,8× po prolazu; sa BaB na obe strane presek pri ~7 podproblema |
| Equal-time poređenje | pri jednakom vremenu Pasado tešnji u 15/15 postavki, 14–50× manje vremena (laptop CPU vs A100) |
| Profil big | matmul u AffineZonotope ~45%+, gelsd 1,7% → hibrid ima smisla |
| GPU spremnost | gels na degenerisanim kutijama greši 15–34 reda veličine → gelsd ostaje na CPU; T4 fp64 je 1/17 fp32, A100 fp64 ≈ fp32 |

Sve je dokumentovano u repou (`results/optimization_section_5_4/`), testovi prolaze
i na Mac-u (arm64) na koji sam prešao.

**Sledeće, sa datumima**

- do 30.9: hibrid na `big` (zonotope tenzori na GPU, gelsd na CPU), korektnost na A100
  sa tolerancijom (ne bit-identično, jer redosled redukcija na GPU-u to ne dozvoljava;
  i sam rad kaže da implementacija nije floating-point sound), merenje end-to-end.
  Očekujem plafon 2–3× preko najbolje CPU verzije.
- do 15.10: karakterizacija po operatoru (A100 / T4 / CPU) kao funkcija širine i broja
  šumnih simbola → gde je tačka preloma, i ablacija koju si tražio: neoptimizovan
  Pasado vs auto_LiRPA.

**Pravci za rad, šta ti zvuči zanimljivo?**

1. *Heterogeno izvršavanje za precizne analize*: statički CPU/GPU hibrid, crossover
   po operatoru i hardveru; poenta je da se optimalan uređaj menja i unutar jednog
   prolaza jer broj šumnih simbola raste kroz slojeve. Ovo je ono što radim sada.
2. *Dinamički scheduler podproblema* (BaB podproblemi su nezavisni): veliki na GPU,
   mali na CPU, istovremeno, sa oracle ablacijom. Ima smisla samo ako merenja iz 1
   pokažu da crossover nije trivijalan.
3. *Metodološki narativ*: equal-K vs equal-time evaluacija menja sliku Pasado vs
   CROWN familija; sa optimizacijama Pasado brži, bez njih sporiji.
4. *Generalnost*: dublje/šire mreže od big, CNN put; eventualno Sekcije 5.2/5.3
   (ODE, Black-Scholes) gde CROWN ne postoji kao konkurencija, ali je taj kod
   ne-tenzorizovan pa je to veći zalogaj.

**Tri pitanja**

- Da li i dalje ciljamo publikaciju, i da li je MLSys (rok 30.10) realan cilj ili
  ciljamo sledeći rok? Ako MLSys, da li je 1 + 3 dovoljno za submission ili želiš i 2?
- Koji od pravaca ti je najzanimljiviji kao centar rada?
- Da li grupa ima stabilniji GPU od Colaba za finalna merenja (A100 sesije pucaju
  na dugim sweep-ovima)?

Javljam ti status na svaka dva dana, kratko.

Dimitrije

---

# Kratka verzija (poslata 12.9.2026)

Zdravo Saša,

sredio sam sve što sam do sada uradio na Pasadu na jedno mesto: optimizacije, poređenje sa auto_LiRPA, equal-time rezultat, profil big-a i šta je spremno za GPU. Sve je u repou sa testovima, prešao sam i na Mac. Sad krećem u hibrid CPU/GPU za big, rezultat očekujem do kraja septembra.

Voleo bih da vidim gde ovo dalje ide, pa dva pitanja:

1. Da li i dalje ciljamo publikaciju, i da li je MLSys (rok 30.10) realan ili idemo na sledeći rok? Pravci koje vidim: heterogeni CPU/GPU hibrid (to radim sad), equal-time narativ naspram CROWN familije, i eventualno dinamički scheduler podproblema. Šta ti od toga zvuči najzanimljivije?

2. Da li bi me povezao sa Jacobom i ostalim koautorima, ako su zainteresovani? Mislim da bismo brže i kvalitetnije radili ako znaju šta radim, pogotovo oko gelsd-a i numerike.

Javljaću ti kratko na svaka dva dana.

Dimitrije
