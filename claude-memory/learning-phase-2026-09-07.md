---
name: learning-phase-2026-09-07
description: Od 7.9.2026 korisnik prvo uči ceo Pasado kod, svoje razlike naspram originala i radove, pa tek onda nastavlja hibridni port; plan u UCENJE.md, diffovi u review/
metadata:
  type: project
---

Korisnikova odluka 7.9.2026: "prvo naučim sav kod i skontam razlike koje sam
napravio u odnosu na Pasado original, čitam radove, pa nastavljam sa radom."
Plan (4–5 dana) je u `~/prakse/UIUC/UCENJE.md`; diff fajlovi naspram
`uiuc-arc/Pasado` main-a su u `~/prakse/UIUC/review/01..04_*.diff`, beleške
idu u `review/REVIEW_NOTES.md`. Tek posle toga korak 0 iz [[gpu-hybrid-port-plan]].

**Why:** korisnik je 23.8 sam rekao da nije sam pisao kod od skidanja Pasada
(radio uz Claude-a) i želi da ga razume pre nego što gradi dalje; vidi
[[mlsys-deadline-and-scope-2026-08-23]] i [[pasado-code-review-plan]].
**How to apply:** u ovoj fazi objašnjavaj, ne piši kod umesto njega; kad pita
za deo koda, uputi ga na hunk u `review/` i test koji ga dokazuje. Ne
predlagati da Claude "sam prođe Nivo 1" osim ako korisnik traži.
