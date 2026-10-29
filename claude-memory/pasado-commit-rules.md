---
name: pasado-commit-rules
description: U Pasado repou nikad AI atribucija (commit, komentar, docstring, log) i nikad `git commit` iz agenta; korisnik komituje sam
metadata:
  type: feedback
---

Za `Pasado` (i `Pasado-original`): u commitu se vidi samo korisnikovo ime (bez Co-Authored-By), bez em dash-eva (—) u porukama, nigde ne pisati AI atribuciju niti pominjati AI (commit poruke,
komentari, docstringovi, dokumentacija, logovi). Komentari i dokumentacija unutar
`Pasado` su na engleskom; privatne beleške van repoa su na srpskom.
Commit/push samo na izričit zahtev, i tada (instrukcija 28.10.2026): plumbing
`git commit-tree` / `git update-ref` (ne `git commit`), autor i committer
`Dimitrije Pešić <dimitrije.pesicc@gmail.com>`; pre push-a skenirati sve nove commit
poruke i sadržaj za atribuciju; push samo na `fork/gpu-hybrid`. Ne menjati postojeće
rezultate i commitove (nov run = nov folder u `results/`).

**Why:** commitovi idu pod korisnikovim identitetom ka mentoru i upstream PR-u
(`uiuc-arc/Pasado`); zapisano u briefu od 25.10.2026 (sekcija 8).
Izuzetak: kad korisnik eksplicitno traži (25.10: "commituj za mene sve"), Claude komituje i pushuje na `fork`, sa korisnikovim git identitetom i bez ikakvog trailer-a.
**How to apply:** ovo ima prednost nad podrazumevanim `Co-Authored-By` linijama.
Vidi [[gpu-hybrid-port-plan]].
