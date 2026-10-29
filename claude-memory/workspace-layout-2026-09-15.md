---
name: workspace-layout-2026-09-15
description: Raspored ~/prakse/UIUC posle sređivanja 15.9.2026 (papers/, notes/, archive/, review/, Pasado-original worktree) i gde je šta commitovano
metadata:
  type: project
---

Workspace sređen 15.9.2026 na korisnikov zahtev ("clean repo, pa čitam");
mapa je u `~/prakse/UIUC/README.md`. Ključno:
- `Pasado/` na grani `sprint-optimizations` (3 commita: kod+testovi,
  experiments+results+logs, reproducibility beleške); status čist. Regenerisani
  artefakti Sekcija 5.2/5.3/5.5 vraćeni na upstream; `Section_5_4/results/*.pth`
  zadržani regenerisani jer ih čita `compare_pasado_lirpa.py`. `profiles/` (65 MB)
  i `MNIST_Data/` su na disku, isključeni kroz `.git/info/exclude`, ne kroz .gitignore.
- `Pasado-original/` = `git worktree` upstream `origin/main` (detached), read-only za
  čitanje; ignorisan u workspace git-u.
- `papers/` (OOPSLA2023, 2510.08726, quantization/), `notes/` (statusi, poruke Saši,
  MAC_SETUP, stari TODO), `archive/` (stari profili/logovi, TVM faza, transfer extras),
  `review/` (diffovi + mapa rada + REVIEW_NOTES), `claude-memory/` = mirror ove memorije.
- Git identitet podešen samo lokalno u oba repoa (`Dimitrije Pešić
  <dimitrije.pesicc@gmail.com>`); globalno nije, pa novi klon traži `git config --global`.
- Ništa nije obrisano osim `Section_5_2/.claude/` (Windows permission fajl).

**Why:** korisnik je hteo originalni Pasado/neptune/radove čiste od artefakata pre faze
učenja; sve reverzibilno preko `mac-transfer` snapshota.
**How to apply:** putanje u [[learning-phase-2026-09-07]] i UCENJE.md važe; PDF-ovi su
sada u `papers/`; za original koristiti `Pasado-original/` ili `git show origin/main:`.
Videti [[gpu-hybrid-port-plan]], [[mac-setup-2026-09-07]].

**28.10.2026:** u korenu workspace-a je `CLAUDE.md` (stalni kontekst: pravila, raspored,
trenutno stanje; učitava se i iz `Pasado/`). `Pasado` je sada na grani `gpu-hybrid`
(remote `fork`); `Pasado-original/` je worktree grane `fix-zonotope-neg-and-boundary`
(upstream PR #3). Handoff-ovi su u `notes/HANDOFF_GPU_HYBRID_*.md`.
