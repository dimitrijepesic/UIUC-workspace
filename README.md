# UIUC workspace (Pasado / MLSys)

Mapa foldera, sređeno 15.9.2026.

| folder / fajl | šta je |
|---|---|
| `Pasado/` | Pasado. Grana `sprint-optimizations` = ceo moj rad (kod, testovi, `experiments/`, `results/`). Upstream je `origin/main`, snapshot pre sređivanja je grana `mac-transfer`. |
| `Pasado-original/` | read-only git worktree upstream Pasada (`origin/main`), za čitanje uporedo sa mojim kodom. Nije u git-u workspace-a. |
| `neptune/` | Neptune klon (upstream `main`) + moj `KNOWN_ISSUES.md` |
| `Local-Lipschitz-Constants/` | Shi et al. 2022, kod baseline-a (klon) |
| `lipschitz-comparison/` | moje poređenje Pasado vs auto_LiRPA (skripte + CSV); `auto_LiRPA_src/` je klon @5a098e8 |
| `papers/` | radovi: OOPSLA 2023 (Pasado), 2510.08726 (Neptune), `quantization/` (ranija faza) |
| `review/` | kit za čitanje: diffovi mog koda naspram upstream-a, mapa rada, beleške |
| `UCENJE.md` | plan čitanja koda i radova |
| `PLAN_MLSYS.md` | plan rada za MLSys |
| `notes/` | istorija: statusi, poruke Saši, zaključci, stari TODO, Mac transfer |
| `claude-memory/` | kopija Claude memorije projekta (izvor: `~/.claude/projects/-Users-dimitrije-prakse-UIUC/memory/`) |
| `archive/` | artefakti koji se ne čitaju: stari profili i logovi, TVM kvantizacija, transfer extras |

Venv-ovi (nisu u git-u): `Pasado/.venv` (py3.13), `lipschitz-comparison/.venv` (py3.11). Kako se prave: `notes/MAC_SETUP.md`.

Brzi test da sve radi:

```bash
cd Pasado && source .venv/bin/activate && cd Section_5_4
python get_lipschitz.py --network 3layer --no-save     # ~6 s
```
