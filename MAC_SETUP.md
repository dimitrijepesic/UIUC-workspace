# Prenos UIUC foldera na MacBook

Sve što je bilo na Windows-u (`C:\everything\prakse\UIUC`) rekonstruiše se iz:

- **UIUC-workspace** (ovaj repo, privatan): beleške, PDF-ovi, profili, logovi, quantization,
  lipschitz-comparison skripte, VS Code / Claude podešavanja, Claude memorija, neptune extras.
- **Pasado-wip** (privatan): Pasado sa svim necommitovanim izmenama, u grani `mac-transfer`.
- **upstream repoi** (javni): neptune, Local-Lipschitz-Constants, auto_LiRPA.
- **venv-ovi**: prave se ponovo iz `requirements-win-freeze.txt` fajlova.

## 1. Alati

```bash
xcode-select --install          # git + kompajler (preskoči ako već imaš)
brew install gh python@3.13 python@3.11
gh auth login                   # GitHub login (treba za privatne repoe)
gh auth setup-git
```

## 2. Kloniranje

```bash
mkdir -p ~/prakse && cd ~/prakse
git clone https://github.com/dimitrijepesic/UIUC-workspace.git UIUC
cd ~/prakse/UIUC

# Pasado: privatni backup, pa vrati remote-ove kao na Windows-u
git clone https://github.com/dimitrijepesic/Pasado-wip.git Pasado
cd Pasado
git remote rename origin backup
git remote add origin https://github.com/uiuc-arc/Pasado.git
git remote add fork https://github.com/dimitrijepesic/Pasado
git checkout fix-compat-bugs
git checkout mac-transfer -- .     # prebaci snapshot u radno stablo
git reset -q                       # i ostavi ga kao necommitovane izmene
git status --short | wc -l         # očekivano: 66 (isto kao na Windows-u)
cd ..

# neptune (nema lokalnih commit-ova) + fajlovi koji nisu bili u git-u
git clone https://github.com/uiuc-arc/neptune.git
cp extras/neptune/KNOWN_ISSUES.md neptune/
mkdir -p neptune/.claude && cp extras/neptune/.claude/settings.local.json neptune/.claude/

# Local-Lipschitz-Constants (čist klon)
git clone https://github.com/shizhouxing/Local-Lipschitz-Constants.git

# auto_LiRPA, zakucan na isti commit kao na Windows-u
git clone https://github.com/Verified-Intelligence/auto_LiRPA lipschitz-comparison/auto_LiRPA_src
git -C lipschitz-comparison/auto_LiRPA_src checkout 5a098e8
```

## 3. Python okruženja

```bash
cd ~/prakse/UIUC/Pasado
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-win-freeze.txt
deactivate

cd ~/prakse/UIUC/lipschitz-comparison
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-win-freeze.txt     # instalira i ./auto_LiRPA_src kao editable
deactivate
```

Ako pip zapne na nekoj verziji, skini pin (`==x.y.z`) za taj paket i probaj ponovo.
Torch na Mac-u je CPU/MPS verzija: kod koji zove `.cuda()` ne radi lokalno, GPU deo ide na Colab.

## 4. Claude Code memorija

Pokreni `claude` jednom iz `~/prakse/UIUC`, izađi, pa nađi folder projekta i prekopiraj memoriju:

```bash
ls ~/.claude/projects/            # nešto kao -Users-<ime>-prakse-UIUC
mkdir -p ~/.claude/projects/-Users-$(whoami)-prakse-UIUC/memory
cp -r ~/prakse/UIUC/claude-memory/. ~/.claude/projects/-Users-$(whoami)-prakse-UIUC/memory/
```

## 5. Provera

```bash
cd ~/prakse/UIUC
ls                                            # isti sadržaj kao na Windows-u
git -C Pasado status --short | head           # M i ?? fajlovi kao na Windows-u
Pasado/.venv/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```
