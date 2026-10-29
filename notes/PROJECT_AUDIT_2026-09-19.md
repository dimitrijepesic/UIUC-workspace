# Stanje projekta i nalazi pregleda — 19.9.2026.

Pregled služi kao polazna tačka za [dnevni plan](../DAILY_PROGRESS_PLAN.md).
Razlikuje sačuvane istorijske rezultate, današnje provere i budući rad.

## Šta projekat radi

Pasado je statička analiza diferencijabilnih programa: računa granice vrednosti
i izvoda nad celim domenom ulaza. Ovde je glavni workload Sekcija 5.4:
MNIST sigmoid mreža, slika `img`, haze parametar `t ∈ [0, epsilon]`,
`x(t) = (1-t)*img + t`. Rezultat po slici je gornja granica
`max_j |dF_j/dt|`, izvedena iz donjih/gornjih granica dualnog zonotopa.
Aktuelni harness proseči rezultate preko 30 tačno klasifikovanih slika.

Precizni sigmoid transformer aproksimira `sigmoid'(x)*y` ravni
`A*x + B*y + C`, pa dodaje grešku koja obuhvata odstupanje na kutiji.
To je **linearni fit**; kubna jednačina se pojavljuje pri traženju kritičnih
tačaka, a ne pri fitovanju. Propagacija čuva centre i generatore zonotopa;
nelinearni slojevi dodaju šumne simbole i povećavaju naredne matrice.

Istraživački pravac: zadržati kvalitet analize, ubrzati njeno izvršavanje
vektorizacijom/batchingom i CPU/GPU podelom rada, pa izmeriti granice
u odnosu na utrošeno vreme protiv auto_LiRPA Jacobian baseline-a.
Ovo nije optimizacija treninga ili običnog inference-a mreže.

## Obuhvat pregleda

- Pregledana je struktura svih projektnih repozitorijuma, lokalna Git istorija,
  planovi, review kit, beleške o komunikaciji i istraživačka memorija.
- Detaljno je praćen aktivni Pasado put: `get_lipschitz.py`, `model.py`,
  `SimpleZono.py`, `precise_transformer.py`, relevantni dual/interval konstruktori,
  correctness i benchmark harness, testovi optimizacija.
- Pregledani su izveštaji o CPU optimizacijama, profilima, GPU numerici,
  realnom kubnom solveru, keširanju i compile eksperimentima, kao i
  auto_LiRPA poređenje, BaB/split skripte, agregacija i CSV podaci.
- Provereni su relevantni delovi Pasado rada (Algoritam 1, metodologija,
  Sekcija 5.4), dokumentacija koda i uvodni deo Neptune rada.
- Neptune i Local-Lipschitz-Constants pregledani su kao prateći projekti;
  ovo nije audit svake linije nasleđenog TVM/auto_LiRPA koda niti ponavljanje
  celog eksperimentalnog dela svih radova. Arhiva kvantizacije je ranija faza.

## Šta je već urađeno

| Oblast | Zatečeno stanje i dokaz | Kako koristiti u daljem radu |
|---|---|---|
| Reprodukcija Pasada | Istorijski runovi 5.2/5.3/5.4 FFNN/5.5 i testova, `Pasado/KNOWN_ISSUES.md` | Prethodni rezultat; nije sve ponovo pokrenuto danas. |
| Doprinos upstream-u | Lokalni `origin/main` sadrži merge PR #1 i #2: dependency dokumentacija, NumPy 2 i putanje | Gotovo prema lokalnoj Git istoriji; ne planirati ponovo te popravke. |
| Neptune reprodukcija | `neptune/KNOWN_ISSUES.md`: svih 10 attention operatora na malom obliku, dve lokalne Docker popravke | Gotov prethodni pregled; nisu urađeni novi Neptune compiler doprinosi. |
| Vektorizacija preciznog transformera | Tenzorske provere uglova/granice i uklonjen povratak tensor→lista→tensor | Implementirano, sa testovima. |
| Batched regresija | `lin_reg_tensor_batched`, isti CPU `gelsd`, [n,25,3] sistemi | Implementirano; uklanja per-neuron pozive, ne menja solver. |
| Batched grid | `_linspace_rows` i `get_linspace_batched` | Implementirano; produkcioni STEPS=5 prolazi lokalnu proveru. |
| CPU performanse | `FORMAL_SUMMARY.md`: 3,24× / 3,64× / 3,37× na uskim mrežama; istorijskih 1,65× na `big` pre grida | Stari hardver/protokol; `big` sa najnovijim gridom nije isti eksperiment. Nisu nova današnja merenja. |
| Korektnost | Jedinični testovi, selector kombinacije, seeded e2e | Postoji jaka regresiona osnova, ali ne dokaz univerzalne bit-identičnosti. |
| auto_LiRPA baseline | Ista haze funkcija i skalarni Jacobian, isti checkpoint-i, fp64; one-pass tabele | Odnosi do približno 1,3–2,8× su **tesnoća granica**, ne ubrzanje. Zavise od epsilona i mreže. |
| Podela domena | Pasado uniformni splits; auto_LiRPA best-first BaB; CSV i crossover tabele | Rezultati postoje; jasno razlikovati broj solve-ova i broj aktivnih intervala. |
| Equal-time | 285 Pasado redova i 75 BaB redova ulaze u proveru konzistentnosti | Brojčane tvrdnje o vremenu su preliminarne; vidi nalaz A ispod. |
| GPU priprema | T4/A100 mikroprovere, full-rank/rank-deficient regresija, audit uređaja | Mikrorezultati postoje. End-to-end hibrid ne postoji u aktivnom kodu. |
| Realni kubni solver | `PASADO_REAL_CUBIC`, Viète, drugačija numerika | Gotov eksperiment; ostaje isključen za osnovni hibrid. |
| Negativni nalazi | Keširanje nema pogođene ponovljene ulaze u uzorku; stari compile pokušaji bez potvrđenog e2e dobitka | Ne vraćati automatski u osnovni backlog. |
| Organizacija | `sprint-optimizations` je čist, HEAD `515c864`; worktree originala, diffovi, freeze i beleške postoje | Commitovanje starog sprinta više nije otvoren task. |

## Šta još nije urađeno

- Nema lokalne grane `gpu-hybrid`; postoje `main`, `fix-compat-bugs`,
  `mac-transfer`, `sprint-optimizations`.
- Nema `PASADO_DEVICE`/CUDA ulaznog puta u `get_lipschitz.py`.
- `lin_reg_tensor_batched` direktno zove `gelsd` na uređaju ulaza;
  nema prenosa na CPU i povratka rezultata na GPU.
- Nema end-to-end CUDA correctness rezultata, brzine hibrida ili njegove
  memorijske potrošnje.
- Nema zaključane savremene četvorostruke tabele
  originalni Pasado / optimizovan CPU / hibrid / auto_LiRPA pod istim uslovima.
- Nema završene GPU karakterizacije po sloju, oblika većih od `big`,
  dinamičkog scheduler-a ni teksta rada.
- `review/REVIEW_NOTES.md` je prazan šablon. Iz toga se ne može zaključiti
  koliko je korisnik pročitao van repoa; u planu tražiti proveru razumevanja,
  ne pretpostavljati da učenje kreće od nule.

## Nalazi koji menjaju prioritete

### A. Equal-time računanje i poreklo CSV-a moraju da se razjasne

`lipschitz-comparison/analyze_equal_time.py` deli vreme GPU sweep-a sa
`sum([1,3,5,7,9]) = 25`, a zatim množi budžetom K.
Međutim, sačuvani `Pasado/experiments/colab_bab_sweep.py:bab_curve` jednom
gradi krivu do budžeta 9, sa **ukupno 9 solve-ova**, beležeći checkpoint-e
1/3/5/7/9. To je potvrđeno izvršavanjem te funkcije sa lažnim solverom
koji samo broji pozive: 9, ne 25.

Sačuvani GPU CSV ima kolone `K,seconds_per_image_per_eps`, dok aktuelni
producer piše `solves,images_averaged,seconds_per_image`. Nije utvrđeno
koja verzija ili konverzija je proizvela baš taj CSV. Zato se istorijska
vremena **ne mogu samo prepraviti množenjem sa 25/9**. Najpre povezati
raw log, kod i CSV; ako to nije moguće, ponoviti mali sweep sa direktnim
kumulativnim vremenom za svaki checkpoint.

Dodatno, agregacija uzima cenu iz eps indeksa 0 za celu mrežu. BaB vreme
uključuje pravljenje modela/grafa i čišćenje memorije po solve-u, a Pasado
kolona meri precise prolaz. Potrebno je prijaviti iste granice tajminga,
uključujući posebno setup i ponovljenu analizu. Ispitati efikasniji bezbedan
životni vek auto_LiRPA grafa, uz kontrolu memorije i validnosti granica.

Nalaz ne poništava postojeće vrednosti granica niti govori da Pasado gubi.
Znači da **15/15 pri jednakom vremenu i 14–50× vremenska prednost još nisu
zaključane tvrdnje za rad**. Provera monotonosti CSV-a ne proverava ovaj protokol.

### B. „Pokrenuo sam default” nije „pokrenuo sam najbolji CPU”

Default-i u modulu su vectorized=1, boundary=1, batched_lstsq=0,
batched_grid=0, real_cubic=0. Za referentni optimizovan CPU put eksplicitno
zaključati prva četiri na 1, real_cubic na 0.
`benchmark_lipschitz.py:run_once` nasleđuje okruženje i prepisuje samo deo
flagova. Npr. spoljni `PASADO_BATCHED_GRID=1` može da kontaminira varijantu
označenu kao original. Potrebno je definisati svih pet flagova po varijanti.

### C. Test može da ispiše FAIL i ipak vrati exit 0

Danas potvrđeno za `test_batched_grid.py`: STEPS=7/9 na arm64 imaju
razliku do 8,882e-16 i test ispisuje FAIL, ali nema neuspešnog exit koda.
STEPS=3/5 i produkcione selector kombinacije prolaze. Razdvojiti
bit-egzaktnost platforme od numeričke ekvivalencije i ispraviti status
testa. Ne menjati aritmetiku grida samo da test pozeleni bez x86 provere.

### D. „Duals je device-aware” je samo delimično tačno

`DualIntervalTensor.__init__` i `HyperDualIntervalTensor.__init__` i dalje
prave podrazumevane izvode preko `torch.zeros(shape)` bez device/dtype.
To treba uključiti u audit aktivnog ulaznog puta, pored `SimpleZono`
(`traceify`, `expand`, konverzija intervala) i konstanti transformera.
Nije potrebno portovati ceo hyper-dual/softplus/CNN API za prvi FFNN rezultat.

### E. Procenti profila imaju različite imenioce

Istorijskih 45% matmul i 1,7% gelsd odnose se na ceo run
(interval + baseline zonotope + precise), u konkretnoj batched varijanti.
Nisu procenti samo precise puta niti profil najnovije grid varijante.
Novi GPU speedup meriti za precise analizu; ukupno vreme starog CLI-ja
prijaviti odvojeno ako se koristi. cProfile vremena nisu wall-clock rezultat.

### F. Numerička bliskost nije formalni dokaz obuhvatanja

Originalni rad eksplicitno navodi ograničenje floating-point soundness-a.
To ne opravdava proizvoljnu toleranciju: CPU/GPU ugovor treba unapred
definisati, prijaviti maksimalne razlike, mesta NaN/Inf i proveriti konkretne
izvode uzorkovanjem kao dodatnu proveru. Uzorkovanje takođe nije dokaz.
Postojeći complex64 koreni mogu otežati vrlo strogu CPU/GPU toleranciju;
ne obećavati `1e-10` na celom `big` bez merenja.

Istorijski `lc_precise_big.pth` je agregat, sa neseedovanom perturbacijom.
Za novu proveru napraviti svežu CPU referencu po slici/eps/seedu,
a stari fajl koristiti kao dodatnu proveru reda veličine.

### G. Stare beleške nisu jedinstven aktuelni status

`TODO_sutra.md` govori o otvorenim PR-ovima i necommitovanom sprintu koji
su u međuvremenu završeni. `PLAN_MLSYS.md` u ranijim delovima ponavlja
završeni grid i raniju pretpostavku o njemu. `GPU_READINESS.md` sadrži
istorijske zaključke pre i posle A100 merenja. `UCENJE.md` pogrešno naziva
fit kvadratnim. Nova dnevna lista zato vodi aktuelni status odvojeno.

## Današnje izvršene provere

Naknadno je pregledan i korisnikov `TASKOVI_HIBRID.md` (T0–T8), potvrđen
tempo 2–3 h dnevno i ta lista je usklađena sa dnevnim rasporedom. Sačuvan
je način rada: korisnik implementira, asistent objašnjava i pregleda diff.

Dopuna Claude-ovog plana: 0-dim CPU skalari imaju izuzetak pri radu sa GPU
tenzorima prema [PyTorch dokumentaciji](https://docs.pytorch.org/docs/main/tensor_attributes.html).
Zato konstante u kubnom solveru nisu potvrđen cross-device kvar niti dokaz
skupog prenosa po pozivu. Menjanje konstanti mora da očuva njihove tipove.
Takođe [sync debug režim](https://docs.pytorch.org/docs/main/generated/torch.cuda.set_sync_debug_mode.html)
ne detektuje sve sinhronizacije. U T4 nedostajao je scikit-learn, koji jezgro
uvozi; dodat je u listu zavisnosti. `conv.py:38` već koristi `np.prod`.

Okruženje: postojeći `Pasado/.venv`, Python 3.13.15, torch 2.12.1,
NumPy 2.5.1, 5 torch threadova, CUDA nedostupna.

| Provera | Rezultat |
|---|---|
| `test_check_corners_tensor.py` | PASS u zadatim tolerancijama; nisu sve razlike nula |
| `test_check_nonlinear_boundary_tensor.py` | PASS |
| `test_compute_max_error_tensor.py` | PASS |
| `test_selector_paths.py` | PASS |
| `test_batched_grid.py` | FAIL za bit-egzaktnost STEPS=7/9; exit 0; produkcioni STEPS=5 prolazi |
| `experiments/test_batched_lin_reg.py` | PASS, bitwise identično u testiranim slučajevima |
| `correctness_e2e.py`, vectorized | PASS, 3 slike × 3 eps; max razlika lc 2,84e-14 |
| `correctness_e2e.py`, batched | PASS, iste konfiguracije; max razlika lc 2,84e-14 |
| `lipschitz-comparison/check_consistency.py` | PASS, 285 Pasado + 75 LiRPA redova, 48 Pasado ćelija |
| Brojanje `bab_curve(...,budget=9)` sa stub solverom | 9 solve-ova; checkpoint-i 1,3,5,7,9 |

E2E je pozvan kroz postojeći modul uz preusmeren `CSV_PATH` u privremeni
folder, eksplicitno isključene grid/real-cubic flagove i odgovarajući BATCHED
flag. Istorijski CSV-i nisu prepisani. Logovi ovog pregleda sačuvani su u
`notes/audit_2026-09-19/`. Nisu rađena nova GPU merenja ni puni vremenski sweep.

## Dogovoreni smer i rok

Prema beleškama 22/23.8. Sašin prioritet je `big` i veće mreže, prvi hibrid
sa CPU regresijom. Kratka poruka 12.9. najavljuje rezultat do kraja septembra
i pita za publikaciju/saradnike; noviji odgovor nije sačuvan u ovom workspace-u.
To je poslednja dokumentovana komunikacija, ne tvrdnja da druge nije bilo.

Datum roka je potvrđen 19.9. na [MLSys 2027 CFP-u](https://mlsys.org/Conferences/2027/CallForResearchPapers):
30.10.2026. CFP i [stranica rokova](https://mlsys.org/Conferences/2027/Dates)
nisu usklađeni oko tačnog sata; interni rok 29.10. izbegava oslanjanje na njega.
Do konferencijskog datuma ima 41 kalendarski dan. Spremnost za submission
zavisi od rezultata, širine evaluacije i procene koautora; ovaj plan je ne garantuje.
