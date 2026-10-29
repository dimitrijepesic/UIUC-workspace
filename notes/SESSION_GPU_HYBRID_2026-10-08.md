# GPU hibrid - završena korektnost, 8.10.2026.

Privatna sesijska beleška. Ne ulazi u Pasado commit.

## Šta je urađeno

### 1. CPU optimizacije i pouzdana osnova

- Precizni transformer je vektorizovan i batch-ovan.
- Ranija merenja: približno 3,2-3,6x na 3/4/5layer i 1,65x na big.
- Seedovana CPU regresija ostala je bit-identična sa prethodnim ponašanjem za
  sve četiri mreže.
- Dodati su testovi za grid, selector putanje, uglove, nelinearnu granicu,
  maksimalnu grešku, batched regresiju, CLI i end-to-end rezultat.

### 2. Device i dtype higijena

- Tenzori koji se kombinuju sada nasleđuju uređaj i dtype.
- Popravljeni su `traceify`, `Zonotope.expand`, dual/hyper-dual intervali,
  legacy corner/boundary putanje i konstrukcija regresionih matrica.
- MPS koristi float32; CPU i CUDA podrazumevano koriste float64.
- Sve 16 selector kombinacije prolaze na MPS-u sa kompleksnim i realnim
  kubnim putem.

### 3. Hibrid CPU/GPU

- Velike tenzorske i matrične operacije mogu da ostanu na GPU-u.
- `torch.linalg.lstsq(driver="gelsd")` ostaje na CPU-u jer CUDA `gels` nije
  bezbedan za rank-deficient kutije.
- Rezultat regresije se vraća na originalni uređaj.
- Puni MPS put prolazi za 3layer, 4layer, 5layer i big.

### 4. Kubni rešavač

- Stari kompleksni Cardano put nije numerički prenosiv između CPU-a i GPU-a.
- GPU ostavlja male imaginarne ostatke; `torch.isreal` zatim odbacuje korene.
- Realna Viète formula uklanja kompleksnu aritmetiku u domenu sa tri realna
  korena.
- CPU testovi prema autograd uzorkovanju potvrđuju validnost realnog puta.
- Pomeraj realnog naspram starog kompleksnog CPU rezultata je mali
  (najviše oko 8,3e-6 relativno), ali eksplicitno dokumentovan.

### 5. Testovi i verifikaciona infrastruktura

- Svih devet glavnih testova prolazi.
- Mutation harness ubija svih deset mutanata.
- Harness više ne može da vrati exit 0 kada mutant preživi ili kada je izabran
  nepoznat/prazan skup mutanata.
- Privremeni test repo više nema writable symlink ka pravom `Section_5_4`.
- CLI validira broj slika, splitove, epsilon indekse, seed, uređaj i dtype.
- Nestandardni runovi ne mogu da pregaze referentne rezultate.
- Experiment skripte ne menjaju tracked logove bez eksplicitnog `--out`.
- Sve argparse skripte rade sa `--help`.

## Commitovi pre finalnog A100 rezultata

- `f51f977` - `fix: harden GPU verification paths`
- `c2a2c65` - `docs: normalize verification artifacts`

Oba su na `fork/gpu-hybrid`.

## A100 quick i full rezultat

Hardver: NVIDIA A100-SXM4-80GB.

Quick matrica:

- 5 slika;
- epsilon indeksi 0, 8 i 15;
- 28/28 PASS.

Full matrica:

- 3layer, 4layer, 5layer i big;
- 30 tačno klasifikovanih slika;
- svih 16 epsilona;
- 28/28 PASS;
- log se završava sa `ALL PASS`.

### Najvažniji brojevi full matrice

| Provera | Najveća relativna razlika |
|---|---:|
| real-cubic float64 precise CPU/CUDA | 1,786e-15 |
| real-cubic float64 zonotope CPU/CUDA | 4,792e-16 |
| real-cubic float64 interval CPU/CUDA | 5,517e-16 |
| real-cubic float32 precise CPU/CUDA | 8,318e-7 |
| ponovljen CUDA run | 0 |
| full naspram precise-only CUDA | 0 |

Pragovi:

- float64: 1e-9;
- float32: 1e-5.

Prva verzija protokola je pogrešno koristila 1e-9 i za float32. To je ispod
praktične preciznosti float32. Neuspeh je sačuvan kao nalaz, prag je opravdano
promenjen na standardni float32 `assert_close` red veličine i matrica je
ponovljena.

### Negativna kontrola

Stari kompleksni rešavač divergira između CPU-a i A100:

- 3layer: 14,39%;
- 4layer: 33,42%;
- 5layer: 48,72%;
- big: 63,78%.

PASS za ove redove znači da je očekivana divergencija uspešno otkrivena. Ne
znači da je kompleksni rešavač prihvatljiv.

## Šta je dokazano

- Real-cubic float64 CPU i A100 se slažu približno do mašinske preciznosti.
- CUDA rezultat je determinističan.
- `--precise-only` ne menja precizni rezultat.
- Hibridna device putanja radi za sve postojeće FFNN mreže.
- Kompleksni kubni put nije bezbedna osnova za CPU/GPU reproduktivnost.

## Šta još nije dokazano

1. Brzina nije kontrolisano izmerena. Notebook je correctness matrica, ne
   benchmark.
2. Full rezultat poredi proseke preko 30 slika po epsilonu. Potreban je
   per-image sweep da se prijavi maksimalna individualna CPU/GPU razlika.
3. Nisu još upoređeni originalni CPU, optimizovani CPU, neoptimizovani GPU,
   optimizovani hibrid i oracle na istoj instanci.
4. Nisu testirane mreže veće od big niti CNN put.
5. Za rad treba ponoviti performanse na još jednom tipu GPU-a ili jasno
   ograničiti tvrdnju na A100.

## Sledeći tehnički koraci

1. Sačuvati per-image `lc` za sve četiri mreže, 30 slika i 16 epsilona.
2. Izmeriti end-to-end vreme na istoj Colab instanci:
   - originalni CPU;
   - optimizovani CPU;
   - neoptimizovani GPU;
   - optimizovani hibrid.
3. Uključiti warm-up, više naizmeničnih ponavljanja, CPU `gelsd`, transfere i
   `cuda.synchronize()`.
4. Napraviti tabelu po sloju: dimenzije, broj generatora, matmul, CPU solve,
   transferi i ukupno vreme.
5. Tek ako postoje netrivijalni crossover-i, razmatrati cost model i scheduler.

## Jednostavan status za Sašu

Završena je hibridna CPU/GPU implementacija Pasada za FFNN put. GPU radi velike
tenzorske operacije, dok rank-sensitive `gelsd` ostaje na CPU-u. Na A100 je
pokrenuta puna korektnosna matrica za četiri mreže, 30 slika i 16 epsilona.
Realni kubni rešavač daje CPU/CUDA slaganje do 1,8e-15 u float64, CUDA je
deterministična, a full i precise-only rezultati su identični. Stari kompleksni
rešavač divergira 14-64% i zato nije prihvatljiv za GPU put. Sledeći korak je
kontrolisano merenje ubrzanja i per-image korektnost; ubrzanje se još ne tvrdi.
