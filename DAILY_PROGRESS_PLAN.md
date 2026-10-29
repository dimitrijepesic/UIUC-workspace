# Dnevni plan — 2–3 sata za Pasado projekat

**Dogovoren tempo: 2–3 sata dnevno.** Prvi cilj je hibrid CPU/GPU za `big`
i rezultat koji možeš da pokažeš Saši. Procena je **12 radnih sesija
(24–36 sati), plus 2–3 sesije rezerve** za okruženje i numeriku.

[Claude-ovi T0–T8](TASKOVI_HIBRID.md) ostaju tehnička lista taskova.
Ovaj dokument ih raspoređuje po danima; nisu dva odvojena posla.
[Audit projekta](notes/PROJECT_AUDIT_2026-09-19.md) beleži šta je već urađeno,
a [dnevna evidencija](notes/DAILY_LOG.md) služi za kratke zapise i izveštaje.

Ti implementiraš i učiš kroz kod; asistent objašnjava, daje nagoveštaj i
pregleda diff. Ovim planiranjem nije implementiran hibrid niti napravljen
njegov razvojni branch. T8 je odložen i ne blokira prvi rezultat.

## Ritam jednog dana

- 15 min: pročitaj prethodni log i odredi jedan glavni ishod.
- 30–45 min: pročitaj baš deo koda/rada koji je potreban za taj korak.
- 60–100 min: implementacija, eksperiment ili otklanjanje jedne blokade.
- 15–20 min: ciljna provera, sačuvan dokaz i zapis sledećeg koraka.

Duži benchmark-i imaju posebno mašinsko vreme. Ako je dan kratak, završi
jedan korak i upiši stanje; ne nadoknađuj propušteno gomilanjem tri taska
sutradan. Ako već razumeš deo, preskoči čitanje i sačuvaj praktičan dokaz.

Jedan smislen commit kad celina prođe relevantne provere. Delimičan rad
nije razlog za lažni „gotovo”. Posle najviše tri dana pošalji međurezultat,
čak i ako je zaključak izolovana greška ili negativno merenje.

## Prvih 12 dana — isti T0–T7, razloženi

### D01 / T0a — Upoznaj referentni put i proveri početno stanje

- [ ] U repou `Pasado/` napravi `gpu-hybrid` od `sprint-optimizations`.
- [ ] Pročitaj Algoritam 1 (Pasado rad, str. 12–13) uz `sigmoid_prime_product_tensor`; zapiši šta su A/B/C, greška i generatori. Ograniči čitanje na 45 min.
- [ ] Pregledaj današnje audit logove i pokreni jedan mali CPU smoke run sa eksplicitnim flagovima i `--no-save`. Zabeleži poznati grid STEPS=7/9 izuzetak; ne tumači exit 0 kao PASS.

**Gotovo:** znaš tačno koji put portuješ, branch postoji i početno stanje je zapisano.
**Dokaz:** 5–10 rečenica u review beleškama + komanda/log; ne moraš ponovo ceo projekat da čitaš.

### D02 / T0b — Zaključaj konfiguraciju i reprodukciju

- [ ] Sredi prijavljivanje neuspeha grid testa i razdvoji poznatu arm64 bit-egzaktnost od produkcionog STEPS=5. Ne menjaj formulu bez razloga.
- [ ] Eksplicitno definiši svih pet flagova za original i najbolji CPU; ukloni nasleđivanje eksperimentalnih flagova iz benchmark varijanti.
- [ ] Pusti dva mala neseedovana runa kao ilustraciju šuma, zatim pripremi isti seed za ponovljivo poređenje. Zapiši da taj šum ne određuje big toleranciju.

**Gotovo:** zna se koja konfiguracija je merena i test ne prijavljuje lažan uspeh.
**Dokaz:** mali diff + test log + zapis razlike između ponavljanja.

### D03 / T1 — Dodaj kontrolisani ulaz za eksperiment

- [ ] Dodaj `--device`, `--num-images` i seed; zapiši image IDs, eps i sve flagove.
- [ ] Omogući precise-only run; prebaci mrežu/ulaz na uređaj i uredi CUDA granice tajminga.
- [ ] Na CPU proveri da ista slika/eps/seed daje isti rezultat pre/posle promene. GPU još može da čeka T2/T3.

**Gotovo:** kratka komanda koja ponavlja isti CPU problem i omogućava budući CUDA run.
**Izveštaj U01:** „Zaključao sam referencu i konfiguraciju; CPU regresija [status], poznati arm64 izuzetak [obim]. Sledeći korak je device put.”
Ako ne završiš sve opcije za 3 h, podeli T1 na još jedan dan i ipak pošalji U01.

### D04 / T2 — Device higijena aktivnog puta

- [ ] Sredi `DualIntervalTensor` pomoćne izvode i `SimpleZono.traceify/expand/HyperDualIntervalToDualZonotope`.
- [ ] Pregledaj konstante u `inverse_poly_tensor`; sačuvaj numeričke tipove. 0-dim CPU skalar nije isto što i CPU vektor.
- [ ] Pokreni pogođene CPU testove i kratku referencu iz D03.

**Gotovo:** CPU rezultat očuvan, pregledan mali diff i spisak mesta koja prave tenzore.
**Dokaz:** test log; ne tvrditi da GPU radi pre CUDA provere.

### D05 / T3 — CPU solver unutar hibrida

- [ ] Dodaj prenos regresionog problema GPU→CPU, isti `gelsd`, koeficijente nazad; CPU put bez suvišnih prenosa.
- [ ] Proširi test na uređaj kao parametar; full-rank, degenerisane i mešovite kutije.
- [ ] Proveri CPU regresiju i napiši šta ulazi/izlazi iz mosta, sa oblicima/dtype-om. CUDA test lokalno jasno prijavljuje SKIP.

**Gotovo:** most implementiran i CPU provere prolaze; CUDA rezultat još nije pripisan.
**Dokaz:** diff + relevantni testovi. `gels` i real-cubic ostaju van osnovnog porta.

### D06 / T4 — Prvo udaljeno GPU okruženje

- [ ] Podesi Modal ili već dostupan CUDA resurs i sačuvaj stvarne verzije, CPU/GPU i threadove.
- [ ] Uključi potrebne zavisnosti, uključujući scikit-learn; pripremi slanje koda bez venv/profila i preuzimanje logova.
- [ ] Pokreni `colab_gpu_profile.py` kao proveru okruženja, pa mali test novog regresionog mosta ako ostane vremena.

**Gotovo:** log sa potvrđenim CUDA hardverom i uspešnim mikroproverama.
**Izveštaj U02:** „Minimalni hibrid je implementiran; [lokalne/CUDA provere] prolaze. Sledeće proveravam numeriku celog prolaza.”
**Rezerva:** ako pristup zapne, D07 možeš spremati na CPU; u izveštaju jasno ostaje blokada.

### D07 / T5a — Napravi CPU/GPU correctness harness

- [ ] Definiši mali fiksni skup slika/epsilona i re-seed pre svakog odgovarajućeg forward-a.
- [ ] Zapiši kriterijum pre GPU rezultata: apsolutna/relativna tolerancija, lb/ub/lc, dtype/oblici i NaN/Inf. Predlog 1e-9 je hipoteza za proveru, ne obećanje.
- [ ] Sačuvaj CPU referencu po slučaju i proveri ponovljivost; koristi isti produkcioni forward put da kopije ne odlutaju.

**Gotovo:** harness razlikuje numerički FAIL, infrastrukturni FAIL i SKIP.
**Dokaz:** CPU referentni CSV i tekst ugovora. Stari prosečni `.pth` je pomoćna provera.

### D08 / T5b — Prvi ceo CUDA prolaz

- [ ] Počni od jednog sloja, zatim `3layer`: jedna slika × jedan epsilon.
- [ ] Uporedi sa CPU-om iz istog udaljenog okruženja; otkloni prvu device/dtype razliku.
- [ ] Proširi na 3 slike × 3 epsilona kad mali slučaj prođe; sačuvaj log i sve izuzetke.

**Gotovo:** prvi kompletan GPU precise run sa numeričkom proverom, ili minimalan reproducer prve blokade.
**Dokaz:** correctness CSV; bez dugog benchmark-a dok ovo ne prođe.

### D09 / T5c — Numerika na `big` i sync pregled

- [ ] Pokreni `big` na 2–3 para, zatim proširi mali skup; isti checkpoint/seeds na CPU/GPU.
- [ ] Ako se ugovor naruši, pronađi prvi sloj gde se razlika pojavljuje. Proveri uzorkovane autograd izvode kao dodatni sanity check, ne formalni dokaz.
- [ ] Pregledaj uočene sync-ove; `warn` režim nije potpun i dodatni sync nije automatski bug.

**Gotovo:** numerički status za `big` sa maksimalnom razlikom i opisanim ograničenjima.
**Izveštaj U03:** „Na [N slučajeva] big CPU/GPU daje [razlika/status]. [Problem ako postoji.] Sledeće merim vreme pod istim uslovima.”

### D10 / T6a — Pilot CPU naspram hibrida

- [ ] U istoj udaljenoj instanci naizmenično meri CPU i hibrid, sa warm-up-om i najmanje tri ponavljanja na malom fiksnom skupu.
- [ ] Uključi CPU solve i prenose; koristi sinhronizaciju pre/posle tajmera i isti broj CPU threadova.
- [ ] Sačuvaj medijanu/raspon, ubrzanje i peak memoriju. Mac rezultat označi kao zasebnu platformu ako ga meriš.

**Gotovo:** prvo fer vreme, i ako je GPU sporiji.
**Dokaz:** sirovi CSV + pilot tabela; ne mešati ukupno CLI vreme i precise latency.

### D11 / T6b — Objasni i ponovi rezultat

- [ ] Snimi mali precise-only profil: matmul, redukcije, cat/expand, solver, prenosi; broj generatora po sloju.
- [ ] Pokreni veći fiksni skup za `big`, cilj 30 slika × tri unapred izabrana epsilona, bez profajlera za finalno vreme. Obeleži nepotpun sweep ako ne završi.
- [ ] Proveri numeriku, rasipanje vremena i ograničenja memorije; napiši kratko zašto hibrid pomaže ili ne pomaže.

**Gotovo:** proširen rezultat koji nije zasnovan na jednom srećnom runu.
**Dokaz:** e2e tabela i odvojeno razlaganje vremena. Profil ne zamenjuje benchmark.

### D12 / T7 — Sastavi prvi hibridni izveštaj

- [ ] Popuni `HYBRID_RESULTS.md`: problem, commit/config, hardver, correctness, vreme, memorija i ograničenja.
- [ ] Napravi jednu tabelu i jedan graf iz sačuvanih podataka; napiši 5–8 rečenica za Sašu.
- [ ] Uz rezultat pripremi pitanja o GPU-u grupe i daljem obimu. Odgovor mentora ne blokira dokumentovanje/reprodukciju rezultata.

**Gotovo:** paket spreman za slanje, sa podacima iz kojih brojke mogu da se reprodukuju.
**Izveštaj U04:** rezultat hibrida, razlog dobitka/gubitka, ograničenje, sledeći eksperiment.
Slanje radiš ti; pripremljen nacrt nije isto što i poslata poruka.

## Posle hibrida — predlog, revidirati posle U04

Ovo je okvir za sledeće dnevne taskove, ne dodatna obaveza pre T7.
Prioriteti su Sašina ablacija i pouzdana evaluacija, zatim veće mreže.
Ako hibrid ne ubrza `big`, prvo odluči prema profilu da li sledeći dan ide
na jednu ciljanu optimizaciju ili na dokumentovanje negativnog rezultata.

## Blok 5 — zaključaj baseline i traženu ablaciju

### D13 — Razjasni BaB vreme i poreklo podataka

- [ ] Poveži `bab_sweep_gpu.csv` sa originalnim logom/verzijom; dokumentuj nepoznato ako izvor nije dostupan.
- [ ] Testom brojača potvrdi razliku između ukupnih solve-ova, checkpoint-a i aktivnih domena. Pri K=9 aktuelni algoritam pravi 9 solve-ova i završava sa 5 živih domena.
- [ ] Uvedi direktno kumulativno vreme po checkpoint-u i jedinstvenu CSV šemu; ukloni zavisnost nove analize od pretpostavke `vreme/25`.

**Gotovo:** mali ponovljivi BaB primer sa brojačem i vremenom; istorijske tabele jasno označene kao preliminarne.

### D14 — Poravnaj trošak setup-a i životni vek baseline-a

- [ ] Za Pasado i auto_LiRPA definiši isti problem, image IDs, preprocessing, dtype, seeds i timing boundary.
- [ ] Izmeri posebno load/trace/setup, bound computation i cleanup; proveri da li se auto_LiRPA model/graf može bezbedno ponovo koristiti bez curenja memorije ili starih bounds.
- [ ] Proveri jedan slučaj protiv starog one-pass rezultata i autograd uzorka. Ako je rekonstrukcija neophodna, prijavi njen trošak otvoreno.

**Gotovo:** jedna mala fer tabela sa cold i steady-state režimom gde je primenljivo. Zaključak važi za testirani auto_LiRPA setup.

### D15 — Original / CPU-opt / hibrid / auto_LiRPA

- [ ] Pokreni sve četiri varijante na `3layer`, `5layer` i `big`, početno jedan epsilon i isti mali skup slika; proširenje pokreni tek kad tajming i numerika prođu.
- [ ] Prikaži zasebno vreme i granicu. Originalni i optimizovani Pasado treba da ostanu numerički usklađeni uz isti seed.
- [ ] Odgovori na Sašinu hipotezu: da li optimizacije zaista menjaju redosled brzine u ovom setup-u? Ako ne, prijavi to.

**Gotovo:** reproducibilna ablaciona tabela, bez pretpostavljenog željenog ishoda.
**Izveštaj 5:** rezultat ablacije i šta je promenjeno u metodologiji equal-time merenja.

## Blok 6 — tesnoća za stvarno vreme

### D16 — Mali direktni equal-time eksperiment

- [ ] Izaberi jednu mrežu/epsilon/sliku i unapred zadate vremenske budžete. Sačuvaj najbolju završenu granicu sa stvarnim elapsed vremenom.
- [ ] Pasado uniformne split runove i LiRPA progresivni BaB označi tačno: nezavisni fiksni K runovi nisu ista izvršna politika kao jedan anytime run.
- [ ] Uključi sve utrošeno vreme u online budžet; ako radiš offline frontier iz nezavisnih runova, tako ga nazovi. Proveri iz tabele nekoliko tačaka ručno.

**Gotovo:** jedna kriva tesnoća–vreme kojoj se svaki punkt može pratiti do raw merenja.

### D17 — Ponovi matricu uskih mreža

- [ ] Zaključaj 3/4/5layer × eps indekse 0–4 × istih 30 slika, uz definisane budžete i ponavljanja.
- [ ] Pokreni resumabilni sweep. Ključ za nastavak mora sadržati celu konfiguraciju, image ID, budžet, seed i reviziju, ne samo mrežu/epsilon.
- [ ] Pregledaj coverage, timeout/OOM i najgore slučajeve; sačuvaj nedostajuće ćelije umesto njihovog prećutkivanja.

**Gotovo:** matrica ili eksplicitna lista preostalih runova sa procenom mašinskog vremena. Duge runove ne mešati sa drugim benchmark-ovima na istom resursu.

### D18 — Dodaj `big` u equal-time evaluaciju

- [ ] Kreni od istih tri epsilona kao D12 i malog pilota, zatim 30 slika ako resursi dozvoljavaju.
- [ ] Sačuvaj vreme, granice i vršnu memoriju; neuspeh baseline-a pri nekoj veličini je podatak uz isti unapred definisan limit.
- [ ] Uporedi nove krive sa starim tvrdnjama 15/15 i 14–50×; zadrži samo brojke potvrđene novim protokolom.

**Gotovo:** figura/tabela sa jasnim obimom i konzistentnim vremenskim budžetom.
**Izveštaj 6:** nova equal-time slika, šta ostaje od stare hipoteze i šta je novo na `big`.

## Blok 7 — veće mreže i objašnjenje granica sistema

### D19 — Omogući promenu širine bez dupliranja forward-a

- [ ] Napravi parametrizovan niz affine/sigmoid slojeva koji obuhvata `big`.
- [ ] Proveri istovetnost učitanog `big` checkpoint-a i rezultata sa postojećim putem.
- [ ] Izaberi dve veće širine prema memorijskom pilotu, npr. 1536 i 2048 ako staju, uz fiksnu dubinu i seed.

**Gotovo:** konfigurabilan systems workload i memorijska procena pre dugog runa.
**Ograda:** sintetički/random-weight workload je dovoljan za karakterizaciju izvršavanja; ne koristiti ga kao dokaz kvaliteta granica na treniranim mrežama.

### D20 — Izmeri skaliranje i broj generatora

- [ ] Na istom hostu meri CPU/hibrid za `big` i izabrana dva veća oblika; kreni od jednog slučaja.
- [ ] Beleži vreme, broj generatora po sloju, peak CPU/GPU memoriju i numerički status.
- [ ] Napravi graf širina→vreme/memorija; izdvoji tačku na kojoj hibrid pomaže ili postaje ograničen memorijom.

**Gotovo:** izmereni trend sa označenim sintetičkim i treniranim slučajevima; bez ekstrapolacije preko OOM tačke.

### D21 — Proveri generalnost na drugom hardveru ili opsegu

- [ ] Ako imaš drugi CUDA uređaj, ponovi mali isti workload na njemu, sa njegovim CPU baseline-om.
- [ ] Ako ga nemaš, ispitaj još jednu dubinu/oblik na postojećem uređaju i jasno ograniči hardversku tvrdnju.
- [ ] Sažmi kada CPU/hibrid pobedi i koliki deo odluke objašnjavaju oblik, FP64, prenosi i rast generatora.

**Gotovo:** rezultat generalnosti ili jasno dokumentovano ograničenje.
**Izveštaj 7:** veće mreže, memorija, slučajevi gde hibrid pomaže i gde ne pomaže.
**Odluka:** scheduler ulazi tek u novi plan ako merenja pokazuju dodatnu priliku koju statički hibrid propušta.

## Blok 8 — paket za saradnike i tekst

### D22 — Zaključaj tvrdnje i strukturu teksta

- [ ] Napravi tabelu tvrdnja→eksperiment→raw CSV→ograničenje. Odvoji sistemsko ubrzanje od tesnoće i korektnosti.
- [ ] Napiši problem, pristup i 2–3 glavna nalaza; odvojeno navedi negativne rezultate.
- [ ] Izaberi zajedno sa mentorom obim koji podaci stvarno podržavaju; odluka o venue-u ne blokira završavanje izveštaja.

**Gotovo:** kratak kostur rada/izveštaja sa svim glavnim tabelama, bez neizmerenih doprinosa.

### D23 — Figure i nezavisna reprodukcija

- [ ] Generiši figure iz sačuvanih CSV-a: CPU/hibrid vreme, ablacija, tesnoća–vreme, skaliranje/memorija.
- [ ] Ponovi jedan reprezentativan eksperiment iz čistog procesa koristeći samo dokumentovane komande i zaključane zavisnosti.
- [ ] Proveri da naziv varijante, commit, hardver i broj slika odgovaraju figurama; proveri šta je potrebno za deljenje artefakta.

**Gotovo:** izvorni podaci + skripte za figure + kratko uputstvo koje drugi može da prati.

### D24 — Sastavi prvi celovit nacrt

- [ ] Spoji dosadašnje blok izveštaje u uvod, dizajn, numerički ugovor, metodologiju, rezultate i ograničenja.
- [ ] Označi nedostajuće related-work/review stavke i dodeli konkretna pitanja saradnicima.
- [ ] Pripremi paket za slanje: kratak sažetak, 3–4 figure, link na kod i nacrt teksta.

**Gotovo:** prvi celovit nacrt za komentare, ne obećanje submission-ready rada za jedan dan.
**Izveštaj 8:** šta je dokazano, šta ostaje i konkretna pitanja za odluku o daljem radu.

## Vremenski okvir i rezerve

Ako D01 počne u ponedeljak **21.9.**, uz pet radnih dana nedeljno:

| Kontrolna tačka | Orijentacioni datum |
|---|---|
| D03 — CPU osnova i T1 | 23.9. |
| D06 — hibridni kod i GPU okruženje | 28.9. |
| D08 — prvi mali CUDA correctness | 30.9. |
| D09 — `big` correctness | 1.10. |
| D12 — prvi hibridni izveštaj | 6.10. |
| D15 — ablacija | 9.10. |
| D18 — equal-time | 14.10. |
| D21 — skaliranje | 19.10. |
| D24 — nacrt | 22.10. |
| Rezerva za debugging, komentare i završavanje | 23–29.10. |

MLSys 2027 datum roka je **30.10.2026.**, potvrđen na
[zvaničnom CFP-u](https://mlsys.org/Conferences/2027/CallForResearchPapers).
Plan ostavlja malo rezerve: GPU pristup i numerika mogu da pomere datume.
Ako D09 kasni, zadrži CPU/hibrid validaciju, fer baseline i tekst; suzi broj
dodatnih mreža/hardvera. Ako D12 još nema pouzdan hibrid, sa mentorom
reproceni obim/rok umesto obećavanja punog scheduler projekta.

Kraj septembra iz prethodne poruke Saši je tesan cilj za kompletan hibridni
rezultat pri 2–3 h dnevno. U ovom rasporedu tada očekujemo prvi mali CUDA
prolaz, a izveštaj za `big` početkom oktobra. Novi okvir javi kroz U01;
ako dani idu brže, pomeri se unapred tek kad kriterijumi prođu.

## Pravila evaluacije koja važe kroz ceo plan

- Svaki run beleži Git reviziju, kompletne flagove, checkpoint, image IDs,
  eps, splits/budžet, seed, dtype, device, CPU threadove i verzije.
- „Originalni Pasado” nije isto što i `forward_zono`: prvi je precise
  transformer bez tvojih optimizacija, drugi je druga apstraktna analiza.
- Hibrid se poredi sa najboljim potvrđenim CPU putem na **istom hostu**.
  Mac rezultat može kao dodatna odvojena platforma.
- Numerika se proverava po slučaju; prosek ne sme da sakrije jedan pogrešan
  bound. `allclose` i autograd uzorkovanje nisu formalni dokaz soundness-a.
- Izbor `gelsd` ostaje isti. GPU-only kolona je N/A dok ne postoji validiran
  ekvivalentan solver; ne praviti je nasilnom zamenom za `gels`.
- Medijana i varijacija uz warm-up/ponavljanja za nove rezultate; istorijske
  minimum-of-N tabele označiti njihovim originalnim protokolom.
- Negativan rezultat, timeout ili reprodukovan kvar mogu biti završen task
  ako odgovaraju na definisano pitanje i ostave dokaz.

## Odloženo dok podaci ne opravdaju proširenje

Dinamički scheduler i oracle, CPU multiprocessing, rank-dispatch za GPU
regresiju, Triton/JAX/MAGMA, `torch.compile`, uključivanje real-cubic puta,
redukcija šumnih simbola, port ODE/Black–Scholes/CNN i povratak na Neptune
implementaciju. To su zasebni istraživački poslovi; nisu uslov za prvi
rezultat koji je već dogovoren sa mentorom.
