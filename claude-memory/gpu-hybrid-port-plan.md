---
name: gpu-hybrid-port-plan
description: "Plan za hibrid CPU(gelsd)/GPU port Pasado precise puta za big mrežu (dogovoren 23.8.2026), 5 koraka, 3-4 dana; kreće od vec_batched_grid zbog .item() sync-ova"
metadata: 
  node_type: memory
  type: project
  originSessionId: 249248a7-b478-494b-a818-87272b7c5590
  modified: 2026-09-19T20:49:29.128Z
---

Glavni zadatak od 23.8.2026 (Sašina odluka, videti
[[misailo-feedback-2026-08-22]]): hibrid port Pasado precise puta za `big`.
Postojeća podloga u repou: `../PLAN_MLSYS.md` (Faze 1–2 = ovo),
`Pasado/results/optimization_section_5_4/GPU_READINESS.md` (spisak
device/dtype problema) i `BIG_PROFILE_ANALYSIS.md`,
`experiments/colab_gpu_profile.py` (GPU-spremni delovi na Colabu, sync obrazac),
`experiments/colab_rank_deficient_lstsq_test.py` (gels nebezbedan — zato gelsd
ostaje na CPU).

Koraci:
0. ✅ 15.9.2026: sprint rad commitovan na grani `sprint-optimizations` (3 commita);
   port ide na granu `gpu-hybrid` od nje.
1. Osnova = varijanta `vec_batched_grid` (sva tri flaga:
   PASADO_VECTORIZED_PRECISE, PASADO_BATCHED_LSTSQ, PASADO_BATCHED_GRID), NE
   `vec_both_batched`: stari `get_linspace` (precise_transformer.py:64-65) ima
   2 `.item()` po neuronu = 7,8M sync-ova na big → ubija GPU. Test:
   `forward_mode_tensorized_src/test_batched_grid.py`.
2. Device higijena: env `PASADO_DEVICE` (cpu/cuda); u `Section_5_4/get_lipschitz.py`
   težine + ulazni zonotopi na uređaj; 39 factory poziva bez `device=` u
   `SimpleZono.py` + `precise_transformer.py`. Duals/HyperDuals već device-aware.
   Razvija se lokalno sa PASADO_DEVICE=cpu (nema lokalnog GPU-a).
3. Hibrid = samo `lin_reg_tensor_batched` (precise_transformer.py:205):
   `.cpu()` → lstsq gelsd → `.to(device)`. Matrice [n,25,3], transfer zanemarljiv.
4. Korektnost na Colab A100 fp64: dimni test pa puni big; poredi sa
   `results/lc_precise_big.pth`. Numerički ugovor unapred: tolerancija
   (npr. max rel. razlika ≤1e-10), ne "bit-identično" (cuBLAS redosled redukcija).
5. Merenje end-to-end big CPU-najbolja vs hibrid, `torch.cuda.synchronize()`
   oko svakog merenja. Očekivani plafon 2–3× (≈70% vremena GPU-friendly).
   Rezultat → poruka Saši.

**19.9.2026: korisnik krenuo sa portom i sam kuca kod** (Claude objašnjava i
pregleda diff, nagoveštaj pre rešenja). Taskovi T0–T8 sa checkbox-ovima su u
`~/prakse/UIUC/TASKOVI_HIBRID.md`. Redosled promenjen: prvo samo vrući put
(~10 mesta: `SimpleZono.py:19,75,328`, `Duals.py:15,17`,
`precise_transformer.py:216-232`, ulaz u `get_lipschitz.py`), pa Modal i prvi GPU
broj; mehanički sweep ostalih ~45 factory poziva je T8 (odloženo). Provereno
tog dana: `Duals.py` NIJE device-aware (`:15,17` + assert na `:19` pada prvi);
`get_lipschitz.py` ne seed-uje `np.random`, pa poređenje sa
`lc_precise_big.pth` važi samo do run-to-run šuma, a tesan ugovor (≤1e-9) samo
u seed-ovanom harness-u po uzoru na `correctness_e2e.py`; 0-dim CPU konstante u
`inverse_poly_tensor` su legalne na CUDA (tvrdnja #2 u GPU_READINESS nije merena).

Procena: 3–4 radna dana. Male mreže/get_linspace za CPU: deprioritizovano.
Imati u vidu: Saša želi mreže veće od big → ne hardkodovati dimenzije.
Videti [[gelsd-profiling-result-2026-08-22]], [[pasado-code-review-plan]].

**Stanje 25.10.2026:** koraci 1–5 završeni (A100 korektnost 28/28, realni Viète
kubni rešavač je osnova; 22.10 ispravljena 2 upstream baga, commit `b526762`,
PR grana `fix-zonotope-neg-and-boundary` u `Pasado-original`). 24.10: batching po
`n_splits` (`batched_splits.py`, necommitovano) daje 7× na malim mrežama ali
1,6× na `big`, jer CPU `gelsd` nosi 84–88% vremena. Sledeći zadatak: GPU zamena
za `gelsd` (zatvorena forma za 5×5 Kartezijev grid); brief je
`~/prakse/UIUC/notes/PROMPT_GPU_REGRESSION_2026-10-25.md`. Posle toga: kompajleri
(Inductor/TVM/XLA) na `big`, `n_splits=32`.

**Stanje 25.10.2026 uveče (poređenje solvera):** pre-registrovano pravilo
(`385397e`, `experiments/compare_lstsq_solvers.py`), v1 → Winner G zbog overflow
baga u zatvorenim formama (|y|=1e300); popravka `077ab86` (skaliranje stepenom
dvojke), v2 → A2 i H izjednačeni posle R3, čeka se R4 (A100 timing) + CUDA box
nivo za P. Rezultati: `Pasado/results/lstsq_comparison_2026-10-25{,_v2_scaled}/`
(zajednički README u v2). U f32 gelsd odseca 129/2000 pravih boxova → pravilo
ranga bitno na MPS. Usput: f32 koren kubne jednačine potcenjuje D
(`experiments/reproduce_f32_root_shortfall.py`). Sledeće: Colab notebook za
`--level timing --device cuda` i `--level box --device cuda`.

**Stanje 28.10.2026 (etapa zatvorena):** A2 konačni solver (v3, pivot-centriran z,
`500ae16`). `planar_fit.py` u batched putu iza prekidača: `PASADO_CLOSED_FORM_FIT`
(eager), `+ PASADO_COMPILED_FIT` (Inductor, opt-in zbog hladnog starta 28–31 s / topao
4,3 s), `PASADO_AUTO_PLANE_FIT` (opt-in, CUDA: gelsd ispod 512 f64 / 640 f32, A2 iznad;
A100 potvrda `ae7c90a`). Podrazumevano i dalje gelsd. `big` S=64 ~13–15,6× naspram
sekvencijalnog. Sledeći task: kompajliranje celog `precise_sigmoid` (plan u
`notes/PLAN_COMPILE_PRECISE_SIGMOID_2026-10-28.md`; šum kao tensor ulaz je preduslov).

**Stanje 28.10.2026 uveče (kompajliranje celog `precise_sigmoid`):** šum na A je tensor
ulaz `precise_sigmoid(z, noise)` (`8e3d0df`), testovi šuma/NumPy stanja i `A != 0` + mutanti
M27-M29 (`a57b560` = HEAD = `fork/gpu-hybrid`). Lokalna CPU analiza
(`~/prakse/UIUC/research_notes/compile_precise_sigmoid/`): sa eager A2 unutra 1 graf, 0
prekida, fullgraph ok, Inductor CPU 17-22 kernela, izlazi do 2,4e-15 rel. Sledeće (čeka
odobrenje): A100 sveska na `a57b560` (eager vs kompajliran, S = 1/8/32/64, kerneli, fuzije,
rekompilacije, cold/warm, break-even, korektnost), pa TVM/XLA. Pun handoff:
`~/prakse/UIUC/notes/HANDOFF_GPU_HYBRID_2026-10-28.md`; stalni kontekst: `~/prakse/UIUC/CLAUDE.md`.
