---
name: gpu-hybrid-port-plan
description: Plan za hibrid CPU(gelsd)/GPU port Pasado precise puta za big mrežu (dogovoren 23.8.2026), 5 koraka, 3-4 dana; kreće od vec_batched_grid zbog .item() sync-ova
metadata:
  type: project
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
0. Commit sprint rada → grana `sprint-optimizations`; port na grani `gpu-hybrid`.
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

Procena: 3–4 radna dana. Male mreže/get_linspace za CPU: deprioritizovano.
Imati u vidu: Saša želi mreže veće od big → ne hardkodovati dimenzije.
Videti [[gelsd-profiling-result-2026-08-22]], [[pasado-code-review-plan]].
