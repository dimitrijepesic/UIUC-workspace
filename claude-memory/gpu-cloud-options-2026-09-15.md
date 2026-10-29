---
name: gpu-cloud-options-2026-09-15
description: Procena 15.9.2026 kako raditi GPU deo hibrida sa MacBook-a (MPS nema fp64) — Modal besplatni krediti kao primarni izbor, RunPod/Kaggle/Colab kao alternative, cene i budžet
metadata:
  type: project
---

Korisnik 15.9.2026 pitao kako da radi hibridni port sa MacBook-a i rekao da je
spreman da plati "minimalno". Nalaz: sve osim koraka 4–5 iz
[[gpu-hybrid-port-plan]] i A100/T4 merenja Faze 1 radi lokalno; GPU deo ide u
cloud jer MPS nema float64 (videti [[mac-setup-2026-09-07]]).

Opcije i cene na 15.9.2026 (web pretraga):
- **Modal**: 30 $/mesec besplatnih kredita; A100 80GB 3,72 $/h, A100 40GB
  2,78 $/h, H100 3,95 $/h → ~8–11 h A100 mesečno besplatno; `modal run` sa
  Mac-a šalje lokalni folder (ne treba git clone), bez notebook-a, bez padanja
  sesija, naplata po sekundi. **Preporučeno primarno.**
- **RunPod**: A100 PCIe 1,19–1,39 $/h, SXM 1,39–1,59 $/h; SSH pod, mora da se
  gasi. Za finalne sweep-ove kad Modal krediti presuše.
- **Kaggle**: 0 $, 30 h/nedelja P100 (fp64 ~4,7 TFLOPS, ~11× Mac) ili T4×2
  (za T4 tačku iz Faze 1). Sesija do 12 h.
- **Colab Pro**: 9,99 $/mes = 100 CU, A100 13–15 CU/h → ~7 h; A100 nije
  garantovan, sesije pucaju (korisnik već iskusio). Ne preporučeno.
- **UIUC grupa / NCSA Delta**: jedina besplatna i stabilna opcija; pitanje o
  GPU-u grupe bilo je samo u dugom nacrtu poruke Saši 12.9, **nije poslato**
  u kratkoj verziji → ubaciti u sledeći update.

Procena potrošnje do 30.10: 10–15 A100-sati (razvoj+korektnost 3–4 h, Faza 1
mikrobenčmark 1–2 h, finalna merenja 3–5 h) → Modal krediti pokrivaju većinu,
ostatak ≤10–20 $.

**Why:** odluka gde se vrti GPU deo određuje tok rada (mount vs git clone) i
metodologiju: CPU-only vs hibrid mora na istoj mašini (cloud box ima slab
CPU), a Mac CPU-only se prijavljuje zasebno.
**How to apply:** predložen tok = korak 0 commit → `pip install modal` →
`experiments/modal_run.py` (image torch+sklearn+tqdm, add_local_dir bez .venv,
gpu A100, cpu 8) → prvi test `colab_gpu_profile.py` kroz Modal (poređenje sa
Fazom 13 u `GPU_READINESS.md`). Videti [[phd-goal-2026-09-12]].
