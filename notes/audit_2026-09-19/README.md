# Logovi pregleda 19.9.2026.

Izvorni Pasado HEAD: `515c864`, radna grana `sprint-optimizations`.
Python 3.13.15, torch 2.12.1, NumPy 2.5.1, 5 torch threadova, CUDA=False.
Skripte su pokrenute postojećim `Pasado/.venv/bin/python`, pojedinačno,
bez nasleđenih `PASADO_*` flagova za jedinične testove.

| Log | Izvorna skripta | Rezultat |
|---|---|---|
| corners.log | forward_mode_tensorized_src/test_check_corners_tensor.py | PASS |
| boundary.log | forward_mode_tensorized_src/test_check_nonlinear_boundary_tensor.py | PASS |
| max_error.log | forward_mode_tensorized_src/test_compute_max_error_tensor.py | PASS |
| selectors.log | forward_mode_tensorized_src/test_selector_paths.py | PASS |
| batched_grid.log | forward_mode_tensorized_src/test_batched_grid.py | FAIL na STEPS=7/9, exit 0; STEPS=5 prolazi |
| batched_lstsq.log | experiments/test_batched_lin_reg.py | PASS |
| csv_consistency.log | lipschitz-comparison/check_consistency.py | PASS |

`correctness_vectorized.csv` i `correctness_batched.csv` dobijeni su pozivom
postojećeg modula `experiments/correctness_e2e.py`, promenom njegovog
`CSV_PATH` na privremeni izlaz i `BATCHED=False/True`, uz
`USE_BATCHED_GRID=False`, `USE_REAL_CUBIC=False`. Obe provere su PASS.
Izvorni referentni CSV-i nisu prepisani.

Za proveru BaB broja poziva izdvojena je neizmenjena funkcija `bab_curve`
iz AST-a `experiments/colab_bab_sweep.py`. `solve_interval` je zamenjen
stubom koji upisuje `(lo, hi)` i vraća širinu intervala. Za budget=9:
9 poziva, checkpoint-i `[1, 3, 5, 7, 9]`. Ovo proverava kontrolni tok,
ne GPU vreme niti numeriku auto_LiRPA.

Zaključci i ograničenja: [audit](../PROJECT_AUDIT_2026-09-19.md).
