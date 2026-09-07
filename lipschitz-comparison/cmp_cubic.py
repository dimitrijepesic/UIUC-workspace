base = [3490.751463457233, 1967.1691505810531, 1288.4117498404194, 847.3514427551503, 535.9627927255128, 309.2134253452785, 194.8153958745007, 150.27700399916642, 124.28751414160955, 112.3854722517994, 107.68368561339366, 105.53800367388723, 104.40769497374805, 103.78249733017932, 103.43855225565406, 103.2494093749031]
import re, ast, sys
txt = open(r"C:\Users\dimit\AppData\Local\Temp\claude\C--everything-prakse-UIUC\d2abac21-fea3-4ed8-b65d-67360d0e8dd8\tasks\b5su9vg2f.output", encoding="utf-8", errors="replace").read()
new = ast.literal_eval(re.search(r"lc_precise (\[.*?\])", txt, re.S).group(1))
tm  = ast.literal_eval(re.search(r"time_precise (\[.*?\])", txt, re.S).group(1))
eps = [10 ** (-k / 4) * 2 for k in range(2, 18)]
print(f"{'eps':>12s} {'complex64':>14s} {'real cubic':>14s} {'rel diff':>11s} {'tighter?':>9s}")
worst = 0.0
for e, b, n in zip(eps, base, new):
    rel = (n - b) / b
    worst = max(worst, abs(rel))
    print(f"{e:12.6g} {b:14.6f} {n:14.6f} {rel:+11.2e} {'yes' if n < b else ('same' if n == b else 'no'):>9s}")
print(f"\nlargest |relative difference| in the reported bound: {worst:.3e}")
print(f"mean time_precise with real cubic: {sum(tm)/len(tm):.5f}s  (baseline 0.01480s)")
