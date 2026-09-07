import re
full = open("pasado_paper.txt", encoding="utf-8").read()
for kw in [r"random", r"sub-?sampl", r"sound", r"noise symbol"]:
    print("="*70); print("KW:", kw); print("="*70)
    ms = list(re.finditer(kw, full, re.I))
    print("count:", len(ms))
    for m in ms[:6]:
        s=max(0,m.start()-400); e=min(len(full), m.end()+400)
        print(f"--- @{m.start()} ---")
        print(full[s:e].replace("\n"," "))
        print()
