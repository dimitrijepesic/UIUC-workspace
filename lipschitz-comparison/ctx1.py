import re
full = open("pasado_paper.txt", encoding="utf-8").read()
def ctx(kw, w=700):
    for m in re.finditer(kw, full, re.I):
        s = max(0, m.start()-w//2); e = min(len(full), m.end()+w//2)
        print(f"----- {kw} @ {m.start()} -----")
        print(full[s:e].replace("\n"," "))
        print()
ctx("GPU")
ctx("parallel", 500)
