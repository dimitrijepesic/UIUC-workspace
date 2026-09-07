import re
full = open("pasado_paper.txt", encoding="utf-8").read()
for m in re.finditer(r"Shi et al", full):
    s=max(0,m.start()-900); e=min(len(full), m.end()+900)
    print(f"----- @{m.start()} -----")
    print(full[s:e].replace("\n"," "))
    print()
