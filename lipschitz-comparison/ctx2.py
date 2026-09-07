import re
full = open("pasado_paper.txt", encoding="utf-8").read()
for m in re.finditer("scalab", full, re.I):
    s=max(0,m.start()-450); e=min(len(full), m.end()+450)
    print(f"----- @{m.start()} -----")
    print(full[s:e].replace("\n"," "))
    print()
