import pypdf, re, sys
r = pypdf.PdfReader(r"C:\everything\prakse\UIUC\OOPSLA2023Final.pdf")
print("PAGES:", len(r.pages))
txt = []
for i, p in enumerate(r.pages):
    t = p.extract_text() or ""
    txt.append(t)
full = "\n".join(txt)
open(r"C:\everything\prakse\UIUC\lipschitz-comparison\pasado_paper.txt","w",encoding="utf-8").write(full)
print("CHARS:", len(full))
for kw in ["GPU","CUDA","parallel","paralleliz","hardware","accelerat","limitation","future work","threat","scalab"]:
    hits = [m.start() for m in re.finditer(kw, full, re.I)]
    print(f"{kw!r}: {len(hits)}")
