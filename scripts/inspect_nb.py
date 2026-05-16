import json
with open("src/notebooks/notebook.ipynb", encoding="utf-8") as f:
    nb = json.load(f)
print("total cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))[:80].replace("\n", " ")
    cid = c.get("id", "?")
    print(f"[{i:2d}] {c['cell_type'][:4]} id={cid[:10]:10} | {src}")
