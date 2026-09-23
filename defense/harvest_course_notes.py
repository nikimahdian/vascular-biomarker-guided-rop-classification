"""Harvest v2 - adds the 30 challenging questions (file 15) and the
presentation-structure file (16) to the long-prose harvest."""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_defense\course\ROP_Defense_Complete_Course")
JS = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_defense\_harvest.json")
IDX = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_defense\_harvest_index.txt")


def clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"^[-*•]\s*", "— ", s)
    s = s.replace("`", "")
    return s.strip()


def blocks_of(text: str):
    out, cur = [], []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if cur:
                out.append("\n".join(cur)); cur = []
            continue
        cur.append(clean(line))
    if cur:
        out.append("\n".join(cur))
    merged = []
    for b in out:
        b = b.strip()
        if not b:
            continue
        if merged and len(merged[-1]) < 120 and not merged[-1].startswith("—"):
            merged[-1] = merged[-1] + " " + b
        else:
            merged.append(b)
    return merged


def sections(text):
    parts = re.split(r"^(#{2,3}) (.+)$", text, flags=re.M)
    i = 1
    while i < len(parts) - 2:
        yield parts[i], parts[i + 1].strip(), parts[i + 2]
        i += 3


data = {}

DASH = re.compile(r"^-{3,}$")


def flat_paragraphs(path, key):
    text = path.read_text(encoding="utf-8")
    out = []
    for blk in blocks_of(text):
        b = blk.strip()
        if not b or DASH.match(b) or b.startswith("#"):
            continue
        out.append(b)
    for i, b in enumerate(out):
        data[f"{key}:P{i}"] = {"file": path.name, "title": f"بند {i}", "blocks": [b]}


for f in sorted(SRC.glob("*.md")):
    n = f.name
    if n.startswith(("00_", "06_", "17_", "18_")):
        continue
    key = n.split("_")[0]
    text = f.read_text(encoding="utf-8")
    qn = 0
    for hashes, title, body in sections(text):
        take = False
        if n.startswith("15_"):
            take = hashes == "##" and re.match(r"^[۰-۹\d]+\.", title)
            tag = f"15-Q{title.split('.')[0]}"
        elif n.startswith("16_"):
            take = hashes in ("##", "###")
            tag = "16-" + re.sub(r"[^0-9A-Za-z]+", "", title)[:8]
            if title.startswith("اسلاید"):
                tag = "16-" + title.split(":")[0].replace("اسلاید", "S").strip()
            elif "متن آماده" in title:
                tag = "16-L"
            elif "روایت" in title:
                tag = "16-N"
        else:
            is_long = "متن بلند" in title
            is_qa = hashes == "###"
            take = is_long or is_qa
            if is_long:
                tag = f"{key}-L"
            elif is_qa:
                qn += 1
                tag = f"{key}-Q{qn}"
            else:
                tag = None
        if not take:
            continue
        bs = blocks_of(body)
        if not bs:
            continue
        if tag in data:
            data[tag]["blocks"].extend(bs)
        else:
            data[tag] = {"file": n, "title": title, "blocks": bs}

JS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

for _f, _k in [("18_متن_پیوسته_کامل_آماده_دفاع.md", "18"),
               ("06_متن_پیوسته_مبانی_جلسات_۱_تا_۵.md", "06")]:
    flat_paragraphs(SRC / _f, _k)

JS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

idx = []
for tag, d in data.items():
    idx.append(f"== {tag} | {d['title']} | blocks={len(d['blocks'])} | chars={sum(len(b) for b in d['blocks'])}")
    for j, b in enumerate(d["blocks"]):
        t = b.replace("\n", " ")
        idx.append(f"   [{tag}:{j}] ({len(b)}) {t[:100]}")
IDX.write_text("\n".join(idx), encoding="utf-8")
print("sections:", len(data), "blocks:", sum(len(d["blocks"]) for d in data.values()),
      "chars:", sum(len(b) for d in data.values() for b in d["blocks"]))
