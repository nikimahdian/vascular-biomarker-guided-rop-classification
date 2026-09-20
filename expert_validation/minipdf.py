"""Minimal Markdown -> PDF renderer built on matplotlib. No external PDF dependency.

Written because this machine has neither LaTeX, pandoc, wkhtmltopdf, reportlab nor fpdf, and
CUPS on macOS only ships a text/plain -> PDF filter (there is no HTML filter chain).

Supports the subset used by the clinician documents: h1-h3, paragraphs, bullets, numbered items,
fenced code, horizontal rules and pipe tables.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402

TTP = TextToPath()
FAMILY = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

PAGE_W, PAGE_H = 8.27, 11.69          # A4 portrait, inches
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 0.85, 0.75, 0.80
FS = 9.6
LEAD = 1.42


def _width_pt(s: str, size: float, bold: bool = False, mono: bool = False) -> float:
    if not s:
        return 0.0
    fp = FontProperties(family=MONO if mono else FAMILY, size=size,
                        weight="bold" if bold else "normal")
    w, _, _ = TTP.get_text_width_height_descent(s, fp, ismath=False)
    return float(w)


def _clean(s: str) -> str:
    """Strip inline Markdown markers. Base-14 fonts cannot mix weights inside a string, so the
    emphasis markers are removed and the words kept, rather than printed literally."""
    s = s.replace("**", "").replace("`", "")
    out, in_em = [], False
    for ch in s:
        if ch == "*":
            in_em = not in_em
            continue
        out.append(ch)
    return "".join(out)


def _wrap(text: str, max_pt: float, size: float, bold: bool = False, mono: bool = False) -> list[str]:
    words, lines, cur = _clean(text).split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if _width_pt(trial, size, bold, mono) <= max_pt or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _blocks(md: str):
    out, table, code_buf, in_code = [], [], [], False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                out.append(("code", code_buf))
                code_buf, in_code = [], False
            else:
                if table:
                    out.append(("table", table))
                    table = []
                in_code = True
            continue
        if in_code:
            code_buf.append(raw)
            continue
        if line.startswith("|") and line.endswith("|"):
            table.append([c.strip() for c in line.strip("|").split("|")])
            continue
        if table:
            out.append(("table", table))
            table = []
        if not line:
            out.append(("gap", ""))
            continue
        if set(line) <= {"-"} and len(line) >= 3:
            out.append(("rule", ""))
            continue
        matched = False
        for n, key in ((3, "h3"), (2, "h2"), (1, "h1")):
            if line.startswith("#" * n + " "):
                out.append((key, line[n + 1:].strip()))
                matched = True
                break
        if matched:
            continue
        stripped = line.lstrip()
        if stripped.startswith(("- ", "* ")):
            out.append(("bullet", stripped[2:].strip()))
        elif stripped[:4].rstrip(".").isdigit() and ". " in stripped[:5]:
            num, _, rest = stripped.partition(". ")
            out.append(("num", f"{num}. {rest.strip()}"))
        elif stripped.startswith(">"):
            out.append(("quote", stripped.lstrip("> ").strip()))
        else:
            out.append(("p", stripped))
    if table:
        out.append(("table", table))
    if code_buf:
        out.append(("code", code_buf))
    return out


def render_pdf(md_text: str, pdf_path: Path, title: str) -> bool:
    body = md_text if md_text.lstrip().startswith("# ") else f"# {title}\n\n{md_text}"
    blocks = _blocks(body)
    left = MARGIN_X / PAGE_W
    max_pt = (PAGE_W - 2 * MARGIN_X) * 72.0

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        y = PAGE_H - MARGIN_TOP
        page_no = 1

        def newpage():
            nonlocal fig, y, page_no
            _footer(fig, page_no, title)
            pdf.savefig(fig)
            plt.close(fig)
            page_no += 1
            fig = plt.figure(figsize=(PAGE_W, PAGE_H))
            y = PAGE_H - MARGIN_TOP

        def put(text, size=FS, bold=False, mono=False, indent=0.0, dy=None):
            nonlocal y
            step = (dy if dy is not None else size * LEAD) / 72.0
            if y - step < MARGIN_BOT:
                newpage()
            # y is in inches and indent is in points; matplotlib wants figure fractions
            fig.text(left + (indent / 72.0) / PAGE_W, (y - size / 72.0 * 0.85) / PAGE_H, text,
                     fontsize=size, family=MONO if mono else FAMILY,
                     weight="bold" if bold else "normal", va="baseline", ha="left")
            y -= step

        for kind, payload in blocks:
            if kind == "gap":
                y -= FS * 0.45 / 72.0
                continue
            if kind == "rule":
                if y - 0.16 < MARGIN_BOT:
                    newpage()
                fig.add_artist(plt.Line2D([left, 1 - MARGIN_X / PAGE_W],
                                          [(y - 0.05) / PAGE_H] * 2,
                                          color="0.7", linewidth=0.7,
                                          transform=fig.transFigure))
                y -= 0.16
                continue
            if kind in ("h1", "h2", "h3"):
                size, bold, pre, post = {"h1": (16, True, 0.10, 0.12),
                                         "h2": (12.5, True, 0.14, 0.07),
                                         "h3": (10.8, True, 0.10, 0.05)}[kind]
                y -= pre
                for ln in _wrap(payload, max_pt, size, bold=True):
                    put(ln, size=size, bold=True)
                y -= post
                continue
            if kind == "code":
                for ln in payload:
                    put(ln, size=8.2, mono=True, dy=8.2 * 1.25)
                y -= 0.06
                continue
            if kind == "table":
                head, body = payload[0], payload[2:]
                ncol = max(len(r) for r in payload)
                colw = max_pt / ncol

                def row(cells, bold):
                    nonlocal y
                    wrapped = [_wrap(str(c), colw - 6, 8.4, bold=bold) for c in cells]
                    for i in range(max(len(w) for w in wrapped)):
                        if y - 8.4 * 1.3 / 72.0 < MARGIN_BOT:
                            newpage()
                        for j, wl in enumerate(wrapped):
                            if i < len(wl):
                                fig.text(left + (j * colw / 72.0) / PAGE_W,
                                         (y - 8.4 / 72.0 * 0.85) / PAGE_H, wl[i],
                                         fontsize=8.4, family=FAMILY,
                                         weight="bold" if bold else "normal",
                                         va="baseline", ha="left")
                        y -= 8.4 * 1.3 / 72.0
                    y -= 0.02

                row(head, True)
                fig.add_artist(plt.Line2D([left, 1 - MARGIN_X / PAGE_W],
                                          [(y + 0.02) / PAGE_H] * 2, color="0.6",
                                          linewidth=0.6, transform=fig.transFigure))
                for r in body:
                    row((r + [""] * ncol)[:ncol], False)
                y -= 0.06
                continue

            size = FS
            indent = 14 if kind in ("bullet", "num", "quote") else 0
            prefix = {"bullet": "\u2022  ", "num": "", "quote": "|  "}.get(kind, "")
            text = prefix + payload
            for ln in _wrap(text, max_pt - indent, size):
                put(ln, size=size, indent=indent)

        _footer(fig, page_no, title)
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path.exists() and pdf_path.stat().st_size > 1000


def _footer(fig, page_no: int, title: str) -> None:
    fig.text(0.5, 0.42 / PAGE_H, f"{title}  |  page {page_no}", fontsize=7.2,
             family=FAMILY, color="0.45", ha="center", va="baseline")
