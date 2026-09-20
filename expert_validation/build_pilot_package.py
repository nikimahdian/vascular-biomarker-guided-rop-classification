#!/usr/bin/env python
"""Section 5-14: build the blinded, clinician-ready pilot package.

Frozen inputs, never modified:
  expert_validation/blinding_key.csv     (authoritative; filtered to cohort == 'pilot')
  expert_validation/ANNOTATION_MANUAL.md (copied verbatim)
  expert_validation/PROTOCOL.md          (internal; NOT shipped)

Privacy rules enforced here:
  * every exported image is rebuilt from the decoded pixel array, so no EXIF/IPTC/XMP/PNG-text
    chunk from the source can survive
  * the output filename is the frozen study_id and nothing else
  * no source, label, split, group, geometry, prediction, automatic mask or key file is written
    anywhere inside the package
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minipdf import render_pdf  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path("/Users/moniaz/niki")
EV = REPO / "expert_validation"
OUT = EV / "package_build"
PKG = OUT / "expert_pilot30_v1.0"
ZIP_PATH = OUT / "expert_pilot30_v1.0.zip"
VERSION = "expert_pilot30_v1.0"
PROTOCOL_VERSION = "1.0-pilot"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- helpers

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md_to_html(md: str, title: str) -> str:
    """Small Markdown subset -> HTML for cupsfilter. Headings, bold, code, lists, tables, rules."""
    out: list[str] = []
    in_code = in_table = False
    rows: list[list[str]] = []

    def flush_table():
        nonlocal rows
        if not rows:
            return
        head, body = rows[0], rows[2:]
        out.append("<table border='1' cellspacing='0' cellpadding='4'>")
        out.append("<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head) + "</tr>")
        for r in body:
            out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>")
        out.append("</table>")
        rows = []

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                out.append("</pre>")
                in_code = False
            else:
                flush_table()
                out.append("<pre>")
                in_code = True
            continue
        if in_code:
            out.append(html.escape(raw))
            continue
        if line.startswith("|") and line.endswith("|"):
            rows.append([c.strip() for c in line.strip("|").split("|")])
            in_table = True
            continue
        if in_table and not line.startswith("|"):
            flush_table()
            in_table = False
        if not line:
            out.append("<p></p>")
            continue
        if set(line) <= {"-"} and len(line) >= 3:
            out.append("<hr/>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{html.escape(m.group(2))}</h{lvl}>")
            continue
        m = re.match(r"^[-*]\s+(.*)$", line)
        if m:
            out.append(f"<p>&#8226; {html.escape(m.group(1))}</p>")
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if m:
            out.append(f"<p>{m.group(1)}. {html.escape(m.group(2))}</p>")
            continue
        txt = html.escape(line)
        txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
        txt = re.sub(r"`(.+?)`", r"<tt>\1</tt>", txt)
        txt = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", txt)
        out.append(f"<p>{txt}</p>")
    flush_table()
    if in_code:
        out.append("</pre>")
    return ("<html><head><meta charset='utf-8'/></head><body>"
            f"<h1>{html.escape(title)}</h1>" + "\n".join(out) + "</body></html>")


def html_to_pdf(html_text: str, pdf_path: Path) -> bool:
    tmp = pdf_path.with_suffix(".html")
    tmp.write_text(html_text, encoding="utf-8")
    try:
        with pdf_path.open("wb") as fh:
            r = subprocess.run(["/usr/sbin/cupsfilter", "-i", "text/html", "-m", "application/pdf",
                                str(tmp)], stdout=fh, stderr=subprocess.DEVNULL, timeout=120)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] cupsfilter failed: {e}")
        return False
    ok = r.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 1000
    if not ok:
        print(f"  [warn] cupsfilter returncode={r.returncode} size="
              f"{pdf_path.stat().st_size if pdf_path.exists() else 0}")
    return ok


def load_font(size: int):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------- schematic examples

def make_examples(dest: Path) -> None:
    """Schematic illustrations, drawn from scratch. No study image and no external dataset is
    used, so there is nothing to license and nothing that could leak."""
    dest.mkdir(parents=True, exist_ok=True)
    W, H = 900, 600
    f_t = load_font(22)
    f_s = load_font(17)

    def base(bg=(28, 22, 20)):
        im = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(im)
        d.ellipse([40, 20, W - 40, H - 20], fill=(150, 92, 72))       # schematic fundus field
        return im, d

    def arcade(d, cx, cy, colour, width, bend, n=6, length=250):
        for k in range(n):
            ang = np.pi * (-0.85 + 0.34 * k)
            pts = []
            for t in np.linspace(0, 1, 60):
                r = length * t
                th = ang + bend * t
                pts.append((cx + r * np.cos(th), cy + r * np.sin(th)))
            d.line(pts, fill=colour, width=width, joint="curve")

    def banner(im, text, colour=(0, 0, 0)):
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 34], fill=colour)
        d.text((8, 6), text, fill=(255, 255, 255), font=f_t)

    # 1 correct vessel annotation
    im, d = base()
    arcade(d, 300, 300, (250, 240, 235), 7, 0.5)
    arcade(d, 300, 300, (250, 240, 235), 4, -0.5)
    arcade(d, 300, 300, (250, 240, 235), 3, 0.9, n=5, length=200)
    d.ellipse([268, 268, 332, 332], outline=(255, 220, 60), width=3)
    banner(im, "SCHEMATIC - correct: main arcades plus traceable second-order branches")
    im.save(dest / "example_vessel_correct.png")

    # 2 common errors
    im, d = base()
    arcade(d, 300, 300, (250, 240, 235), 7, 0.5)
    arcade(d, 300, 300, (250, 240, 235), 4, -0.5)
    d.ellipse([268, 268, 332, 332], fill=(240, 230, 225))              # WRONG: disc filled in
    d.line([(520, 200), (640, 260), (760, 430)], fill=(255, 255, 255), width=5)   # WRONG: reflection
    d.ellipse([430, 400, 520, 470], fill=(120, 40, 40))                # haemorrhage
    d.line([(470, 435), (700, 500)], fill=(255, 255, 255), width=4)    # WRONG: traced through it
    banner(im, "SCHEMATIC - three errors: disc filled, reflection traced, haemorrhage crossed")
    im.save(dest / "example_vessel_common_error.png")

    # 3 disc correct
    im, d = base()
    arcade(d, 300, 300, (250, 240, 235), 6, 0.5)
    d.ellipse([268, 268, 332, 332], fill=(235, 226, 205), outline=(0, 220, 0), width=4)
    d.line([296, 300, 304, 300], fill=(255, 0, 0), width=2)
    d.line([300, 296, 300, 304], fill=(255, 0, 0), width=2)
    banner(im, "SCHEMATIC - complete disc boundary with centre marked")
    im.save(dest / "example_disc_correct.png")

    # 4 disc not gradable
    im, d = base()
    d.ellipse([-40, 250, 60, 350], fill=(235, 226, 205))               # disc cut off by the edge
    banner(im, "SCHEMATIC - disc cut off by the field edge: gradable_disc = FALSE")
    im.save(dest / "example_disc_not_gradable.png")

    # 5 uncertain
    im, d = base()
    arcade(d, 300, 300, (250, 240, 235), 6, 0.5)
    pts = [(x, 470 + 40 * np.sin(x / 90.0)) for x in range(500, 700)]
    d.line(pts, fill=(250, 240, 235), width=2)
    d.rectangle([498, 430, 706, 512], outline=(255, 140, 0), width=3)
    d.text((500, 516), "fades below visibility - stop the trace and note it",
           fill=(255, 255, 255), font=f_s)
    banner(im, "SCHEMATIC - uncertain peripheral vessel: do not guess")
    im.save(dest / "example_uncertain.png")


# --------------------------------------------------------------------------- package

def build() -> int:
    key = pd.read_csv(EV / "blinding_key.csv")
    pilot = key[key.cohort == "pilot"].copy().sort_values("study_id")
    assert len(pilot) == 30, f"pilot must be 30, got {len(pilot)}"
    assert pilot.study_id.is_unique and pilot.image_path.is_unique and pilot.group_id.is_unique

    if PKG.exists():
        shutil.rmtree(PKG)
    for sub in ("images", "vessel_masks", "disc_masks", "av_labels"):
        (PKG / sub).mkdir(parents=True, exist_ok=True)
    print(f"staging: {PKG}")

    # ---- 5. blinded image export, metadata-free --------------------------------
    audit_rows = []
    for _, r in pilot.iterrows():
        src = Path(r.image_path)
        sid = r.study_id
        if not src.exists():
            print(f"  [ERROR] source image missing: {src}")
            return 1
        with Image.open(src) as im:
            im.load()
            orig_size = im.size
            orig_mode = im.mode
            arr = np.array(im.convert("RGB"))
            src_info = sorted(im.info.keys())
            src_exif = {}
            try:
                e = im.getexif()
                src_exif = {str(k): str(v)[:80] for k, v in e.items()} if e else {}
            except Exception:  # noqa: BLE001
                pass
        dst = PKG / "images" / f"{sid}.png"
        # rebuild from the array so no source metadata object can carry over
        Image.fromarray(arr, mode="RGB").save(dst, format="PNG", optimize=False)

        with Image.open(dst) as chk:
            chk.load()
            out_size = chk.size
            out_info = sorted(chk.info.keys())
            out_exif = dict(chk.getexif()) if chk.getexif() else {}
            same_pixels = bool((np.array(chk.convert("RGB")) == arr).all())

        sensitive = [k for k in src_info
                     if any(s in k.lower() for s in ("exif", "icc", "comment", "software", "xmp",
                                                     "iptc", "xml", "profile", "dpi", "jfif"))]
        audit_rows.append({
            "study_id": sid,
            "output_file": f"images/{sid}.png",
            "source_size": f"{orig_size[0]}x{orig_size[1]}",
            "output_size": f"{out_size[0]}x{out_size[1]}",
            "original_metadata_present": "|".join(src_info) or "none",
            "source_exif_tags": len(src_exif),
            "sensitive_metadata_found": "|".join(sensitive) or "none",
            "metadata_removed": "yes" if (src_info and not out_info) or sensitive else
                                ("yes" if src_info != out_info else "already-clean"),
            "output_metadata_present": "|".join(out_info) or "none",
            "output_exif_tags": len(out_exif),
            "pixel_dimensions_preserved": "yes" if orig_size == out_size else "NO",
            "pixels_identical_to_source": "yes" if same_pixels else "NO",
            "notes": "",
        })
    aud = pd.DataFrame(audit_rows)
    aud.to_csv(PKG / "metadata_audit.csv", index=False)
    bad_dim = aud[aud.pixel_dimensions_preserved != "yes"]
    bad_px = aud[aud.pixels_identical_to_source != "yes"]
    print(f"  exported {len(aud)} images; dimension mismatches={len(bad_dim)}; "
          f"pixel mismatches={len(bad_px)}")
    print(f"  source images carrying flagged metadata: "
          f"{int((aud.sensitive_metadata_found != 'none').sum())}")
    print(f"  exported images carrying any metadata: "
          f"{int((aud.output_metadata_present != 'none').sum())}")

    # ---- schematic examples the manual refers to -------------------------------
    make_examples(PKG / "examples")
    print(f"  examples: {len(list((PKG / 'examples').glob('*.png')))} schematic images")

    # ---- 9. README_FIRST -------------------------------------------------------
    readme = f"""# Blinded pilot annotation study — please read first

Thank you for helping with this study. It has **30 fundus photographs**. Your task is to
annotate retinal vessels and the optic disc, and to record your clinical impression, using the
images alone.

## The most important points

1. **This is a blinded study.** The images come from more than one clinical source, but that
   information is deliberately withheld, and so are the original diagnoses and all computer
   output. Please do not try to work out which images come from where, or what the "right"
   answer is meant to be. Your independent judgement is the entire point; guessing the expected
   answer would destroy the value of your work.

2. **Work from the photograph only on the first pass.** Do not ask for, and do not open, any
   automatic mask or computer output before you have saved your own annotation for that image.
   If such a file is ever offered to you, please decline and tell the study contact.

3. **You are not being tested.** Where you are unsure, recording the uncertainty is more useful
   than guessing. A vessel you cannot follow should be left out; a disc you cannot define should
   be marked not gradable.

## What to do, per image

1. Open `images/<study_id>.png`. Zoom and pan freely. You may change brightness and contrast to
   see better, but do not crop, rotate or resize.
2. Record `image_quality`: `gradable`, `borderline` or `ungradable`. If it is not `gradable`, tick
   the matching reason columns.
3. Draw the **vessel mask** and save it as `vessel_masks/<study_id>.png`.
   PNG, same width and height as the photograph, `0` = background, `255` = retinal vessel.
4. Draw the **optic-disc mask** and save it as `disc_masks/<study_id>.png`, same format.
   Also record the disc centre and radius in the sheet. If the disc cannot be defined, set
   `disc_gradable = FALSE` and do not draw one.
5. Record `rop_vascular_grade` and `grade_confidence`.
6. Record `arterial_tortuosity_score` and `venous_dilation_score`, each 0 to 4.
7. For the images listed for artery/vein labelling, label each **major vessel branch** leaving
   the disc as `A` (artery), `V` (vein) or `U` (uncertain) and save under `av_labels/`.

## An image you cannot grade

Set `image_quality = ungradable`, tick a reason, and set `vessels_gradable = FALSE` and
`disc_gradable = FALSE`. **Do not draw a mask for that image.** A missing mask is a valid and
expected answer; an invented one is not.

## Uncertainty

- A thin vessel you cannot follow: leave it out and note it.
- A disc cut off by the edge of the field, or whose boundary you cannot see: `disc_gradable = FALSE`.
- A branch you cannot confidently call: mark it `U`, not a guess.

## The grading sheet

`grading_sheet.csv` has one row per image. Column meanings and the allowed values are listed in
`ANNOTATION_MANUAL.pdf`. Please fill in the columns and leave `notes` for anything you wanted to
say but had no column for.

## Files in this package

```
README_FIRST.pdf          this document
ANNOTATION_MANUAL.pdf     the full instructions, with worked schematic examples
grading_sheet.csv         one row per image, to be filled in
images/                   the 30 photographs, named by study ID
vessel_masks/             save your vessel masks here
disc_masks/               save your optic-disc masks here
av_labels/                save your artery/vein branch labels here
examples/                 schematic illustrations (not study images)
PACKAGE_MANIFEST_SHA256.txt   checksums, so you can confirm nothing was corrupted in transit
```

Total time is roughly **45 to 60 minutes** for 30 images. Please work in sessions of no more than
about 90 minutes, with a break.

## Contact

`[STUDY CONTACT PLACEHOLDER — name, role, email, and the preferred channel for questions]`

Please send questions rather than deciding an ambiguous case alone. Finding those ambiguities is
one of the purposes of this pilot.
"""
    (PKG / "README_FIRST.md").write_text(readme, encoding="utf-8")

    # ---- 10. annotation manual, verbatim + examples ----------------------------
    # The frozen manual was written for the full 120-image study. The body is shipped unedited;
    # a pilot header states the one factual difference (image count) so nothing in the frozen
    # procedure is rewritten.
    manual_md = (EV / "ANNOTATION_MANUAL.md").read_text(encoding="utf-8")
    pilot_header = (
        "> **PILOT PACKAGE — 30 images.** This package contains **30** fundus photographs, the\n"
        "> first cohort of the study. The procedure below is the frozen annotation procedure and\n"
        "> applies unchanged. Where the text refers to 120 images or to a final validation cohort,\n"
        "> read **30 images** for this pilot. Everything else — the seven steps per image, the\n"
        "> vessel and disc rules, the uncertainty rules, the grading sheet — is exactly as written.\n\n"
        "---\n\n"
    )
    (PKG / "ANNOTATION_MANUAL.md").write_text(pilot_header + manual_md, encoding="utf-8")

    ok_r = render_pdf(readme, PKG / "README_FIRST.pdf", "Blinded pilot annotation study")
    ok_m = render_pdf(pilot_header + manual_md, PKG / "ANNOTATION_MANUAL.pdf",
                      "Annotation Manual - version 1.0 (pilot)")
    print(f"  PDFs: README_FIRST.pdf={'ok' if ok_r else 'FAILED'} "
          f"({(PKG / 'README_FIRST.pdf').stat().st_size if ok_r else 0} bytes), "
          f"ANNOTATION_MANUAL.pdf={'ok' if ok_m else 'FAILED'} "
          f"({(PKG / 'ANNOTATION_MANUAL.pdf').stat().st_size if ok_m else 0} bytes)")
    if not (ok_r and ok_m):
        return 1
    for f in ("README_FIRST.md", "ANNOTATION_MANUAL.md"):
        if (PKG / f).exists():
            (PKG / f).unlink()

    # ---- 11. grading sheet -----------------------------------------------------
    cols = ["study_id", "grader_id", "annotation_date", "image_quality",
            "quality_reason_blur", "quality_reason_low_contrast",
            "quality_reason_poor_illumination", "quality_reason_artifact",
            "quality_reason_incomplete_field", "quality_reason_other",
            "vessels_gradable", "disc_gradable", "disc_x", "disc_y", "disc_radius",
            "rop_vascular_grade", "grade_confidence",
            "arterial_tortuosity_score", "venous_dilation_score",
            "av_annotation_completed", "notes"]
    gs = pd.DataFrame({c: [""] * 30 for c in cols})
    gs["study_id"] = pilot.study_id.tolist()
    gs.to_csv(PKG / "grading_sheet.csv", index=False)

    (PKG / "vessel_masks" / "README.txt").write_text(
        "Save one vessel mask per image here, named exactly after the study ID.\n"
        "Example: images/ROP_0007.png  ->  vessel_masks/ROP_0007.png\n"
        "Format: PNG, single channel, same width and height as the photograph, "
        "0 = background, 255 = retinal vessel.\n"
        "No mask is required for an image you marked ungradable.\n", encoding="utf-8")
    (PKG / "disc_masks" / "README.txt").write_text(
        "Save one optic-disc mask per image here, named exactly after the study ID.\n"
        "Same geometry and format rules as the vessel masks.\n"
        "If disc_gradable = FALSE, do not create a file.\n", encoding="utf-8")
    (PKG / "av_labels" / "README.txt").write_text(
        "For the images listed for artery/vein labelling, save the major-vessel branch labels\n"
        "here, labelled A (artery), V (vein) or U (uncertain).\n"
        "Label whole branches, not individual pixels.\n", encoding="utf-8")

    # ---- 14. per-file manifest -------------------------------------------------
    files = sorted(p for p in PKG.rglob("*") if p.is_file())
    lines = [f"package           : {VERSION}",
             f"protocol_version  : {PROTOCOL_VERSION}",
             f"creation_commit   : {git_head()}",
             f"locked_manifest   : {locked_manifest_hash()}",
             f"image_count       : {len(list((PKG / 'images').glob('*.png')))}",
             f"files             : {len(files)}",
             ""]
    for p in files:
        lines.append(f"{sha256_file(p)}  {p.relative_to(PKG).as_posix()}")
    (PKG / "PACKAGE_MANIFEST_SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  PACKAGE_MANIFEST_SHA256.txt: {len(files)} files hashed")

    # ---- zip -------------------------------------------------------------------
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(PKG.rglob("*")):
            if p.is_file():
                z.write(p, f"{VERSION}/{p.relative_to(PKG).as_posix()}")
    zh = sha256_file(ZIP_PATH)
    (OUT / f"{VERSION}.zip.sha256").write_text(f"{zh}  {VERSION}.zip\n", encoding="utf-8")
    print(f"  ZIP {ZIP_PATH.name}  {ZIP_PATH.stat().st_size:,} bytes")
    print(f"  SHA256 {zh}")
    return 0


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def locked_manifest_hash() -> str:
    t = (EV / "MANIFEST_SHA256.txt").read_text()
    m = re.search(r"sha256\s*=\s*([0-9a-f]{64})", t)
    return m.group(1) if m else "unknown"


if __name__ == "__main__":
    sys.exit(build())
