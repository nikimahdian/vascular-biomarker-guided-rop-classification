"""Download FARFUM-RoP (Normal / Pre-Plus / Plus) from Figshare.

Supports the proposal's 3-class Pre-Plus goal. The main Drive datasets stay
binary; this public set adds expert 3-class labels (1,533 images, 68 patients).

Usage:
    python -m src.data.download_farfum
    python -m src.data.download_farfum --only labels
    python -m src.data.download_farfum --only patients
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import requests

from src.utils.common import ensure_dirs, load_config

COLLECTION_ID = 6721269
API = "https://api.figshare.com/v2"
CHUNK = 1024 * 1024


def log(msg: str) -> None:
    print(msg, flush=True)


def list_articles() -> list[dict]:
    articles: list[dict] = []
    page = 1
    while True:
        r = requests.get(
            f"{API}/collections/{COLLECTION_ID}/articles",
            params={"page": page, "page_size": 100},
            timeout=120,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        articles.extend(batch)
        page += 1
    return articles


def article_files(article_id: int) -> list[dict]:
    r = requests.get(f"{API}/articles/{article_id}/files", timeout=120)
    r.raise_for_status()
    return r.json()


def download_file(url: str, dest: Path, expected_size: int = 0, max_tries: int = 15) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_size and dest.stat().st_size >= expected_size * 0.99:
        log(f"  skip complete {dest.name}")
        return True
    have = dest.stat().st_size if dest.exists() else 0
    for attempt in range(1, max_tries + 1):
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=180) as r:
                if have and r.status_code == 200:
                    have = 0
                    dest.unlink(missing_ok=True)
                if r.status_code not in (200, 206):
                    log(f"  bad status {r.status_code} for {dest.name}")
                    time.sleep(8)
                    continue
                mode = "ab" if have else "wb"
                written = have
                with open(dest, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                target = expected_size or int(r.headers.get("Content-Length") or written)
                if r.status_code == 206 and expected_size:
                    target = expected_size
                if written >= target * 0.99:
                    log(f"  saved {dest.name} ({written / 1e6:.1f} MB)")
                    return True
                have = written
                log(f"  partial {dest.name} ({written / 1e6:.1f}/{target / 1e6:.1f} MB)")
                time.sleep(5)
        except Exception as e:
            log(f"  attempt {attempt} failed {dest.name}: {type(e).__name__}: {e}")
            have = dest.stat().st_size if dest.exists() else 0
            time.sleep(8)
    return False


def extract_rar(archive: Path, out_dir: Path) -> None:
    import shutil
    import subprocess

    marker = out_dir / ".extracted"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer 7-Zip on Windows (rarfile needs UnRAR binary).
    seven = shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe"
    if Path(seven).exists():
        try:
            subprocess.run(
                [seven, "x", "-y", f"-o{out_dir}", str(archive)],
                check=True,
                capture_output=True,
            )
            marker.write_text("ok", encoding="utf-8")
            log(f"  extracted {archive.name}")
            return
        except Exception as e:
            log(f"  [warn] 7-Zip extract failed {archive.name}: {e}")

    try:
        import rarfile  # optional dependency

        with rarfile.RarFile(archive) as rf:
            rf.extractall(out_dir)
        marker.write_text("ok", encoding="utf-8")
        log(f"  extracted {archive.name}")
    except Exception:
        log(f"  [warn] could not auto-extract {archive.name} (install rarfile or 7-Zip)")



def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw_dir"])
    out_root = raw / "farfum_rop"
    labels_dir = out_root / "metadata"
    patients_dir = out_root / "patients"
    archives_dir = out_root / "archives"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=["labels", "patients", "all"],
        default="all",
        help="labels = xlsx metadata only; patients = 68 patient rars",
    )
    args = ap.parse_args()

    articles = list_articles()
    log(f"[farfum] {len(articles)} articles in collection {COLLECTION_ID}")

    if args.only in ("labels", "all"):
        for art in articles:
            title = art["title"]
            if "Label" not in title and "Detail" not in title:
                continue
            for f in article_files(art["id"]):
                dest = labels_dir / f["name"]
                if dest.exists() and dest.stat().st_size >= f.get("size", 0) * 0.99:
                    log(f"[skip] {dest.name}")
                    continue
                log(f"[download] {title} -> {dest.name}")
                download_file(f["download_url"], dest, expected_size=f.get("size", 0))

    if args.only in ("patients", "all"):
        failed: list[str] = []
        for art in articles:
            title = art["title"]
            if not title.lower().endswith(".rar"):
                continue
            for f in article_files(art["id"]):
                dest = archives_dir / f["name"]
                expected = int(f.get("size") or 0)
                if dest.exists() and expected and dest.stat().st_size >= expected * 0.99:
                    log(f"[skip] {dest.name}")
                else:
                    log(f"[download] {title}")
                    if not download_file(f["download_url"], dest, expected_size=expected):
                        failed.append(dest.name)
                patient_out = patients_dir / dest.stem
                if dest.suffix.lower() == ".rar" and dest.exists():
                    extract_rar(dest, patient_out)
        if failed:
            log(f"[warn] failed archives ({len(failed)}): {', '.join(failed)}")
            raise SystemExit(1)

    log(f"[done] FARFUM-RoP in {out_root}")


if __name__ == "__main__":
    main()
