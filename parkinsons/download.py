"""Fetch the two REAL Parkinson's voice datasets into data/.

  1. UCI Parkinsons (Little et al. 2007)        -> data/parkinsons.data       (40 KB, committed)
  2. UCI PD Classification (Sakar et al. 2018)  -> data/pd_speech_features.csv (5 MB, fetched)

Dataset 2 ships as a .rar inside a .zip; we extract it with bsdtar (libarchive, present on
macOS and most Linux). No third-party Python deps.

Run:  cd ~/mentat && PYTHONPATH=. ~/swechats/.venv/bin/python -m reference.parkinsons.download
"""
from __future__ import annotations

import subprocess
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).parent / "data"
LITTLE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
SAKAR_ZIP = "https://archive.ics.uci.edu/static/public/470/parkinson+s+disease+classification.zip"


def _get(url: str, dest: Path):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:       # noqa: S310 (trusted UCI host)
        dest.write_bytes(r.read())
    print(f"  -> {dest}  ({dest.stat().st_size // 1024} KB)")


def fetch_little():
    dest = DATA / "parkinsons.data"
    if dest.exists():
        print(f"  ok  {dest.name} already present")
        return
    _get(LITTLE_URL, dest)


def fetch_sakar():
    csv = DATA / "pd_speech_features.csv"
    if csv.exists():
        print(f"  ok  {csv.name} already present")
        return
    zip_path = DATA / "_sakar.zip"
    _get(SAKAR_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as z:                      # the zip holds a single .rar
        rar_name = next(n for n in z.namelist() if n.endswith(".rar"))
        z.extract(rar_name, DATA)
    rar_path = DATA / rar_name
    # bsdtar (libarchive) reads .rar without a separate unrar binary
    subprocess.run(["bsdtar", "-xf", rar_path.name], cwd=DATA, check=True)
    rar_path.unlink(missing_ok=True)
    zip_path.unlink(missing_ok=True)
    print(f"  -> {csv.name}  ({csv.stat().st_size // 1024} KB)")


def main() -> int:
    DATA.mkdir(exist_ok=True)
    print("Fetching real Parkinson's voice datasets...")
    fetch_little()
    try:
        fetch_sakar()
    except (StopIteration, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  WARN  could not prepare the Sakar dataset ({e}). detect_sakar.py will skip.")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
