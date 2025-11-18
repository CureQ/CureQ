# -*- coding: utf-8 -*-
"""
Manifest builder (zonder CLI):
- Scan een map voor .h5/.hdf5 (optioneel: recursief of alle bestandstypen)
- Haal datum uit de EERSTE 8 cijfers van de bestandsnaam (YYYYMMDD)
- Converteer naar ISO (YYYY-MM-DD)
- Ken een stabiel numeriek id toe (gesorteerd op date_iso -> filename)
- Schrijf manifest.json (compact) en optioneel manifest_pretty.json

Gebruik:
1) Pas de VARIABELEN hieronder aan (DATA_DIR, OUTPUT_PATH, ...).
2) Run het script.
   Of importeer de functies en roep `build_manifest_from_dir(...)` aan.
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# =========================
# VARIABELEN (pas aan)
# =========================
DATA_DIR       = r"D:\SCA1_A"          # map met je .h5/.hdf5 bestanden
OUTPUT_PATH    = r"./manifest.json"    # pad voor manifest.json
RECURSIVE      = False                 # True = ook submappen doorzoeken
INCLUDE_NONH5  = False                 # True = alle bestandstypen meenemen
WRITE_PRETTY   = False                 # True = ook manifest_pretty.json schrijven

# =========================
# IMPLEMENTATIE
# =========================
DATE_RE = re.compile(r"^(\d{8})")  # eerste 8 cijfers als YYYYMMDD

def extract_date_from_name(fname: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (date_token, date_iso).
    date_token: 'YYYYMMDD' als gevonden aan het BEGIN van de bestandsnaam, anders None
    date_iso:   'YYYY-MM-DD' als token valide, anders None
    """
    m = DATE_RE.match(fname)
    if not m:
        return None, None
    token = m.group(1)
    try:
        dt = datetime.strptime(token, "%Y%m%d").date()
        return token, dt.isoformat()
    except ValueError:
        return token, None

def is_h5(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in (".h5", ".hdf5")

def scan_files(data_dir: str, recursive: bool = False, include_nonh5: bool = False) -> List[str]:
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Geen geldige map: {data_dir}")

    if recursive:
        paths = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                full = os.path.join(root, f)
                if include_nonh5 or is_h5(full):
                    paths.append(full)
        return paths
    else:
        entries = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))
        ]
        return [p for p in entries if (include_nonh5 or is_h5(p))]

def build_records(paths: List[str]) -> List[Dict]:
    records = []
    cwd = os.getcwd()
    for p in paths:
        fname = os.path.basename(p)
        date_token, date_iso = extract_date_from_name(fname)

        # Probeer een relatieve path, maar val terug op None als het een andere drive is
        try:
            path_rel = os.path.relpath(p, start=cwd)
        except ValueError:
            # Bijvoorbeeld: p op D:, cwd op C: → niet relatieve path mogelijk
            path_rel = None

        rec = {
            "filename": fname,
            # absoluut pad (hier werken we primair mee in de CNN-code)
            "path": os.path.abspath(p),
            # optioneel relatieve pad, alleen als het kan
            "path_rel": path_rel,
            "date_token": date_token,
            "date_iso": date_iso,
        }
        records.append(rec)

    # sorteer: date_iso (None gaat achteraan), dan filename
    def sort_key(r):
        k = r["date_iso"] if r["date_iso"] is not None else "9999-12-31"
        return (k, r["filename"])

    records.sort(key=sort_key)

    # geef stabiele numerieke id's
    for i, r in enumerate(records, start=1):
        r["id"] = i

    return records

def write_manifest(records: List[Dict], out_path: str, pretty: bool = False) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[OK] manifest geschreven: {out_path} ({len(records)} items)")

    if pretty:
        pretty_path = os.path.join(os.path.dirname(out_path), "manifest_pretty.json")
        with open(pretty_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[OK] pretty manifest geschreven: {pretty_path}")

def build_manifest_from_dir(
    data_dir: str,
    out_path: str = "manifest.json",
    recursive: bool = False,
    include_nonh5: bool = False,
    write_pretty: bool = True,
) -> List[Dict]:
    """
    High-level helper om in code te gebruiken. Retourneert ook de records.
    """
    paths = scan_files(data_dir, recursive=recursive, include_nonh5=include_nonh5)
    if not paths:
        raise RuntimeError(f"Geen bestanden gevonden in: {data_dir}")
    records = build_records(paths)

    # waarschuwing voor missende/onvalide datums
    missing = [r for r in records if r["date_token"] is None or r["date_iso"] is None]
    if missing:
        print("[WAARSCHUWING] Sommige bestanden missen een geldige leading YYYYMMDD:")
        for r in missing:
            print("   -", r["filename"])

    write_manifest(records, out_path, pretty=write_pretty)
    return records

# =========================
# RUN (zonder terminal)
# =========================
if __name__ == "__main__":
    build_manifest_from_dir(
        data_dir=DATA_DIR,
        out_path=OUTPUT_PATH,
        recursive=RECURSIVE,
        include_nonh5=INCLUDE_NONH5,
        write_pretty=WRITE_PRETTY,
    )
