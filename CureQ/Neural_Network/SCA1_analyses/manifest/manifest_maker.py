# make_manifest_simple.py
# ------------------------------------------------------------
# Genereert manifest.json met:
#  - id = oplopend (1, 2, 3, ...)
#  - date = "YYYY-MM-DD" (geen tijd)
#  - alleen bestandsnamen (geen paden)
#  - koppelt well- en electrode-bestanden via datum in bestandsnaam
# ------------------------------------------------------------

from pathlib import Path
import json, re
import pandas as pd

# === Vul hier de mappen in ===
WELLS_DIR       = r"C:\Users\chenp\Documents\SCA1_Features\Well"
ELECTRODES_DIR  = r"C:\Users\chenp\Documents\SCA1_Features\Electrode"
WELLS_PATTERN   = "*Features.csv"
ELECTRODES_PATTERN = "*Electrode_Features.csv"
REQUIRE_BOTH    = True
WELLS_COUNT     = 48
ELECS_PER_WELL  = 16
# ==============================

SCRIPT_DIR   = Path(__file__).resolve().parent
OUT_MANIFEST = SCRIPT_DIR / "manifest.json"

DATE_PREFIX_RE = re.compile(r"^(\d{8})")  # YYYYMMDD
BATCH_RE = re.compile(
    r"^\d{8}[_-]([^_]+?)(?:_rechunked_output|_output|_|\.)",
    flags=re.IGNORECASE,
)

def parse_date_prefix(name: str):
    """Haal datum uit voorste 8 cijfers: return ('YYYYMMDD', 'YYYY-MM-DD')."""
    m = DATE_PREFIX_RE.match(name)
    if not m:
        return None, None
    yyyymmdd = m.group(1)
    try:
        dt = pd.to_datetime(yyyymmdd, format="%Y%m%d")
        return yyyymmdd, dt.strftime("%Y-%m-%d")
    except Exception:
        return None, None

def parse_batch_key(name: str) -> str:
    """Haal batch key zoals 'SCA1(000)' uit bestandsnaam."""
    m = BATCH_RE.match(name)
    if m:
        return m.group(1)
    m2 = re.match(r"^\d{8}[_-]([^_]+)", name)
    return m2.group(1) if m2 else ""

def collect_files(base_dir: Path, pattern: str):
    """Zoek CSV-bestanden en geef lijst met (fname, date_id, date_str, batch_key)."""
    base_dir = base_dir.resolve()
    results = []
    for f in base_dir.rglob(pattern):
        if not f.is_file() or f.suffix.lower() != ".csv":
            continue
        name = f.name
        date_id, date_str = parse_date_prefix(name)
        if not date_id:
            continue
        batch_key = parse_batch_key(name)
        results.append((f.name, date_id, date_str, batch_key))
    return results

def main():
    wells_list = collect_files(Path(WELLS_DIR), WELLS_PATTERN)
    elecs_list = collect_files(Path(ELECTRODES_DIR), ELECTRODES_PATTERN)

    wells_map, elecs_map = {}, {}
    for fname, date_id, date_str, batch in wells_list:
        wells_map.setdefault((date_id, batch), []).append((fname, date_str))
    for fname, date_id, date_str, batch in elecs_list:
        elecs_map.setdefault((date_id, batch), []).append((fname, date_str))

    keys = set(wells_map.keys()) | set(elecs_map.keys())

    measurements = []
    for i, (date_id, batch) in enumerate(sorted(keys, key=lambda k: (k[0], k[1])), start=1):
        well_csv = None
        if (date_id, batch) in wells_map:
            well_csv = sorted([n for n, _ in wells_map[(date_id, batch)]])[0]
        electrode_csv = None
        if (date_id, batch) in elecs_map:
            electrode_csv = sorted([n for n, _ in elecs_map[(date_id, batch)]])[0]
        if REQUIRE_BOTH and (well_csv is None or electrode_csv is None):
            continue

        # gebruik datum uit de prefix
        _, date_str = parse_date_prefix(date_id + "_x.csv")
        measurements.append({
            "id": i,                 # oplopend nummer
            "date": date_str,        # enkel datum, geen tijd
            "well_csv": well_csv,
            "electrode_csv": electrode_csv
        })

    manifest = {
        "measurements": measurements,
        "grid": {"wells": WELLS_COUNT, "electrodes_per_well": ELECS_PER_WELL}
    }

    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"✅ Manifest geschreven: {OUT_MANIFEST}")
    print(f"📦 Metingen: {len(measurements)}")
    if REQUIRE_BOTH:
        missing = [m for m in measurements if not m['well_csv'] or not m['electrode_csv']]
        if missing:
            print(f"⚠️ {len(missing)} incomplete paren zijn gefilterd.")

if __name__ == "__main__":
    main()
