# parse_spacetrack_blocks.py
# Produces lists of the form:
# [index, "OBJECT_NAME (OBJECT_ID)", altitude_km, inclination_deg, raan_deg, 1]

import re
from pathlib import Path
from typing import List, Tuple


OrbitList = List[object]  # [int, str, float, float, float, int]


def parse_orbits(raw_text: str) -> List[Tuple[str, float, float, float]]:
    """
    Parse blocks that contain:
      <OBJECT_NAME>...</OBJECT_NAME>
      <OBJECT_ID>...</OBJECT_ID>
      <INCLINATION>...</INCLINATION>
      <RA_OF_ASC_NODE>...</RA_OF_ASC_NODE>
      Altitude ###

    Returns tuples:
      (name_and_id, altitude_km, inclination_deg, raan_deg)
    """
    chunks = re.split(r"(?=<OBJECT_NAME>)", raw_text)

    out: List[Tuple[str, float, float, float]] = []

    for chunk in chunks:
        name = re.search(r"<OBJECT_NAME>\s*(.*?)\s*</OBJECT_NAME>", chunk)
        oid  = re.search(r"<OBJECT_ID>\s*(.*?)\s*</OBJECT_ID>", chunk)
        inc  = re.search(r"<INCLINATION>\s*(.*?)\s*</INCLINATION>", chunk)
        raan = re.search(r"<RA_OF_ASC_NODE>\s*(.*?)\s*</RA_OF_ASC_NODE>", chunk)
        alt  = re.search(r"Altitude\s*([0-9]+(?:\.[0-9]+)?)", chunk)

        # Skip incomplete blocks
        if not (name and oid and inc and raan and alt):
            continue

        name_and_id = f"{name.group(1)} ({oid.group(1)})"

        out.append((
            name_and_id,
            float(alt.group(1)),
            float(inc.group(1)),
            float(raan.group(1)),
        ))

    return out


def dedupe_preserve_order(items):
    """Remove exact duplicates while preserving order."""
    return list(dict.fromkeys(items))


def main():
    # ---------- CHOOSE INPUT METHOD ----------
    # Method A (recommended): put your big paste into a text file and set INPUT_TXT to that filename.
    INPUT_TXT = Path(__file__).parent /"raw_orbits.txt"  # <- create this file and paste your big text into it

    # Method B: paste directly here (only used if reading file fails)
    RAW_FALLBACK = r"""
    <!-- paste big text here if you don't want a .txt file -->
    """

    try:
        with open(INPUT_TXT, "r", encoding="utf-8") as f:
            raw_text = f.read()
        print(f"Loaded input from: {INPUT_TXT}")
    except FileNotFoundError:
        raw_text = RAW_FALLBACK
        print(f"Could not find {INPUT_TXT}; using RAW_FALLBACK string instead.")

    # ---------- PARSE ----------
    parsed = parse_orbits(raw_text)
    print(f"Parsed {len(parsed)} orbit records (including duplicates).")

    # ---------- DEDUPE ----------
    unique = dedupe_preserve_order(parsed)
    print(f"After dedupe: {len(unique)} unique orbit records.")

    # ---------- ADD INDEX AS FIRST ELEMENT AND CREATE LISTS WITH VALUE 1 ----------
    indexed: List[OrbitList] = []
    for idx, (name_and_id, alt, inc, raan) in enumerate(unique, start=1):
        indexed.append([idx, name_and_id, alt, inc, raan, 1])

    # ---------- WRITE AS PYTHON LIST FORMAT ----------
    out_path = "orbits_cleaned.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("orbits = [\n")
        for row in indexed:
            f.write(f"    {row},\n")
        f.write("]\n")

    print(f"Wrote {len(indexed)} orbit records to: {out_path}")

    # Optional: show the first few
    print("\nFirst 5 lists:")
    for row in indexed[:5]:
        print(row)


if __name__ == "__main__":
    main()
