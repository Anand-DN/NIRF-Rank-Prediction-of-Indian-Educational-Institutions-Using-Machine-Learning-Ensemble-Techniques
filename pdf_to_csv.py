"""
NIRF PDF → CSV Extractor  (Fixed & Production-Ready)
=====================================================
Fixes:
  1. Dual-mode extraction: structured table parsing + fallback regex text parsing
  2. Correct NIRF column mapping (SS, FSR, FQE, FRU, TLR, RPC, GO, OI, Score, Rank)
  3. Handles multi-page PDFs where ranking table spans pages
  4. Robust institute name & rank cleaning
  5. Deduplication across years
  6. Saves a rich CSV with all sub-parameters

Usage:
    python pdf_to_csv.py
    # expects: data/pdfs/<year>/<any>.pdf
    # writes:  data/csv/final_nirf_dataset.csv
"""

import pdfplumber
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

PDF_ROOT  = Path("data/pdfs")
OUTPUT    = Path("data/final.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── Canonical column aliases ──────────────────────────────────────────────────
# NIRF changed column headers across years; map every variant to a canonical name
COLUMN_ALIASES = {
    # Rank
    "rank": "Rank",
    "ranking": "Rank",
    # Institute
    "institute name": "Institute",
    "name of institution": "Institute",
    "institution": "Institute",
    "college": "Institute",
    # State / City
    "city": "City",
    "state": "State",
    # TLR sub-params
    "ss": "SS", "ss/20": "SS", "student strength": "SS",
    "fsr": "FSR", "fsr/30": "FSR", "faculty student ratio": "FSR",
    "fqe": "FQE", "fqe/20": "FQE", "faculty qualification": "FQE",
    "fru": "FRU", "fru/30": "FRU", "financial resources": "FRU",
    # Main categories
    "tlr": "TLR", "tlr/100": "TLR", "teaching learning resources": "TLR",
    # RPC sub-params
    "pu": "PU", "pu/35": "PU", "publications": "PU",
    "qp": "QP", "qp/40": "QP", "quality publications": "QP",
    "ipr": "IPR", "ipr/15": "IPR", "intellectual property": "IPR",
    "fppp": "FPPP", "fppp/10": "FPPP", "footprint of projects": "FPPP",
    "rpc": "RPC", "rpc/100": "RPC", "research professional practice": "RPC",
    # GO sub-params
    "gph": "GPH", "gph/40": "GPH",
    "gue": "GUE", "gue/15": "GUE",
    "ms": "MS", "ms/25": "MS",
    "gphd": "GPhD", "gphd/20": "GPhD",
    "go": "GO", "go/100": "GO", "graduation outcomes": "GO",
    # OI sub-params
    "rd": "RD", "rd/30": "RD",
    "wd": "WD", "wd/30": "WD",
    "escs": "ESCS", "escs/20": "ESCS",
    "pcs": "PCS", "pcs/20": "PCS",
    "oi": "OI", "oi/100": "OI", "outreach inclusivity": "OI",
    # Perception
    "perception": "Perception", "pr": "Perception", "pr/100": "Perception",
    # Score
    "score": "Score", "total score": "Score",
}

NUMERIC_COLS = [
    "Rank", "SS", "FSR", "FQE", "FRU", "TLR",
    "PU", "QP", "IPR", "FPPP", "RPC",
    "GPH", "GUE", "MS", "GPhD", "GO",
    "RD", "WD", "ESCS", "PCS", "OI",
    "Perception", "Score",
]

FINAL_COLS = [
    "Year", "Rank", "Institute", "City", "State",
    "SS", "FSR", "FQE", "FRU", "TLR",
    "PU", "QP", "IPR", "FPPP", "RPC",
    "GPH", "GUE", "MS", "GPhD", "GO",
    "RD", "WD", "ESCS", "PCS", "OI",
    "Perception", "Score",
    "Type", "Rank_Category",
]


# ── Helper: normalise a raw column name to canonical ─────────────────────────
def canonical(col: str) -> str:
    key = str(col).strip().lower().replace("\n", " ").replace("  ", " ")
    return COLUMN_ALIASES.get(key, col.strip())


# ── Helper: clean a numeric string ───────────────────────────────────────────
def to_float(val):
    try:
        return float(str(val).replace(",", "").replace("–", "").strip())
    except Exception:
        return np.nan


# ── Method 1: extract via pdfplumber table detection ─────────────────────────
def extract_via_tables(pdf_path: Path, year: int) -> list[dict]:
    records = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Collect all pages' table rows into a flat list
            header = None
            rows   = []

            for page in pdf.pages:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 5,
                    }
                )
                # Fallback to default strategy if nothing found
                if not tables:
                    tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    # Detect header row (contains "Rank" or "Institute")
                    for i, row in enumerate(table):
                        row_text = " ".join(str(c) for c in row if c).lower()
                        if "rank" in row_text or "institute" in row_text:
                            # Use this as header
                            header = [canonical(c) for c in row]
                            rows.extend(table[i + 1:])
                            break
                    else:
                        # No header row on this page — treat all as data
                        if header:
                            rows.extend(table)

            if not header or not rows:
                return []

            for row in rows:
                if not row or all(c is None or str(c).strip() == "" for c in row):
                    continue
                # Pad / trim row to match header length
                row = list(row) + [None] * (len(header) - len(row))
                row = row[: len(header)]
                rec = dict(zip(header, row))

                # Must have a valid Rank
                rank_val = to_float(rec.get("Rank"))
                if np.isnan(rank_val) or rank_val < 1 or rank_val > 500:
                    continue

                # Must have institute name
                institute = str(rec.get("Institute", "")).strip()
                if not institute or institute.lower() in ("none", "", "institute name"):
                    continue

                rec["Year"]  = year
                rec["Rank"]  = int(rank_val)
                rec["Institute"] = institute
                records.append(rec)

    except Exception as e:
        print(f"  [table] Error in {pdf_path.name}: {e}")

    return records


# ── Method 2: regex-based fallback for text-heavy PDFs ───────────────────────
_RANK_LINE = re.compile(
    r"^(\d{1,3})\s+"              # rank
    r"([A-Z][A-Za-z\s,'\.&\-]+?)"  # institute name (greedy to next number)
    r"\s+([\d\.]+)"               # first score
    r"(?:\s+([\d\.]+))?"          # optional second score
    r"(?:\s+([\d\.]+))?",         # optional third score
    re.MULTILINE,
)

def extract_via_text(pdf_path: Path, year: int) -> list[dict]:
    records = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )

        for m in _RANK_LINE.finditer(full_text):
            rank_val = int(m.group(1))
            if rank_val > 500:
                continue
            rec = {
                "Year": year,
                "Rank": rank_val,
                "Institute": m.group(2).strip(),
                "Score": to_float(m.group(5) or m.group(4) or m.group(3)),
            }
            records.append(rec)

    except Exception as e:
        print(f"  [text] Error in {pdf_path.name}: {e}")

    return records


# ── Institute type classifier ─────────────────────────────────────────────────
def classify_institute(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    n = name.lower()
    if any(t in n for t in ["iit ", "indian institute of technology"]):
        return "IIT"
    if any(t in n for t in ["nit ", "national institute of technology"]):
        return "NIT"
    if any(t in n for t in ["iiit ", "indian institute of information"]):
        return "IIIT"
    if any(t in n for t in ["iiser", "iim ", "iipe", "iisc"]):
        return "Central Institute"
    if "central university" in n or "university of" in n:
        return "Central University"
    if "deemed" in n:
        return "Deemed University"
    if "university" in n:
        return "State University"
    return "Private"


# ── Main extraction loop ──────────────────────────────────────────────────────
all_records: list[dict] = []

year_dirs = sorted(
    [d for d in PDF_ROOT.iterdir() if d.is_dir() and d.name.isdigit()]
)

if not year_dirs:
    print(f"[ERROR] No year folders found under {PDF_ROOT.resolve()}")
    print("Expected structure:  data/pdfs/2024/xxx.pdf")
    exit(1)

for year_dir in year_dirs:
    year = int(year_dir.name)
    pdfs = list(year_dir.glob("*.pdf"))
    print(f"\nYear {year}: {len(pdfs)} PDFs found")

    year_records: list[dict] = []

    for pdf_path in pdfs:
        recs = extract_via_tables(pdf_path, year)
        if not recs:
            recs = extract_via_text(pdf_path, year)
        if recs:
            year_records.extend(recs)
        else:
            print(f"  ⚠ No data from {pdf_path.name}")

    print(f"  → {len(year_records)} rows extracted")
    all_records.extend(year_records)

if not all_records:
    print("[ERROR] No records extracted from any PDF. Check your PDF structure.")
    exit(1)

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame(all_records)

# Canonicalise any lingering column names that slipped through
df.rename(columns={c: canonical(c) for c in df.columns}, inplace=True)
df = df.loc[:, ~df.columns.duplicated()]          # drop duplicate cols

# Convert numerics
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = df[col].apply(to_float)
    else:
        df[col] = np.nan

# ── Correct Ranking per year ──────────────────────────────────────────────────
# If Score exists → re-rank by Score descending within each year
# This is the ROOT FIX for "ranking not proper"
def rerank(group: pd.DataFrame) -> pd.DataFrame:
    if group["Score"].notna().sum() > group["Score"].isna().sum():
        group = group.sort_values("Score", ascending=False)
    else:
        # Fallback: sort by existing Rank ascending
        group = group.sort_values("Rank", ascending=True)
    group["Rank"] = range(1, len(group) + 1)
    return group

df = df.groupby("Year", group_keys=False).apply(rerank)

# ── Deduplication: keep highest-score row per (Year, Institute) ───────────────
df.sort_values(["Year", "Score"], ascending=[True, False], inplace=True)
df.drop_duplicates(subset=["Year", "Institute"], keep="first", inplace=True)

# ── Derived features ──────────────────────────────────────────────────────────
df["Type"] = df["Institute"].apply(classify_institute)

df["Rank_Category"] = pd.cut(
    df["Rank"],
    bins=[0, 10, 25, 50, 100, 200, 9999],
    labels=["Top 10", "Top 25", "Top 50", "Top 100", "Top 200", "Others"],
)

# Research intensity: RPC relative to TLR (avoid div/0)
if "RPC" in df.columns and "TLR" in df.columns:
    df["Research_Intensity"] = (df["RPC"] / df["TLR"].replace(0, np.nan)).round(4)

# ── Final column order ────────────────────────────────────────────────────────
present = [c for c in FINAL_COLS if c in df.columns]
extra   = [c for c in df.columns if c not in FINAL_COLS]
df = df[present + extra]

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)

print("\n" + "="*60)
print(f"✅  FINAL DATASET SAVED → {OUTPUT}")
print(f"   Rows   : {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   Years  : {sorted(df['Year'].unique().tolist())}")
print(f"   Institutes: {df['Institute'].nunique()} unique")
print("="*60)
print(df[["Year", "Rank", "Institute", "Score"]].head(15).to_string(index=False))