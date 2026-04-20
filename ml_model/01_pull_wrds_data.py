"""
Script 1 of 4 — WRDS Data Extraction
======================================
Connects to WRDS and pulls Compustat annual fundamentals (1990–2025)
plus company delisting data for bankruptcy labels.

Run once:
    pip install wrds
    python 01_pull_wrds_data.py

You will be prompted for your WRDS username and password.
Credentials are cached in ~/.pgpass after the first run.
"""

import wrds
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  WRDS Data Extraction — Financial Distress ML Model")
print("=" * 65)
print("\n  Connecting to WRDS (enter your username and password)...\n")

db = wrds.Connection()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Compustat Annual Fundamentals (comp.funda)
# ─────────────────────────────────────────────────────────────────────────────
print("[1/3] Pulling Compustat annual fundamentals (1990–2025)...")
print("      This can take 3–8 minutes for the full dataset.\n")

query_funda = """
    SELECT
        a.gvkey,            -- Global company key (unique firm ID)
        a.datadate,         -- Fiscal year end date
        a.fyear,            -- Fiscal year
        a.sich,             -- SIC industry code
        a.gsector,          -- GICS sector code
        a.at,               -- Total Assets
        a.lt,               -- Total Liabilities Net Minority Interest
        a.wcap,             -- Working Capital (act - lct)
        a.act,              -- Total Current Assets
        a.lct,              -- Total Current Liabilities
        a.re,               -- Retained Earnings
        a.ebit,             -- EBIT (Earnings Before Interest & Tax)
        a.sale,             -- Net Sales / Revenue
        a.csho,             -- Common Shares Outstanding
        a.prcc_f,           -- Stock Price at Fiscal Year End
        a.mkvalt,           -- Market Value of Equity (total)
        a.oancf,            -- Operating Cash Flow
        a.ib,               -- Income Before Extraordinary Items
        a.ni,               -- Net Income
        a.dltt,             -- Long-Term Debt
        a.dlc,              -- Debt in Current Liabilities
        a.dp,               -- Depreciation & Amortization
        a.rect,             -- Receivables (net)
        a.cogs,             -- Cost of Goods Sold
        a.xsga,             -- SG&A Expense
        a.ppent,            -- Net PPE (Property, Plant & Equipment)
        a.che,              -- Cash & Short-Term Investments
        a.gp                -- Gross Profit
    FROM comp.funda a
    WHERE a.fyear BETWEEN 1990 AND 2025
      AND a.indfmt  = 'INDL'   -- Industrial format only
      AND a.datafmt = 'STD'    -- Standard (not restated)
      AND a.popsrc  = 'D'      -- Domestic US companies
      AND a.consol  = 'C'      -- Consolidated statements
      AND a.at      > 0        -- Require positive total assets
      AND a.sale    > 0        -- Require positive sales
"""

funda = db.raw_sql(query_funda, date_cols=["datadate"])
print(f"  ✓ {len(funda):,} firm-year observations")
print(f"  ✓ Unique companies: {funda['gvkey'].nunique():,}")
print(f"  ✓ Years: {int(funda['fyear'].min())} – {int(funda['fyear'].max())}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Company Delisting / Bankruptcy Info (comp.company)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] Pulling company delisting data...")

query_company = """
    SELECT
        c.gvkey,
        c.coname,
        c.dldte,        -- Date company was removed from Compustat
        c.dlrsn         -- Reason code for removal
    FROM comp.company c
"""

company = db.raw_sql(query_company, date_cols=["dldte"])
print(f"  ✓ {len(company):,} company records")

# Show reason codes — user should verify which ones = bankruptcy
print("\n  Delisting reason (dlrsn) distribution:")
dist = company["dlrsn"].value_counts().reset_index()
dist.columns = ["code", "count"]
for _, row in dist.iterrows():
    print(f"    {str(row['code']).ljust(6)} → {int(row['count']):>6,} companies")

print("""
  ── IMPORTANT ──────────────────────────────────────────────────
  Check which dlrsn code means "Bankruptcy" in your WRDS version.
  Common mappings:
    '02' or '04' = Bankruptcy / Chapter 11
    '07'         = Liquidation
  Update BANKRUPTCY_CODES in 02_prepare_features.py accordingly.
  ───────────────────────────────────────────────────────────────
""")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Save to Parquet
# ─────────────────────────────────────────────────────────────────────────────
print("[3/3] Saving data files...")

funda.to_parquet(DATA_DIR / "compustat_funda_1990_2025.parquet",   index=False)
company.to_parquet(DATA_DIR / "compustat_company_delistings.parquet", index=False)

print(f"  ✓ data/compustat_funda_1990_2025.parquet    ({len(funda):,} rows)")
print(f"  ✓ data/compustat_company_delistings.parquet ({len(company):,} rows)")

db.close()

print("""
Done! Next step:
  → Update BANKRUPTCY_CODES in 02_prepare_features.py
  → Run: python 02_prepare_features.py
""")
