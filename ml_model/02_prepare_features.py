"""
Script 2 of 4 — Feature Engineering
=====================================
Reads the raw CSV files downloaded from WRDS/Compustat,
creates forward-looking bankruptcy labels, engineers all features
used by the Z/O/F/M models plus YoY delta ratios, and saves
the cleaned dataset ready for XGBoost training.

Run:
    python 02_prepare_features.py

Output:
    data/ml_dataset.parquet
    data/feature_cols.json
    data/feature_medians.json
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

print("=" * 60)
print("  Feature Engineering — Financial Distress ML Model")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading raw data...")

funda   = pd.read_csv(DATA_DIR / "compustat_funda.csv",   low_memory=False)
company = pd.read_csv(DATA_DIR / "compustat_company.csv", low_memory=False)

print(f"  Funda  : {len(funda):,} rows, {funda['gvkey'].nunique():,} companies")
print(f"  Company: {len(company):,} rows, {company['gvkey'].nunique():,} companies")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Clean fundamentals
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Cleaning fundamentals...")

df = funda.copy()
for col, val in [("consol","C"), ("datafmt","STD"), ("indfmt","INDL"), ("curcd","USD")]:
    if col in df.columns:
        df = df[df[col] == val]

df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
df = df.dropna(subset=["datadate", "gvkey", "fyear", "at"])
df = df[df["at"] > 0]
df = df[df["fyear"].between(1989, 2024)]
df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)
df = df.drop_duplicates(subset=["gvkey", "fyear"], keep="last")

print(f"  After cleaning: {len(df):,} rows, {df['gvkey'].nunique():,} companies")
print(f"  Year range: {int(df['fyear'].min())} – {int(df['fyear'].max())}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build bankruptcy labels
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Building bankruptcy labels...")

bk = company[company["dlrsn"].isin([2, 2.0, "2", "02"])].copy()
bk["dldte"] = pd.to_datetime(bk["dldte"], errors="coerce")
bk = bk.dropna(subset=["dldte"])
bk = bk.groupby("gvkey")["dldte"].min().reset_index()
bk.columns = ["gvkey", "bankrupt_date"]
print(f"  Bankrupt companies: {len(bk):,}")

df = df.merge(bk, on="gvkey", how="left")

for horizon, days in [("1yr", 365), ("2yr", 730), ("3yr", 1095)]:
    df[f"distress_{horizon}"] = (
        df["bankrupt_date"].notna() &
        (df["bankrupt_date"] > df["datadate"]) &
        ((df["bankrupt_date"] - df["datadate"]).dt.days <= days)
    ).astype(int)

print(f"  Distress 1yr rate: {df['distress_1yr'].mean():.3%}")
print(f"  Distress 2yr rate: {df['distress_2yr'].mean():.3%}")
print(f"  Distress 3yr rate: {df['distress_3yr'].mean():.3%}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Engineer features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Engineering features...")

# Ensure numeric
num_cols = ["at","act","lct","lt","re","dp","ebit","sale","ni","ib",
            "oancf","dltt","dlc","rect","cogs","xsga","ppent","che","ceq","csho"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        df[c] = np.nan

# Market cap proxy (use book equity when price not available)
if "prcc_f" in df.columns:
    df["prcc_f"] = pd.to_numeric(df["prcc_f"], errors="coerce")
    df["mkt_cap"] = (df["csho"] * df["prcc_f"]).clip(lower=0)
else:
    df["mkt_cap"] = df["ceq"].clip(lower=0)

df["wcap"] = df["act"] - df["lct"]

# ── Prior-year (lag) values via shift within company group ───────────────────
lag_cols = ["at","sale","ni","oancf","dltt","dlc","act","lct","ceq",
            "cogs","dp","ppent","xsga","lt","rect","csho"]
df = df.sort_values(["gvkey","fyear"]).reset_index(drop=True)
for c in lag_cols:
    if c in df.columns:
        df[f"{c}_lag"] = df.groupby("gvkey")[c].shift(1)

def g(col):
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)

# ── Altman Z-Score components ────────────────────────────────────────────────
z_x1 = g("wcap")    / g("at")
z_x2 = g("re")      / g("at")
z_x3 = g("ebit")    / g("at")
z_x4 = g("mkt_cap") / g("lt").replace(0, np.nan)
z_x5 = g("sale")    / g("at")
z_sc = 1.2*z_x1 + 1.4*z_x2 + 3.3*z_x3 + 0.6*z_x4 + 1.0*z_x5

# ── Ohlson O-Score components ────────────────────────────────────────────────
ib   = g("ib").fillna(g("ni"))
o_x1 = np.log(g("at").clip(lower=0.01))
o_x2 = g("lt")   / g("at").replace(0, np.nan)
o_x3 = g("wcap") / g("at").replace(0, np.nan)
o_x4 = (g("lt") > g("at")).astype(float)
o_x5 = (ib + g("dp")) / g("lt").replace(0, np.nan)
o_x6 = ib / g("at").replace(0, np.nan)
o_x7 = (ib + g("dp")) / g("at").replace(0, np.nan)
ni_l = g("ni_lag")
o_x8 = ((ib < 0) & (ni_l < 0)).astype(float)
denom9 = ib.abs() + ni_l.abs()
o_x9 = (ib - ni_l) / denom9.replace(0, np.nan)
o_raw = (-1.32 - 0.407*o_x1 + 6.03*o_x2 - 1.43*o_x3 + 0.076*o_x4
         - 1.72*o_x5 - 2.37*o_x6 - 1.83*o_x7 + 0.285*o_x8 - 0.521*o_x9)
o_prob = 1 / (1 + np.exp(-o_raw.clip(-50, 50)))

# ── Piotroski F-Score components ─────────────────────────────────────────────
roa0  = g("ni")   / g("at").replace(0, np.nan)
roa1  = g("ni_lag") / g("at_lag").replace(0, np.nan)
cfo0  = g("oancf")
cfo_ta= cfo0 / g("at").replace(0, np.nan)
lev0  = g("dltt")     / g("at").replace(0, np.nan)
lev1  = g("dltt_lag") / g("at_lag").replace(0, np.nan)
cr0   = g("act")     / g("lct").replace(0, np.nan)
cr1   = g("act_lag") / g("lct_lag").replace(0, np.nan)
gm0   = (g("sale") - g("cogs")) / g("sale").replace(0, np.nan)
gm1   = (g("sale_lag") - g("cogs_lag")) / g("sale_lag").replace(0, np.nan)
at0   = g("sale")     / g("at").replace(0, np.nan)
at1   = g("sale_lag") / g("at_lag").replace(0, np.nan)

f1 = (roa0 > 0).astype(float)
f2 = (cfo0 > 0).astype(float)
f3 = np.where(roa1.notna(), (roa0 > roa1).astype(float), 0.0)
f4 = (cfo_ta > roa0).astype(float)
f5 = np.where(lev1.notna(), (lev0 < lev1).astype(float), 0.0)
f6 = np.where(cr1.notna(),  (cr0  > cr1 ).astype(float), 0.0)
f7 = np.where(g("csho_lag").notna(), (g("csho") <= g("csho_lag")).astype(float), 0.0)
f8 = np.where(gm1.notna(), (gm0 > gm1).astype(float), 0.0)
f9 = np.where(at1.notna(), (at0 > at1).astype(float), 0.0)
f_sc = f1+f2+f3+f4+f5+f6+f7+f8+f9

# ── Beneish M-Score components ────────────────────────────────────────────────
dsri = (g("rect") / g("sale").replace(0,np.nan)) / (g("rect_lag") / g("sale_lag").replace(0,np.nan))
gmi  = ((g("sale_lag")-g("cogs_lag")) / g("sale_lag").replace(0,np.nan)) / ((g("sale")-g("cogs")) / g("sale").replace(0,np.nan))
nonpp0 = g("at") - g("ppent") - g("che")
nonpp1 = g("at_lag") - g("ppent_lag")
aqi  = (nonpp0/g("at").replace(0,np.nan)) / (nonpp1/g("at_lag").replace(0,np.nan))
sgi  = g("sale") / g("sale_lag").replace(0,np.nan)
dep0 = g("dp") / (g("dp")+g("ppent")).replace(0,np.nan)
dep1 = g("dp_lag") / (g("dp_lag")+g("ppent_lag")).replace(0,np.nan)
depi = dep1 / dep0.replace(0,np.nan)
sgai = (g("xsga")/g("sale").replace(0,np.nan)) / (g("xsga_lag")/g("sale_lag").replace(0,np.nan))
tata = (ib - g("oancf")) / g("at").replace(0,np.nan)
dbt0 = (g("dltt").fillna(0)+g("dlc").fillna(0)) / g("at").replace(0,np.nan)
dbt1 = (g("dltt_lag").fillna(0)+g("dlc_lag").fillna(0)) / g("at_lag").replace(0,np.nan)
lvgi = dbt0 / dbt1.replace(0,np.nan)
m_sc = (-4.84 + 0.920*dsri + 0.528*gmi + 0.404*aqi + 0.892*sgi
        + 0.115*depi - 0.172*sgai + 4.679*tata - 0.327*lvgi)

# ── Additional ratios ─────────────────────────────────────────────────────────
delta_sale = (g("sale") - g("sale_lag")) / g("sale_lag").replace(0,np.nan)
delta_at   = (g("at")   - g("at_lag"))   / g("at_lag").replace(0,np.nan)

feat_dict = {
    "z_x1": z_x1, "z_x2": z_x2, "z_x3": z_x3, "z_x4": z_x4, "z_x5": z_x5,
    "z_score": z_sc,
    "o_x1": o_x1, "o_x2": o_x2, "o_x3": o_x3, "o_x4": o_x4, "o_x5": o_x5,
    "o_x6": o_x6, "o_x7": o_x7, "o_x8": o_x8, "o_x9": o_x9,
    "o_score_raw": o_raw, "o_prob": o_prob,
    "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
    "f6": f6, "f7": f7, "f8": f8, "f9": f9, "f_score": f_sc,
    "dsri": dsri, "gmi": gmi, "aqi": aqi, "sgi": sgi, "depi": depi,
    "sgai": sgai, "tata": tata, "lvgi": lvgi, "m_score": m_sc,
    "roa": roa0, "cfo_ta": cfo_ta, "cr": cr0, "lev": lev0,
    "gm": gm0, "asset_turn": at0,
    "delta_roa":  roa0 - roa1,
    "delta_cr":   cr0  - cr1,
    "delta_lev":  lev0 - lev1,
    "delta_gm":   gm0  - gm1,
    "delta_sale": delta_sale,
    "delta_at":   delta_at,
    "debt_ratio":   g("lt")  / g("at").replace(0,np.nan),
    "equity_ratio": g("ceq") / g("at").replace(0,np.nan),
    "log_at":   np.log(g("at").clip(lower=0.01)),
    "log_sale": np.log(g("sale").clip(lower=0.01)),
}

feat_df = pd.DataFrame(feat_dict, index=df.index)
FEATURE_COLS = list(feat_df.columns)
print(f"  Features engineered: {len(FEATURE_COLS)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Assemble and clean
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Assembling and cleaning dataset...")

meta_cols = ["gvkey", "fyear", "datadate", "distress_1yr", "distress_2yr", "distress_3yr"]
avail_meta = [c for c in meta_cols if c in df.columns]
out = pd.concat([df[avail_meta].reset_index(drop=True),
                 feat_df.reset_index(drop=True)], axis=1)

# Winsorize at 1%–99%
for col in FEATURE_COLS:
    lo = out[col].quantile(0.01)
    hi = out[col].quantile(0.99)
    out[col] = out[col].clip(lower=lo, upper=hi)

# Medians from training set (pre-2016) for imputation
train_mask = out["fyear"] <= 2015
medians = out.loc[train_mask, FEATURE_COLS].median().to_dict()
for col in FEATURE_COLS:
    out[col] = out[col].fillna(medians.get(col, 0.0))

out = out.dropna(subset=["distress_1yr"]).reset_index(drop=True)

print(f"  Final dataset  : {len(out):,} rows")
print(f"  Companies      : {out['gvkey'].nunique():,}")
print(f"  Distress 1yr   : {out['distress_1yr'].mean():.3%}")
print(f"  Year range     : {int(out['fyear'].min())} – {int(out['fyear'].max())}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Saving...")

out.to_csv(DATA_DIR / "ml_dataset.csv", index=False)
with open(DATA_DIR / "feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
with open(DATA_DIR / "feature_medians.json", "w") as f:
    json.dump({k: float(v) for k, v in medians.items()}, f, indent=2)

print("  ✓ data/ml_dataset.csv")
print(f"  ✓ data/feature_cols.json  ({len(FEATURE_COLS)} features)")
print("  ✓ data/feature_medians.json")
print("\n  Next step: python 03_train_model.py")
