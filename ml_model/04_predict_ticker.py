"""
Script 4 of 4 — Real-Time Ticker Prediction
=============================================
Uses the trained Logistic Regression model (03_train_model.py)
to compute a distress probability for any publicly traded company
via yfinance. No external ML libraries required — pure NumPy.

Also exposes  predict_ticker(ticker)  as an importable function
so the Streamlit app can call it directly.

Usage (standalone):
    python 04_predict_ticker.py AAPL
    python 04_predict_ticker.py GE     # historically distressed

Prerequisites:
    pip install yfinance
    python 03_train_model.py    # model must be trained first
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

_HERE     = Path(__file__).parent
MODEL_DIR = _HERE / "model"
DATA_DIR  = _HERE / "data"

# ── Module-level cache ────────────────────────────────────────────────────────
_bundle    = None   # full model JSON
_feat_meds = None   # training-set medians for imputation
_threshold = None

def _load_artefacts() -> bool:
    global _bundle, _feat_meds, _threshold
    if _bundle is not None:
        return True
    model_path = MODEL_DIR / "distress_model.json"
    if not model_path.exists():
        return False
    with open(model_path) as f:
        _bundle = json.load(f)
    medians_path = DATA_DIR / "feature_medians.json"
    if medians_path.exists():
        with open(medians_path) as f:
            _feat_meds = json.load(f)
    else:
        _feat_meds = {}
    thr_path = MODEL_DIR / "threshold.json"
    _threshold = 0.5
    if thr_path.exists():
        with open(thr_path) as f:
            _threshold = json.load(f).get("threshold", 0.5)
    return True


def _predict_raw(feature_vector: list) -> float:
    """Run logistic regression on a standardised feature vector."""
    w   = np.array(_bundle["weights"], dtype=np.float64)
    b   = float(_bundle["bias"])
    mu  = np.array(_bundle["feature_mu"],  dtype=np.float64)
    std = np.array(_bundle["feature_std"], dtype=np.float64)
    x   = (np.array(feature_vector, dtype=np.float64) - mu) / std
    logit = float(np.dot(x, w) + b)
    logit = max(-50.0, min(50.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


# ── Financial ratio helpers ───────────────────────────────────────────────────
def _s(x, default=math.nan):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except: return default

def _d(a, b):
    a, b = _s(a), _s(b)
    if math.isnan(b) or b == 0 or math.isnan(a): return math.nan
    return a / b


def _get_fundamentals(ticker: str):
    tk = yf.Ticker(ticker)
    info = {}
    try: info = tk.info or {}
    except: pass

    def gi(row, col=0):
        try:
            inc = tk.income_stmt
            if inc is None or inc.empty or row not in inc.index: return math.nan
            return _s(inc.loc[row].iloc[col])
        except: return math.nan

    def gb(row, col=0):
        try:
            bs = tk.balance_sheet
            if bs is None or bs.empty or row not in bs.index: return math.nan
            return _s(bs.loc[row].iloc[col])
        except: return math.nan

    def gc(row, col=0):
        try:
            cf = tk.cashflow
            if cf is None or cf.empty or row not in cf.index: return math.nan
            return _s(cf.loc[row].iloc[col])
        except: return math.nan

    def _p(*keys, src="bs", col=0):
        fn = {"bs":gb,"inc":gi,"cf":gc}[src]
        for k in keys:
            v = fn(k, col)
            if not math.isnan(v): return v
        return math.nan

    at0  = _p("Total Assets","TotalAssets")
    lt0  = _p("Total Liabilities Net Minority Interest","Total Liab","TotalLiabilities")
    act0 = _p("Current Assets","Total Current Assets","CurrentAssets")
    lct0 = _p("Current Liabilities","Total Current Liabilities","CurrentLiabilities")
    re0  = _p("Retained Earnings","RetainedEarnings")
    ebit0= _p("EBIT","Ebit",src="inc")
    sale0= _p("Total Revenue","Revenue","Net Revenues",src="inc")
    ni0  = _p("Net Income","NetIncome",src="inc")
    ib0  = _p("Net Income Common Stockholders","Net Income","NetIncome",src="inc")
    cfo0 = _p("Operating Cash Flow","Cash Flow From Operations",src="cf")
    dltt0= _p("Long Term Debt","LongTermDebt")
    dlc0 = _p("Current Debt","Short Long Term Debt","Short Term Borrowings","CurrentDebt")
    dp0  = _p("Depreciation And Amortization","Depreciation","Reconciled Depreciation",src="cf")
    rect0= _p("Receivables","Net Receivables","Accounts Receivable")
    cogs0= _p("Cost Of Revenue","Cost Of Goods Sold","CostOfRevenue",src="inc")
    xsga0= _p("Selling General Administrative","SGA","General And Administrative Expense",src="inc")
    ppent0=_p("Net PPE","Property Plant Equipment","Net Property Plant And Equipment")
    che0 = _p("Cash And Cash Equivalents","Cash Cash Equivalents And Short Term Investments")
    ceq0 = _p("Stockholders Equity","Common Stock Equity","Total Stockholders Equity")
    csho0= _s(info.get("sharesOutstanding"))
    price= _s(info.get("currentPrice") or info.get("regularMarketPrice"))
    mc0  = _s(info.get("marketCap") or (_s(csho0) * _s(price)))
    mkt0 = mc0 if not math.isnan(mc0) else max(0, ceq0) if not math.isnan(ceq0) else math.nan

    at1  = _p("Total Assets","TotalAssets",col=1)
    lt1  = _p("Total Liabilities Net Minority Interest","Total Liab",col=1)
    sale1= _p("Total Revenue","Revenue",src="inc",col=1)
    ni1  = _p("Net Income","NetIncome",src="inc",col=1)
    cfo1 = _p("Operating Cash Flow",src="cf",col=1)
    dltt1= _p("Long Term Debt","LongTermDebt",col=1)
    dlc1 = _p("Current Debt","Short Long Term Debt",col=1)
    act1 = _p("Current Assets","Total Current Assets",col=1)
    lct1 = _p("Current Liabilities","Total Current Liabilities",col=1)
    cogs1= _p("Cost Of Revenue","Cost Of Goods Sold",src="inc",col=1)
    dp1  = _p("Depreciation And Amortization",src="cf",col=1)
    ppent1=_p("Net PPE","Property Plant Equipment",col=1)
    xsga1= _p("Selling General Administrative",src="inc",col=1)
    rect1= _p("Receivables","Net Receivables",col=1)

    cur = dict(at=at0,lt=lt0,act=act0,lct=lct0,re=re0,ebit=ebit0,
               sale=sale0,ni=ni0,ib=ib0,oancf=cfo0,dltt=dltt0,dlc=dlc0,
               dp=dp0,rect=rect0,cogs=cogs0,xsga=xsga0,ppent=ppent0,
               che=che0,ceq=ceq0,csho=csho0,mkt_cap=mkt0)
    prv = dict(at=at1,lt=lt1,sale=sale1,ni=ni1,oancf=cfo1,dltt=dltt1,dlc=dlc1,
               act=act1,lct=lct1,cogs=cogs1,dp=dp1,ppent=ppent1,xsga=xsga1,rect=rect1)
    return cur, prv


def _build_features(c, p):
    D = {}
    at  = _s(c["at"]);  lt  = _s(c["lt"])
    act = _s(c["act"]); lct = _s(c["lct"])
    wcap = act - lct if not (math.isnan(act) or math.isnan(lct)) else math.nan
    mkt = _s(c["mkt_cap"])
    ib  = _s(c.get("ib", c.get("ni", math.nan)))
    ni0 = _s(c["ni"]); ni1 = _s(p["ni"])
    dp0 = _s(c["dp"])

    # Z-Score
    D["z_x1"] = _d(wcap, at); D["z_x2"] = _d(_s(c["re"]), at)
    D["z_x3"] = _d(_s(c["ebit"]), at); D["z_x4"] = _d(mkt, lt)
    D["z_x5"] = _d(_s(c["sale"]), at)
    D["z_score"] = 1.2*D["z_x1"]+1.4*D["z_x2"]+3.3*D["z_x3"]+0.6*D["z_x4"]+D["z_x5"]

    # O-Score
    D["o_x1"] = math.log(at) if at > 0 else math.nan
    D["o_x2"] = _d(lt, at); D["o_x3"] = _d(wcap, at)
    D["o_x4"] = 1. if lt > at else 0.
    D["o_x5"] = _d(ib + dp0, lt); D["o_x6"] = _d(ib, at)
    D["o_x7"] = _d(ib + dp0, at)
    D["o_x8"] = 1. if (ni0 < 0 and ni1 < 0) else 0.
    dn9 = abs(ni0)+abs(ni1)
    D["o_x9"] = (ni0-ni1)/dn9 if dn9 > 0 else math.nan
    o_r = (-1.32-0.407*D["o_x1"]+6.03*D["o_x2"]-1.43*D["o_x3"]+0.076*D["o_x4"]
           -1.72*D["o_x5"]-2.37*D["o_x6"]-1.83*D["o_x7"]+0.285*D["o_x8"]-0.521*D["o_x9"])
    D["o_score_raw"] = o_r
    D["o_prob"] = 1/(1+math.exp(-max(-50,min(50,o_r)))) if math.isfinite(o_r) else math.nan

    # F-Score
    roa0=_d(ni0,at); roa1=_d(_s(p["ni"]),_s(p["at"]))
    cfo0=_s(c["oancf"]); cfota=_d(cfo0,at)
    lev0=_d(_s(c["dltt"]),at); lev1=_d(_s(p["dltt"]),_s(p["at"]))
    cr0=_d(act,lct); cr1=_d(_s(p["act"]),_s(p["lct"]))
    gm0=_d(_s(c["sale"])-_s(c["cogs"]),_s(c["sale"]))
    gm1=_d(_s(p["sale"])-_s(p.get("cogs",math.nan)),_s(p["sale"]))
    at0=_d(_s(c["sale"]),at); at1=_d(_s(p["sale"]),_s(p["at"]))
    D["f1"]=(1. if roa0>0 else 0.); D["f2"]=(1. if cfo0>0 else 0.)
    D["f3"]=(1. if (not math.isnan(roa1) and roa0>roa1) else 0.)
    D["f4"]=(1. if (not math.isnan(cfota) and cfota>roa0) else 0.)
    D["f5"]=(1. if (not math.isnan(lev1) and lev0<lev1) else 0.)
    D["f6"]=(1. if (not math.isnan(cr1) and cr0>cr1) else 0.)
    sh0=_s(c.get("csho",math.nan)); sh1=sh0  # approximate
    D["f7"]=(1. if (not math.isnan(sh1) and sh0<=sh1) else 0.)
    D["f8"]=(1. if (not math.isnan(gm1) and gm0>gm1) else 0.)
    D["f9"]=(1. if (not math.isnan(at1) and at0>at1) else 0.)
    D["f_score"]=sum(D[f"f{i}"] for i in range(1,10) if math.isfinite(D[f"f{i}"]))

    # M-Score
    sale0=_s(c["sale"]); sale1=_s(p["sale"])
    dsri=_d(_d(_s(c["rect"]),sale0),_d(_s(p["rect"]),sale1))
    gmi=_d(_d(sale1-_s(p.get("cogs",math.nan)),sale1),_d(sale0-_s(c["cogs"]),sale0))
    np0=at-_s(c["ppent"])-_s(c["che"]); np1=_s(p["at"])-_s(p["ppent"])
    aqi=_d(_d(np0,at),_d(np1,_s(p["at"])))
    sgi=_d(sale0,sale1)
    dp0_=_s(c["dp"]); dp1_=_s(p["dp"]); pp0=_s(c["ppent"]); pp1=_s(p["ppent"])
    dep0=_d(dp0_,dp0_+pp0) if (dp0_+pp0)!=0 else math.nan
    dep1=_d(dp1_,dp1_+pp1) if not math.isnan(pp1) and (dp1_+pp1)!=0 else math.nan
    depi=_d(dep1,dep0)
    sgai=_d(_d(_s(c["xsga"]),sale0),_d(_s(p["xsga"]),sale1))
    tata=_d(ib-cfo0,at)
    dbt0=(_s(c["dltt"])+_s(c["dlc"]))/at if at else math.nan
    dbt1=(_s(p["dltt"])+_s(p["dlc"]))/_s(p["at"]) if _s(p["at"]) else math.nan
    lvgi=_d(dbt0,dbt1)
    D.update(dict(dsri=dsri,gmi=gmi,aqi=aqi,sgi=sgi,depi=depi,
                  sgai=sgai,tata=tata,lvgi=lvgi))
    m=(-4.84+0.920*dsri+0.528*gmi+0.404*aqi+0.892*sgi+0.115*depi
       -0.172*sgai+4.679*tata-0.327*lvgi)
    D["m_score"]=m if math.isfinite(m) else math.nan

    # Extra ratios
    D.update(dict(roa=roa0, cfo_ta=cfota, cr=cr0, lev=lev0, gm=gm0, asset_turn=at0,
                  delta_roa=roa0-roa1 if not math.isnan(roa1) else math.nan,
                  delta_cr=cr0-cr1 if not math.isnan(cr1) else math.nan,
                  delta_lev=lev0-lev1 if not math.isnan(lev1) else math.nan,
                  delta_gm=gm0-gm1 if not math.isnan(gm1) else math.nan,
                  delta_sale=_d(sale0-sale1,sale1),
                  delta_at=_d(at-_s(p["at"]),_s(p["at"])),
                  debt_ratio=_d(lt,at),
                  equity_ratio=_d(_s(c.get("ceq",math.nan)),at),
                  log_at=math.log(at) if at>0 else math.nan,
                  log_sale=math.log(sale0) if sale0>0 else math.nan))
    return D


# ── Public API ────────────────────────────────────────────────────────────────
def model_available() -> bool:
    return (MODEL_DIR / "distress_model.json").exists()


def predict_ticker(ticker: str) -> dict:
    if not _load_artefacts():
        return {"error": "Model not trained. Run 03_train_model.py first.", "probability": None}
    try:
        cur, prv = _get_fundamentals(ticker)
    except Exception as e:
        return {"error": f"Failed to fetch data for {ticker}: {e}", "probability": None}
    if math.isnan(_s(cur.get("at", math.nan))):
        return {"error": f"No financial data available for {ticker}.", "probability": None}

    feat_dict = _build_features(cur, prv)
    feat_cols = _bundle["feature_cols"]

    row = []
    n_missing = 0
    for feat in feat_cols:
        val = feat_dict.get(feat, math.nan)
        if math.isnan(_s(val)):
            val = _feat_meds.get(feat, 0.0)
            n_missing += 1
        row.append(float(val))

    prob = _predict_raw(row)

    if prob < 0.15:
        risk_label, risk_color = "Low Risk",      "#3FCF8E"
    elif prob < 0.40:
        risk_label, risk_color = "Elevated Risk", "#F0A030"
    else:
        risk_label, risk_color = "High Risk",     "#E85555"

    return dict(
        probability  = prob,
        prediction   = int(prob >= _threshold),
        threshold    = _threshold,
        risk_label   = risk_label,
        risk_color   = risk_color,
        features     = feat_dict,
        missing_pct  = n_missing / len(feat_cols),
        error        = None,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ticker_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    print(f"\n  ML Distress Prediction — {ticker_arg}\n")
    if not model_available():
        print("  ✗  Run 03_train_model.py first.")
        sys.exit(1)
    r = predict_ticker(ticker_arg)
    if r.get("error"):
        print(f"  ✗  {r['error']}")
        sys.exit(1)
    print(f"  Distress Probability : {r['probability']:.1%}")
    print(f"  Risk Label           : {r['risk_label']}")
    print(f"  Prediction           : {'DISTRESSED' if r['prediction'] else 'HEALTHY'}")
    print(f"  Threshold            : {r['threshold']:.4f}")
    print(f"  Missing features     : {r['missing_pct']:.0%}")
    print()
    for k in ["z_score","o_prob","f_score","m_score","roa","lev","cr","delta_sale"]:
        v = r["features"].get(k, math.nan)
        print(f"  {k:<20} {_s(v, float('nan')):>10.4f}")
