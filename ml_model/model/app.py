import streamlit as st
import streamlit.components.v1 as components
try:
    import plotly.graph_objects as pgo
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
import yfinance as yf
import math
import sys
from pathlib import Path

# ── ML model integration (optional — graceful fallback if not trained yet) ───
_ML_DIR = Path(__file__).parent / "ml_model"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

try:
    from ml_model.predict_ticker import predict_ticker as _ml_predict, model_available as _ml_available  # type: ignore
    _ML_IMPORTED = True
except Exception:
    try:
        from predict_ticker import predict_ticker as _ml_predict, model_available as _ml_available  # type: ignore
        _ML_IMPORTED = True
    except Exception:
        # ── Inline fallback: pure JSON + NumPy prediction (no joblib/xgboost) ──
        import json as _json, numpy as _np
        _ML_IMPORTED = False
        _ML_MODEL_PATH = Path(__file__).parent / "ml_model" / "model" / "distress_model.json"
        _ML_MEDS_PATH  = Path(__file__).parent / "ml_model" / "data"  / "feature_medians.json"
        _ML_THR_PATH   = Path(__file__).parent / "ml_model" / "model" / "threshold.json"
        _ml_bundle_cache = {}

        def _ml_available():
            return _ML_MODEL_PATH.exists()

        def _load_ml_bundle():
            if not _ml_bundle_cache:
                with open(_ML_MODEL_PATH) as f:
                    _ml_bundle_cache["bundle"] = _json.load(f)
                _ml_bundle_cache["meds"] = {}
                if _ML_MEDS_PATH.exists():
                    with open(_ML_MEDS_PATH) as f:
                        _ml_bundle_cache["meds"] = _json.load(f)
                _ml_bundle_cache["thr"] = 0.5
                if _ML_THR_PATH.exists():
                    with open(_ML_THR_PATH) as f:
                        _ml_bundle_cache["thr"] = _json.load(f).get("threshold", 0.5)

        def _ml_logit_predict(feature_row):
            _load_ml_bundle()
            b = _ml_bundle_cache["bundle"]
            w   = _np.array(b["weights"],      dtype=_np.float64)
            bi  = float(b["bias"])
            mu  = _np.array(b["feature_mu"],   dtype=_np.float64)
            std = _np.array(b["feature_std"],  dtype=_np.float64)
            x   = (_np.array(feature_row, dtype=_np.float64) - mu) / std
            logit = float(_np.dot(x, w) + bi)
            logit = max(-50.0, min(50.0, logit))
            return 1.0 / (1.0 + math.exp(-logit))

        def _ml_predict(ticker: str) -> dict:
            if not _ml_available():
                return {"error": "Model not trained yet.", "probability": None}
            _load_ml_bundle()
            b    = _ml_bundle_cache["bundle"]
            meds = _ml_bundle_cache["meds"]
            thr  = _ml_bundle_cache["thr"]
            feat_cols = b["feature_cols"]

            # Build features using the same yfinance logic as the full script
            stock = yf.Ticker(ticker)
            info  = stock.info or {}

            def _gb(row, col=0):
                try:
                    bs = stock.balance_sheet
                    if bs is None or bs.empty or row not in bs.index: return None
                    return float(bs.loc[row].iloc[col])
                except: return None
            def _gi(row, col=0):
                try:
                    inc = stock.income_stmt
                    if inc is None or inc.empty or row not in inc.index: return None
                    return float(inc.loc[row].iloc[col])
                except: return None
            def _gc(row, col=0):
                try:
                    cf = stock.cashflow
                    if cf is None or cf.empty or row not in cf.index: return None
                    return float(cf.loc[row].iloc[col])
                except: return None
            def _p(*keys, src="bs", col=0):
                fn = {"bs":_gb,"inc":_gi,"cf":_gc}[src]
                for k in keys:
                    v = fn(k, col)
                    if v is not None and math.isfinite(v): return v
                return math.nan

            at0  = _p("Total Assets","TotalAssets")
            lt0  = _p("Total Liabilities Net Minority Interest","Total Liab")
            act0 = _p("Current Assets","Total Current Assets","CurrentAssets")
            lct0 = _p("Current Liabilities","Total Current Liabilities","CurrentLiabilities")
            re0  = _p("Retained Earnings","RetainedEarnings")
            ebit0= _p("EBIT","Ebit",src="inc")
            sale0= _p("Total Revenue","Revenue","Net Revenues",src="inc")
            ni0  = _p("Net Income","NetIncome",src="inc")
            ib0  = _p("Net Income Common Stockholders","Net Income","NetIncome",src="inc")
            cfo0 = _p("Operating Cash Flow",src="cf")
            dltt0= _p("Long Term Debt","LongTermDebt")
            dlc0 = _p("Current Debt","Short Long Term Debt")
            dp0  = _p("Depreciation And Amortization","Depreciation",src="cf")
            rect0= _p("Receivables","Net Receivables","Accounts Receivable")
            cogs0= _p("Cost Of Revenue","Cost Of Goods Sold",src="inc")
            xsga0= _p("Selling General Administrative",src="inc")
            ppent0=_p("Net PPE","Property Plant Equipment")
            che0 = _p("Cash And Cash Equivalents","Cash Cash Equivalents And Short Term Investments")
            ceq0 = _p("Stockholders Equity","Common Stock Equity","Total Stockholders Equity")
            mkt0 = float(info.get("marketCap") or max(0, ceq0) if not math.isnan(ceq0) else math.nan)

            at1  = _p("Total Assets",col=1); lt1=_p("Total Liabilities Net Minority Interest","Total Liab",col=1)
            sale1= _p("Total Revenue","Revenue",src="inc",col=1); ni1=_p("Net Income","NetIncome",src="inc",col=1)
            cfo1 = _p("Operating Cash Flow",src="cf",col=1); dltt1=_p("Long Term Debt",col=1); dlc1=_p("Current Debt",col=1)
            act1 = _p("Current Assets","Total Current Assets",col=1); lct1=_p("Current Liabilities","Total Current Liabilities",col=1)
            cogs1= _p("Cost Of Revenue","Cost Of Goods Sold",src="inc",col=1); dp1=_p("Depreciation And Amortization",src="cf",col=1)
            ppent1=_p("Net PPE","Property Plant Equipment",col=1); xsga1=_p("Selling General Administrative",src="inc",col=1)
            rect1= _p("Receivables","Net Receivables",col=1)

            def sd(a,b_): return a/b_ if (not math.isnan(a) and not math.isnan(b_) and b_!=0) else math.nan
            wcap=act0-lct0 if not(math.isnan(act0) or math.isnan(lct0)) else math.nan
            ib=ib0 if not math.isnan(ib0) else ni0

            feat = {}
            feat["z_x1"]=sd(wcap,at0); feat["z_x2"]=sd(re0,at0); feat["z_x3"]=sd(ebit0,at0)
            feat["z_x4"]=sd(mkt0,lt0); feat["z_x5"]=sd(sale0,at0)
            feat["z_score"]=1.2*feat["z_x1"]+1.4*feat["z_x2"]+3.3*feat["z_x3"]+0.6*feat["z_x4"]+feat["z_x5"]
            feat["o_x1"]=math.log(at0) if at0>0 else math.nan
            feat["o_x2"]=sd(lt0,at0); feat["o_x3"]=sd(wcap,at0)
            feat["o_x4"]=1. if lt0>at0 else 0.
            feat["o_x5"]=sd(ib+dp0,lt0); feat["o_x6"]=sd(ib,at0); feat["o_x7"]=sd(ib+dp0,at0)
            feat["o_x8"]=1. if (ni0<0 and ni1<0) else 0.
            dn9=abs(ni0)+abs(ni1)
            feat["o_x9"]=(ni0-ni1)/dn9 if (dn9>0 and not math.isnan(dn9)) else math.nan
            o_r=(-1.32-0.407*feat["o_x1"]+6.03*feat["o_x2"]-1.43*feat["o_x3"]+0.076*feat["o_x4"]
                 -1.72*feat["o_x5"]-2.37*feat["o_x6"]-1.83*feat["o_x7"]+0.285*feat["o_x8"]-0.521*feat["o_x9"])
            feat["o_score_raw"]=o_r; feat["o_prob"]=1/(1+math.exp(-max(-50,min(50,o_r)))) if math.isfinite(o_r) else math.nan
            roa0v=sd(ni0,at0); roa1v=sd(ni1,at1); cfo_ta=sd(cfo0,at0)
            lev0v=sd(dltt0,at0); lev1v=sd(dltt1,at1)
            cr0v=sd(act0,lct0); cr1v=sd(act1,lct1)
            gm0v=sd(sale0-cogs0,sale0); gm1v=sd(sale1-cogs1,sale1)
            at0v=sd(sale0,at0); at1v=sd(sale1,at1)
            feat["f1"]=1. if roa0v>0 else 0.; feat["f2"]=1. if cfo0>0 else 0.
            feat["f3"]=1. if (not math.isnan(roa1v) and roa0v>roa1v) else 0.
            feat["f4"]=1. if (not math.isnan(cfo_ta) and cfo_ta>roa0v) else 0.
            feat["f5"]=1. if (not math.isnan(lev1v) and lev0v<lev1v) else 0.
            feat["f6"]=1. if (not math.isnan(cr1v) and cr0v>cr1v) else 0.
            feat["f7"]=0.; feat["f8"]=1. if (not math.isnan(gm1v) and gm0v>gm1v) else 0.
            feat["f9"]=1. if (not math.isnan(at1v) and at0v>at1v) else 0.
            feat["f_score"]=sum(feat[f"f{i}"] for i in range(1,10))
            dsri=sd(sd(rect0,sale0),sd(rect1,sale1)); gmi=sd(sd(sale1-cogs1,sale1),sd(sale0-cogs0,sale0))
            np0v=at0-ppent0-che0; np1v=at1-ppent1
            aqi=sd(sd(np0v,at0),sd(np1v,at1)); sgi=sd(sale0,sale1)
            dep0v=sd(dp0,dp0+ppent0); dep1v=sd(dp1,dp1+ppent1); depi=sd(dep1v,dep0v)
            sgai=sd(sd(xsga0,sale0),sd(xsga1,sale1)); tata=sd(ib-cfo0,at0)
            dbt0v=(dltt0+dlc0)/at0 if at0 else math.nan; dbt1v=(dltt1+dlc1)/at1 if (not math.isnan(at1) and at1) else math.nan
            lvgi=sd(dbt0v,dbt1v)
            feat.update(dict(dsri=dsri,gmi=gmi,aqi=aqi,sgi=sgi,depi=depi,sgai=sgai,tata=tata,lvgi=lvgi))
            m=(-4.84+0.920*dsri+0.528*gmi+0.404*aqi+0.892*sgi+0.115*depi-0.172*sgai+4.679*tata-0.327*lvgi)
            feat["m_score"]=m if math.isfinite(m) else math.nan
            feat.update(dict(roa=roa0v,cfo_ta=cfo_ta,cr=cr0v,lev=lev0v,gm=gm0v,asset_turn=at0v,
                             delta_roa=roa0v-roa1v if not math.isnan(roa1v) else math.nan,
                             delta_cr=cr0v-cr1v if not math.isnan(cr1v) else math.nan,
                             delta_lev=lev0v-lev1v if not math.isnan(lev1v) else math.nan,
                             delta_gm=gm0v-gm1v if not math.isnan(gm1v) else math.nan,
                             delta_sale=sd(sale0-sale1,sale1), delta_at=sd(at0-at1,at1),
                             debt_ratio=sd(lt0,at0), equity_ratio=sd(ceq0,at0),
                             log_at=math.log(at0) if at0>0 else math.nan,
                             log_sale=math.log(sale0) if sale0>0 else math.nan))

            row = []
            n_miss = 0
            for fc in feat_cols:
                v = feat.get(fc, math.nan)
                if math.isnan(float(v) if v is not None else math.nan):
                    v = meds.get(fc, 0.0); n_miss += 1
                row.append(float(v))

            prob = _ml_logit_predict(row)
            if prob < 0.15:   rl, rc = "Low Risk",      "#3FCF8E"
            elif prob < 0.40: rl, rc = "Elevated Risk", "#F0A030"
            else:              rl, rc = "High Risk",     "#E85555"

            return dict(probability=prob, prediction=int(prob>=thr), threshold=thr,
                        risk_label=rl, risk_color=rc, features=feat,
                        missing_pct=n_miss/len(feat_cols), error=None)

st.set_page_config(
    page_title="DistressIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session-state navigation (no browser page reloads) ────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "history" not in st.session_state:
    st.session_state.history = []

current_page = st.session_state.page

def go(page):
    """Navigate to a page without any browser reload."""
    if page != st.session_state.page:
        st.session_state.history.append(st.session_state.page)
    st.session_state.page = page
    st.rerun()

def go_back():
    """Go to the previous page."""
    if st.session_state.history:
        st.session_state.page = st.session_state.history.pop()
    else:
        st.session_state.page = "home"
    st.rerun()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #EEEDF8 !important;
    color: #1E1B4B !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    background-color: #2A2869 !important;
    border-right: none !important;
    min-width: 220px !important;
    max-width: 220px !important;
    width: 220px !important;
    transform: none !important;
}
[data-testid="stSidebarContent"] { padding: 1.8rem 0.8rem 1rem !important; }
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarUserContent"] { padding: 0 !important; }

.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; padding: 0 0.4rem; }
.brand-icon { width: 38px; height: 38px; background: rgba(255,255,255,0.15); border-radius: 9px;
              display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.brand-icon-inner { width: 16px; height: 16px; background: #FFFFFF; border-radius: 3px; opacity: 0.85; }
.brand-name-1 { font-size: 17px; font-weight: 800; color: #FFFFFF; line-height: 1.2; letter-spacing:-0.3px; }
.brand-name-2 { font-size: 17px; font-weight: 800; color: rgba(255,255,255,0.55); }

.nav-section-label {
    font-size: 10px; color: rgba(255,255,255,0.35); letter-spacing: 1.5px; font-weight: 600;
    text-transform: uppercase; margin: 0.7rem 0 0.3rem; padding: 0 0.5rem;
}
.nav-divider { height: 0.5px; background: rgba(255,255,255,0.08); margin: 0.5rem 0; }

/* ── Sidebar nav buttons ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: transparent !important; color: rgba(255,255,255,0.55) !important;
    border: none !important; text-align: left !important; justify-content: flex-start !important;
    padding: 7px 12px !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-size: 13px !important;
    font-weight: 500 !important; height: auto !important; min-height: 32px !important;
    letter-spacing: 0 !important; line-height: 1.4 !important; box-shadow: none !important;
    margin-bottom: 1px !important; width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.08) !important; color: rgba(255,255,255,0.9) !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:focus {
    box-shadow: none !important; outline: none !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #FFFFFF !important; border: 1.5px solid #E0DDF5 !important;
    border-radius: 10px !important; color: #1E1B4B !important;
    font-family: 'DM Mono', monospace !important; font-size: 15px !important;
    font-weight: 500 !important; letter-spacing: 2px !important; padding: 0 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #5652D8 !important; box-shadow: 0 0 0 3px rgba(86,82,216,0.1) !important;
}
[data-testid="stTextInput"] label { color: #6B7280 !important; font-size: 12px !important; }

/* ── Main buttons ────────────────────────────────────────────────────────── */
.main [data-testid="stButton"] > button,
[data-testid="stMain"] [data-testid="stButton"] > button {
    background: #5652D8 !important; color: #FFFFFF !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important; font-size: 14px !important;
    font-weight: 600 !important; height: 44px !important;
    letter-spacing: 0.2px !important; width: 100% !important;
    box-shadow: 0 2px 8px rgba(86,82,216,0.25) !important;
}
.main [data-testid="stButton"] > button:hover,
[data-testid="stMain"] [data-testid="stButton"] > button:hover {
    background: #4440C9 !important; color: #FFFFFF !important;
}
.back-btn [data-testid="stButton"] > button {
    background: #FFFFFF !important; color: #6B7280 !important;
    border: 1.5px solid #E0DDF5 !important; font-size: 13px !important;
    height: 36px !important; width: auto !important;
    padding: 0 14px !important; box-shadow: none !important;
}
.back-btn [data-testid="stButton"] > button:hover {
    background: #F4F3FF !important; color: #1E1B4B !important;
}
.card-open-btn [data-testid="stButton"] > button {
    background: #5652D8 !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-size: 13px !important; height: 40px !important;
    box-shadow: 0 1px 4px rgba(86,82,216,0.2) !important;
}
.card-open-btn [data-testid="stButton"] > button:hover { background: #4440C9 !important; }

/* ── Model cards ─────────────────────────────────────────────────────────── */
.model-card {
    background: #FFFFFF; border: 1.5px solid #E8E5F8; border-radius: 14px;
    padding: 1.2rem 1.4rem 0.8rem; color: #1E1B4B !important;
    box-shadow: 0 2px 8px rgba(86,82,216,0.06);
}
.model-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.model-card-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.model-card-name { font-size: 15px; font-weight: 700; color: #1E1B4B; }
.model-card-badge { font-size: 9px; font-weight: 700; letter-spacing: 0.5px;
                    padding: 3px 8px; border-radius: 5px; font-family: 'DM Mono', monospace; }
.model-card-desc { font-size: 12px; color: #6B7280; line-height: 1.5; margin-top: 4px; margin-bottom: 0.8rem; }

/* ── Result panels ───────────────────────────────────────────────────────── */
.result-card { background: #FFFFFF; border: 1.5px solid #E8E5F8; border-radius: 14px;
               padding: 1.2rem 1.5rem; box-shadow: 0 2px 8px rgba(86,82,216,0.06); margin-bottom: 1rem; }
.company-card { background: #FFFFFF; border: 1.5px solid #E8E5F8; border-radius: 14px;
                padding: 1.2rem 1.5rem; display: flex; align-items: center;
                justify-content: space-between; flex-wrap: wrap; gap: 12px;
                margin: 1rem 0 1.5rem; box-shadow: 0 2px 8px rgba(86,82,216,0.06); }
.fin-row { display:flex; justify-content:space-between; align-items:center;
           background:#F8F7FF; border-radius:8px; padding:10px 14px;
           border:1.5px solid #EAE8F8; margin-bottom:8px; }
.fin-row-label { font-size:12px; color:#6B7280; }
.fin-row-val { font-family:'DM Mono',monospace; font-size:13px; font-weight:500; color:#1E1B4B; }
.ratio-card { background:#F8F7FF; border:1.5px solid #EAE8F8; border-radius:10px;
              padding:0.9rem 0.8rem; text-align:center; }
.ratio-name { font-size:11px; font-weight:700; color:#5652D8; margin-bottom:3px; }
.ratio-formula { font-size:9px; color:#9CA3AF; margin-bottom:8px; line-height:1.3; }
.ratio-val { font-family:'DM Mono',monospace; font-size:17px; font-weight:600; color:#1E1B4B; }
.ratio-wt { font-size:9px; color:#9CA3AF; margin-top:3px; font-family:'DM Mono',monospace; }
.ticker-badge { background:rgba(86,82,216,0.08); border:1.5px solid rgba(86,82,216,0.2);
                border-radius:6px; padding:4px 10px; font-family:'DM Mono',monospace;
                font-size:13px; font-weight:600; color:#5652D8; letter-spacing:1px; }
.interp-box { background:#F4F3FF; border-left:3px solid #5652D8; border-radius:0 10px 10px 0;
              padding:1rem 1.2rem; font-size:13px; color:#374151; line-height:1.7; margin-top:1.2rem; }
.alert-error { background:rgba(239,68,68,0.06); border:1.5px solid rgba(239,68,68,0.2);
               border-radius:10px; padding:1rem 1.2rem; color:#DC2626; font-size:13px; margin-top:1rem; }
.alert-warn { background:rgba(245,158,11,0.06); border:1.5px solid rgba(245,158,11,0.2);
              border-radius:10px; padding:1rem 1.2rem; color:#D97706; font-size:13px; margin-top:1rem; }

/* ── Section headers ─────────────────────────────────────────────────────── */
.sec-hdr { display: flex; align-items: center; gap: 10px; margin: 1.5rem 0 0.8rem; }
.sec-lbl { font-size: 10px; font-weight: 700; color: #9CA3AF;
           letter-spacing: 2px; text-transform: uppercase; white-space: nowrap; }
.sec-line { flex: 1; height: 1.5px; background: #EAE8F8; }

/* ── Quick start ─────────────────────────────────────────────────────────── */
.qs-box { background: #FFFFFF; border: 1.5px solid #E8E5F8; border-radius: 14px;
          padding: 1.3rem 1.6rem; box-shadow: 0 2px 8px rgba(86,82,216,0.06); }
.qs-step { display: flex; align-items: flex-start; gap: 12px;
           margin-bottom: 10px; font-size: 13px; color: #6B7280; line-height: 1.5; }
.qs-step:last-child { margin-bottom: 0; }
.qs-num { font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500;
          color: #5652D8; background: rgba(86,82,216,0.1); border-radius: 50%;
          width: 22px; height: 22px; display: flex; align-items: center;
          justify-content: center; flex-shrink: 0; margin-top: 1px; }

/* ── Piotroski criteria ──────────────────────────────────────────────────── */
.criteria-item { display:flex; align-items:center; gap:10px; padding:8px 12px;
                 background:#F8F7FF; border-radius:8px; border:1.5px solid #EAE8F8; margin-bottom:6px; }
.criteria-pass { color:#16A34A; font-weight:700; font-size:14px; }
.criteria-fail { color:#DC2626; font-weight:700; font-size:14px; }
.criteria-label { font-size:13px; color:#1E1B4B; font-weight:500; flex:1; }
.criteria-desc { font-size:11px; color:#9CA3AF; }
.criteria-val { font-family:'DM Mono',monospace; font-size:11px; color:#6B7280; }

/* ── Animations ──────────────────────────────────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fu1 { animation: fadeUp 0.4s ease both; }
.fu2 { animation: fadeUp 0.4s 0.08s ease both; }
.fu3 { animation: fadeUp 0.4s 0.16s ease both; }
.fu4 { animation: fadeUp 0.4s 0.24s ease both; }
.fu5 { animation: fadeUp 0.4s 0.32s ease both; }
.fu6 { animation: fadeUp 0.4s 0.40s ease both; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(val):
    if val is None: return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12: return f"{sign}${abs_val/1e12:.2f}T"
    if abs_val >= 1e9:  return f"{sign}${abs_val/1e9:.2f}B"
    if abs_val >= 1e6:  return f"{sign}${abs_val/1e6:.2f}M"
    return f"{sign}${abs_val:,.0f}"

def get_val(df, *keys):
    for k in keys:
        if k in df.index:
            try: return float(df.loc[k].iloc[0])
            except: pass
    return None


# ── Models ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def compute_zscore(ticker_str):
    stock = yf.Ticker(ticker_str)
    info  = stock.info
    bs    = stock.balance_sheet
    inc   = stock.income_stmt

    mc = info.get("marketCap", 0) or 0
    wc = get_val(bs, "Working Capital", "WorkingCapital")
    ta = get_val(bs, "Total Assets", "TotalAssets")
    re = get_val(bs, "Retained Earnings", "RetainedEarnings")
    tl = get_val(bs, "Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities")
    eb = get_val(inc, "EBIT", "Ebit", "Operating Income", "OperatingIncome")
    rv = get_val(inc, "Total Revenue", "TotalRevenue", "Revenue")

    if wc is None:
        ca = get_val(bs, "Current Assets", "CurrentAssets")
        cl = get_val(bs, "Current Liabilities", "CurrentLiabilities")
        if ca and cl: wc = ca - cl

    name    = info.get("longName", ticker_str)
    sector  = info.get("sector", "N/A")
    country = info.get("country", "N/A")
    website = info.get("website", "") or ""
    domain  = website.replace("https://","").replace("http://","").split("/")[0]
    logo    = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""

    vals = [wc, ta, re, tl, eb, rv]
    if all(v is not None for v in vals) and ta and ta != 0:
        x1 = wc / ta
        x2 = re / ta
        x3 = eb / ta
        x4 = (mc / tl) if (mc and tl and tl != 0) else 0
        x5 = rv / ta
        z  = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        x4_warn = (not mc or mc == 0) or x4 > 10
        return dict(ok=True, name=name, sector=sector, country=country,
                    logo=logo, mc=mc, z=z,
                    x1=x1, x2=x2, x3=x3, x4=x4, x5=x5,
                    wc=wc, ta=ta, re=re, tl=tl, eb=eb, rv=rv,
                    x4_warn=x4_warn)
    return dict(ok=False, name=name, sector=sector, country=country, logo=logo, mc=mc)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_revenue_ebit_trend(ticker_str):
    """Fetch up to 4 years of Revenue and EBIT for the trend chart."""
    try:
        inc = yf.Ticker(ticker_str).income_stmt
        def _row(df, *keys):
            for k in keys:
                if k in df.index:
                    return df.loc[k]
            return None
        rev_s  = _row(inc, "Total Revenue", "TotalRevenue", "Revenue")
        ebit_s = _row(inc, "EBIT", "Ebit", "Operating Income", "OperatingIncome")
        if rev_s is None:
            return None
        cols   = sorted(inc.columns, reverse=False)   # oldest → newest
        years  = [str(c.year) for c in cols]
        def _safe(series, col):
            try:
                v = float(series[col])
                return None if math.isnan(v) else v / 1e9
            except Exception:
                return None
        revenue = [_safe(rev_s,  c) for c in cols]
        ebit    = [_safe(ebit_s, c) if ebit_s is not None else None for c in cols]
        if all(v is None for v in revenue):
            return None
        return dict(years=years, revenue=revenue, ebit=ebit)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def compute_oscore(ticker_str):
    stock = yf.Ticker(ticker_str)
    info  = stock.info
    bs    = stock.balance_sheet
    inc   = stock.income_stmt
    cf    = stock.cash_flow

    name    = info.get("longName", ticker_str)
    sector  = info.get("sector", "N/A")
    country = info.get("country", "N/A")
    mc      = info.get("marketCap", 0) or 0
    website = info.get("website", "") or ""
    domain  = website.replace("https://","").replace("http://","").split("/")[0]
    logo    = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""

    ta  = get_val(bs, "Total Assets", "TotalAssets")
    tl  = get_val(bs, "Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities")
    wc  = get_val(bs, "Working Capital", "WorkingCapital")
    ca  = get_val(bs, "Current Assets", "CurrentAssets")
    cl  = get_val(bs, "Current Liabilities", "CurrentLiabilities")
    ni  = get_val(inc, "Net Income", "NetIncome")
    cfo = get_val(cf,  "Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Free Cash Flow")

    if wc is None and ca and cl: wc = ca - cl

    ni_prev = None
    if inc is not None and len(inc.columns) > 1:
        for k in ["Net Income", "NetIncome"]:
            if k in inc.index:
                try:
                    ni_prev = float(inc.loc[k].iloc[1])
                    break
                except: pass

    required = [ta, tl, wc, ca, cl, ni, cfo]
    if not all(v is not None for v in required) or ta == 0 or tl == 0:
        return dict(ok=False, name=name, sector=sector, country=country, logo=logo, mc=mc)

    x1 = math.log(abs(ta)) if ta > 0 else 0
    x2 = tl / ta
    x3 = wc / ta
    x4 = cl / ca if ca != 0 else 0
    x5 = 1 if tl > ta else 0
    x6 = ni / ta
    x7 = cfo / tl if tl != 0 else 0
    x8 = 1 if (ni_prev is not None and ni < 0 and ni_prev < 0) else 0
    if ni_prev is not None and (abs(ni) + abs(ni_prev)) != 0:
        x9 = (ni - ni_prev) / (abs(ni) + abs(ni_prev))
    else:
        x9 = 0

    o = (-1.32 - 0.407*x1 + 6.03*x2 - 1.43*x3 + 0.076*x4
         - 1.72*x5 - 2.37*x6 - 1.83*x7 + 0.285*x8 - 0.521*x9)
    prob = 1 / (1 + math.exp(-o))

    return dict(
        ok=True, name=name, sector=sector, country=country, logo=logo, mc=mc,
        o=o, prob=prob,
        x1=x1, x2=x2, x3=x3, x4=x4, x5=x5,
        x6=x6, x7=x7, x8=x8, x9=x9,
        ta=ta, tl=tl, wc=wc, ca=ca, cl=cl, ni=ni, cfo=cfo, ni_prev=ni_prev
    )




@st.cache_data(ttl=3600, show_spinner=False)
def compute_oscore_trend(ticker_str):
    """Return multi-year TL/TA (leverage) and NI/TA (profitability) for O-Score trend chart."""
    try:
        stock = yf.Ticker(ticker_str)
        bs  = stock.balance_sheet
        inc = stock.income_stmt
        if bs is None or inc is None:
            return None
        rows = []
        for col in bs.columns:
            try:
                year = str(col.year) if hasattr(col, 'year') else str(col)[:4]
                ta_v = None
                for k in ["Total Assets", "TotalAssets"]:
                    if k in bs.index:
                        try: ta_v = float(bs.loc[k, col]); break
                        except: pass
                tl_v = None
                for k in ["Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities"]:
                    if k in bs.index:
                        try: tl_v = float(bs.loc[k, col]); break
                        except: pass
                ni_v = None
                if col in inc.columns:
                    for k in ["Net Income", "NetIncome"]:
                        if k in inc.index:
                            try: ni_v = float(inc.loc[k, col]); break
                            except: pass
                if ta_v and ta_v != 0 and tl_v is not None and ni_v is not None:
                    rows.append({"year": year, "leverage": tl_v / ta_v, "profitability": ni_v / ta_v})
            except:
                continue
        if not rows:
            return None
        rows.sort(key=lambda r: r["year"])
        return {
            "years": [r["year"] for r in rows],
            "leverage": [r["leverage"] for r in rows],
            "profitability": [r["profitability"] for r in rows],
        }
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def compute_zscore2(ticker_str):
    """Zmijewski (1984) probit model: X = -4.336 - 4.513*ROA + 5.679*LEV + 0.004*LIQ
    P(distress) = standard normal CDF of X.
    """
    stock = yf.Ticker(ticker_str)
    info  = stock.info
    bs    = stock.balance_sheet
    inc   = stock.income_stmt

    name    = info.get("longName", ticker_str)
    sector  = info.get("sector", "N/A")
    country = info.get("country", "N/A")
    mc      = info.get("marketCap", 0) or 0
    website = info.get("website", "") or ""
    domain  = website.replace("https://","").replace("http://","").split("/")[0]
    logo    = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""

    ta  = get_val(bs, "Total Assets", "TotalAssets")
    tl  = get_val(bs, "Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities")
    ca  = get_val(bs, "Current Assets", "CurrentAssets")
    cl  = get_val(bs, "Current Liabilities", "CurrentLiabilities")
    ni  = get_val(inc, "Net Income", "NetIncome")

    if any(v is None for v in [ta, tl, ca, cl, ni]) or ta == 0 or cl == 0:
        return dict(ok=False, name=name, sector=sector, country=country, logo=logo, mc=mc)

    roa = ni / ta
    lev = tl / ta
    liq = ca / cl
    x   = -4.336 - 4.513 * roa + 5.679 * lev + 0.004 * liq
    prob = 0.5 * math.erfc(-x / math.sqrt(2))

    return dict(ok=True, name=name, sector=sector, country=country, logo=logo, mc=mc,
                x=x, prob=prob, roa=roa, lev=lev, liq=liq, ta=ta, tl=tl, ca=ca, cl=cl, ni=ni)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_zscore2_trend(ticker_str):
    """Return multi-year ROA (NI/TA) and Leverage (TL/TA) for Zmijewski trend chart."""
    try:
        stock = yf.Ticker(ticker_str)
        bs  = stock.balance_sheet
        inc = stock.income_stmt
        if bs is None or inc is None:
            return None
        rows = []
        for col in bs.columns:
            try:
                year = str(col.year) if hasattr(col, 'year') else str(col)[:4]
                ta_v = None
                for k in ["Total Assets", "TotalAssets"]:
                    if k in bs.index:
                        try: ta_v = float(bs.loc[k, col]); break
                        except: pass
                tl_v = None
                for k in ["Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities"]:
                    if k in bs.index:
                        try: tl_v = float(bs.loc[k, col]); break
                        except: pass
                ni_v = None
                if col in inc.columns:
                    for k in ["Net Income", "NetIncome"]:
                        if k in inc.index:
                            try: ni_v = float(inc.loc[k, col]); break
                            except: pass
                if ta_v and ta_v != 0 and tl_v is not None and ni_v is not None:
                    rows.append({
                        "year": year,
                        "roa": ni_v / ta_v,
                        "leverage": tl_v / ta_v,
                    })
            except:
                continue
        if not rows:
            return None
        rows.sort(key=lambda r: r["year"])
        return {
            "years":    [r["year"]    for r in rows],
            "roa":      [r["roa"]     for r in rows],
            "leverage": [r["leverage"] for r in rows],
        }
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def compute_fscore(ticker_str):
    stock = yf.Ticker(ticker_str)
    info  = stock.info
    bs    = stock.balance_sheet
    inc   = stock.income_stmt
    cf    = stock.cash_flow

    name    = info.get("longName", ticker_str)
    sector  = info.get("sector", "N/A")
    country = info.get("country", "N/A")
    mc      = info.get("marketCap", 0) or 0
    website = info.get("website", "") or ""
    domain  = website.replace("https://","").replace("http://","").split("/")[0]
    logo    = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""

    def _g(df, *keys, yr=0):
        for k in keys:
            if df is not None and k in df.index:
                try:
                    if len(df.columns) > yr:
                        v = float(df.loc[k].iloc[yr])
                        if not math.isnan(v):
                            return v
                except: pass
        return None

    ta0  = _g(bs,  "Total Assets",  "TotalAssets")
    ta1  = _g(bs,  "Total Assets",  "TotalAssets",                                        yr=1)
    ni0  = _g(inc, "Net Income",    "NetIncome")
    ni1  = _g(inc, "Net Income",    "NetIncome",                                           yr=1)
    cfo0 = _g(cf,  "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    ltd0 = _g(bs,  "Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation")
    ltd1 = _g(bs,  "Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation", yr=1)
    ca0  = _g(bs,  "Current Assets",      "CurrentAssets")
    ca1  = _g(bs,  "Current Assets",      "CurrentAssets",       yr=1)
    cl0  = _g(bs,  "Current Liabilities", "CurrentLiabilities")
    cl1  = _g(bs,  "Current Liabilities", "CurrentLiabilities",  yr=1)
    gp0  = _g(inc, "Gross Profit",  "GrossProfit")
    gp1  = _g(inc, "Gross Profit",  "GrossProfit",               yr=1)
    rv0  = _g(inc, "Total Revenue", "TotalRevenue", "Revenue")
    rv1  = _g(inc, "Total Revenue", "TotalRevenue", "Revenue",   yr=1)
    sh0  = _g(bs,  "Ordinary Shares Number", "Share Issued")
    sh1  = _g(bs,  "Ordinary Shares Number", "Share Issued",     yr=1)

    if ta0 is None or ni0 is None or cfo0 is None:
        return dict(ok=False, name=name, sector=sector, country=country, logo=logo, mc=mc)

    roa0   = ni0 / ta0 if ta0 else 0
    roa1   = (ni1 / ta1) if (ni1 is not None and ta1) else None
    cfo_ta = cfo0 / ta0 if ta0 else 0

    def _pct(a, b): return f"{a:.3f} vs {b:.3f}"

    flags = {}
    # ── A: Profitability ──────────────────────────────────────────────────────
    flags["F1"] = ("ROA > 0",       "Net income / assets positive",       roa0 > 0,         f"{roa0:.3f}")
    flags["F2"] = ("CFO > 0",       "Operating cash flow positive",       cfo0 > 0,         fmt(cfo0))
    flags["F3"] = ("ΔROA > 0",      "ROA improved year-over-year",
                   (roa0 > roa1) if roa1 is not None else False,
                   (_pct(roa0, roa1) if roa1 is not None else "N/A"))
    flags["F4"] = ("Accruals",      "CFO / TA > ROA  (earnings quality)", cfo_ta > roa0,    _pct(cfo_ta, roa0))
    # ── B: Leverage & Liquidity ───────────────────────────────────────────────
    if ltd0 is not None and ltd1 is not None and ta0 and ta1:
        lev0, lev1 = ltd0/ta0, ltd1/ta1
        flags["F5"] = ("ΔLeverage",    "Long-term leverage decreased",   lev0 < lev1,       _pct(lev0, lev1))
    else:
        flags["F5"] = ("ΔLeverage",    "Long-term leverage decreased",   False,             "N/A")
    if ca0 and cl0 and ca1 and cl1:
        cr0, cr1 = ca0/cl0, ca1/cl1
        flags["F6"] = ("ΔCurr Ratio",  "Current ratio improved",         cr0 > cr1,         f"{cr0:.2f} vs {cr1:.2f}")
    else:
        flags["F6"] = ("ΔCurr Ratio",  "Current ratio improved",         False,             "N/A")
    if sh0 is not None and sh1 is not None:
        flags["F7"] = ("No Dilution",  "Shares outstanding not increased", sh0 <= sh1,      f"{sh0/1e6:.1f}M vs {sh1/1e6:.1f}M")
    else:
        flags["F7"] = ("No Dilution",  "Shares outstanding not increased", False,           "N/A")
    # ── C: Operating Efficiency ───────────────────────────────────────────────
    if gp0 and gp1 and rv0 and rv1:
        gm0, gm1 = gp0/rv0, gp1/rv1
        flags["F8"] = ("ΔGross Margin","Gross margin improved",           gm0 > gm1,        _pct(gm0, gm1))
    else:
        flags["F8"] = ("ΔGross Margin","Gross margin improved",           False,            "N/A")
    if rv0 and rv1 and ta0 and ta1:
        at0, at1 = rv0/ta0, rv1/ta1
        flags["F9"] = ("ΔAsset Turn.", "Asset turnover improved",         at0 > at1,        _pct(at0, at1))
    else:
        flags["F9"] = ("ΔAsset Turn.", "Asset turnover improved",         False,            "N/A")

    score = sum(1 for v in flags.values() if v[2])
    return dict(
        ok=True, name=name, sector=sector, country=country, logo=logo, mc=mc,
        score=score, flags=flags,
        ta0=ta0, ni0=ni0, cfo0=cfo0, ltd0=ltd0, ca0=ca0, cl0=cl0,
        gp0=gp0, rv0=rv0, roa0=roa0
    )


@st.cache_data(ttl=3600, show_spinner=False)
def compute_mscore(ticker_str):
    stock = yf.Ticker(ticker_str)
    info  = stock.info
    bs    = stock.balance_sheet
    inc   = stock.income_stmt
    cf    = stock.cash_flow

    name    = info.get("longName", ticker_str)
    sector  = info.get("sector", "N/A")
    country = info.get("country", "N/A")
    mc      = info.get("marketCap", 0) or 0
    website = info.get("website", "") or ""
    domain  = website.replace("https://","").replace("http://","").split("/")[0]
    logo    = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""

    def _g(df, *keys, yr=0):
        for k in keys:
            if df is not None and k in df.index:
                try:
                    if len(df.columns) > yr:
                        v = float(df.loc[k].iloc[yr])
                        if not math.isnan(v):
                            return v
                except: pass
        return None

    rec0  = _g(bs,  "Net Receivables",   "Accounts Receivable",    "Receivables")
    rec1  = _g(bs,  "Net Receivables",   "Accounts Receivable",    "Receivables",                         yr=1)
    rv0   = _g(inc, "Total Revenue",     "TotalRevenue",            "Revenue")
    rv1   = _g(inc, "Total Revenue",     "TotalRevenue",            "Revenue",                             yr=1)
    cogs0 = _g(inc, "Cost Of Revenue",   "CostOfRevenue",           "Cost Of Goods Sold")
    cogs1 = _g(inc, "Cost Of Revenue",   "CostOfRevenue",           "Cost Of Goods Sold",                  yr=1)
    ca0   = _g(bs,  "Current Assets",    "CurrentAssets")
    ca1   = _g(bs,  "Current Assets",    "CurrentAssets",                                                  yr=1)
    ppe0  = _g(bs,  "Net PPE",           "Property Plant Equipment Net", "Net Property Plant And Equipment")
    ppe1  = _g(bs,  "Net PPE",           "Property Plant Equipment Net", "Net Property Plant And Equipment", yr=1)
    ta0   = _g(bs,  "Total Assets",      "TotalAssets")
    ta1   = _g(bs,  "Total Assets",      "TotalAssets",                                                    yr=1)
    dep0  = _g(cf,  "Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion")
    dep1  = _g(cf,  "Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion", yr=1)
    sga0  = _g(inc, "Selling General And Administrative", "SGAExpense", "Selling And Marketing Expense")
    sga1  = _g(inc, "Selling General And Administrative", "SGAExpense", "Selling And Marketing Expense",   yr=1)
    ni0   = _g(inc, "Net Income",        "NetIncome")
    cfo0  = _g(cf,  "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    ltd0  = _g(bs,  "Long Term Debt",    "LongTermDebt",            "Long Term Debt And Capital Lease Obligation")
    ltd1  = _g(bs,  "Long Term Debt",    "LongTermDebt",            "Long Term Debt And Capital Lease Obligation", yr=1)
    cl0   = _g(bs,  "Current Liabilities", "CurrentLiabilities")
    cl1   = _g(bs,  "Current Liabilities", "CurrentLiabilities",                                           yr=1)

    if any(v is None for v in [rv0, rv1, ta0, ta1]):
        return dict(ok=False, name=name, sector=sector, country=country, logo=logo, mc=mc)

    idx = {}  # key → (full_name, value, elevated_flag)

    if rec0 and rec1 and rv0 and rv1:
        v = (rec0/rv0) / (rec1/rv1)
        idx["DSRI"] = ("Days Sales Receivable Index",     v,    v > 1.031)
    else:
        idx["DSRI"] = ("Days Sales Receivable Index",     None, False)

    if cogs0 and cogs1 and rv0 and rv1:
        gm0 = (rv0 - cogs0) / rv0;  gm1 = (rv1 - cogs1) / rv1
        v   = (gm1 / gm0) if gm0 != 0 else None
        idx["GMI"]  = ("Gross Margin Index",              v,    (v > 1.014) if v else False)
    else:
        idx["GMI"]  = ("Gross Margin Index",              None, False)

    if ca0 and ppe0 and ta0 and ca1 and ppe1 and ta1:
        aq0 = 1 - (ca0+ppe0)/ta0;  aq1 = 1 - (ca1+ppe1)/ta1
        v   = (aq0 / aq1) if aq1 != 0 else None
        idx["AQI"]  = ("Asset Quality Index",             v,    (v > 1.039) if v else False)
    else:
        idx["AQI"]  = ("Asset Quality Index",             None, False)

    v = rv0/rv1 if rv1 != 0 else None
    idx["SGI"]  = ("Sales Growth Index",              v,    (v > 1.134) if v else False)

    if dep0 and ppe0 and dep1 and ppe1:
        d0 = dep0/(dep0+ppe0) if (dep0+ppe0) != 0 else None
        d1 = dep1/(dep1+ppe1) if (dep1+ppe1) != 0 else None
        v  = (d1/d0) if (d0 and d0 != 0) else None
        idx["DEPI"] = ("Depreciation Index",              v,    (v > 1.001) if v else False)
    else:
        idx["DEPI"] = ("Depreciation Index",              None, False)

    if sga0 and sga1 and rv0 and rv1:
        v = (sga0/rv0) / (sga1/rv1)
        idx["SGAI"] = ("SG&A Expense Index",              v,    v > 1.054)
    else:
        idx["SGAI"] = ("SG&A Expense Index",              None, False)

    if ni0 is not None and cfo0 is not None and ta0:
        v = (ni0 - cfo0) / ta0
        idx["TATA"] = ("Total Accruals / Total Assets",   v,    v > 0.018)
    else:
        idx["TATA"] = ("Total Accruals / Total Assets",   None, False)

    if ltd0 is not None and cl0 and ta0 and ltd1 is not None and cl1 and ta1:
        lev0 = (ltd0+cl0)/ta0;  lev1 = (ltd1+cl1)/ta1
        v    = (lev0/lev1) if lev1 != 0 else None
        idx["LVGI"] = ("Leverage Index",                  v,    (v > 1.0) if v else False)
    else:
        idx["LVGI"] = ("Leverage Index",                  None, False)

    def _v(k): return idx[k][1] if idx[k][1] is not None else 1.0

    m = (-4.84
         + 0.920 * _v("DSRI")
         + 0.528 * _v("GMI")
         + 0.404 * _v("AQI")
         + 0.892 * _v("SGI")
         + 0.115 * _v("DEPI")
         - 0.172 * _v("SGAI")
         + 4.679 * _v("TATA")
         - 0.327 * _v("LVGI"))

    return dict(
        ok=True, name=name, sector=sector, country=country, logo=logo, mc=mc,
        m=m, idx=idx, rv0=rv0, ta0=ta0, ni0=ni0, cfo0=cfo0
    )


# ── Zone helpers ──────────────────────────────────────────────────────────────
def zone_info(z):
    if z > 2.99:
        return "Safe Zone",     "Z > 2.99",        "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"
    if z > 1.81:
        return "Grey Zone",     "1.81 < Z < 2.99", "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
    return     "Distress Zone", "Z < 1.81",         "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"

def gauge_pct(z):
    if z > 2.99: return min(94, 50 + (z - 2.99) * 8)
    if z > 1.81: return 35 + (z - 1.81) * 12
    return max(5, z * 10)

def o_zone(prob):
    p = prob * 100
    if p < 20:
        return "Low Risk",    f"{p:.1f}%", "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"
    if p < 50:
        return "Medium Risk", f"{p:.1f}%", "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
    return     "High Risk",   f"{p:.1f}%", "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"


# ── Render components ─────────────────────────────────────────────────────────
def render_zscore_panel(z, d):
    zl, zs, zc, zbg, zbd = zone_info(z)
    gp = gauge_pct(z)
    warn_html = ""
    if d.get('x4_warn'):
        if not d['mc'] or d['mc'] == 0:
            warn_html = '<div style="margin-top:12px;font-size:11px;color:#D97706;">&#9888; Market cap unavailable — X4 set to 0, score may be understated.</div>'
        elif d['x4'] > 10:
            warn_html = '<div style="margin-top:12px;font-size:11px;color:#D97706;">&#9888; X4 is very large — may inflate Z-score for high-cap firms.</div>'
    html = f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Inter',sans-serif;}}
.panel{{background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:16px;padding:1.4rem 1.5rem;}}
.top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;}}
.label{{font-size:10px;color:#CBD5E1;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}}
.zscore{{font-family:'DM Mono',monospace;font-size:56px;font-weight:500;color:#5652D8;line-height:1;}}
.zone-box{{display:flex;align-items:center;gap:8px;background:{zbg};border:0.5px solid {zbd};border-radius:10px;padding:10px 16px;}}
.zone-dot{{width:8px;height:8px;border-radius:50%;background:{zc};flex-shrink:0;}}
.zone-label{{font-size:14px;font-weight:700;color:{zc};}}.zone-sub{{font-size:11px;color:{zc};opacity:0.6;margin-top:1px;}}
.gauge-wrap{{position:relative;height:6px;background:#242830;border-radius:3px;margin-bottom:8px;overflow:visible;}}
.gauge-track{{position:absolute;left:0;top:0;height:100%;width:100%;border-radius:3px;background:linear-gradient(90deg,#F06060 0%,#F0A030 40%,#3FCF8E 100%);}}
.gauge-marker{{position:absolute;top:-3px;left:5%;width:10px;height:10px;background:#F0EDE6;border-radius:50%;transform:translateX(-50%);border:2px solid #FFFFFF;transition:left 1.2s cubic-bezier(.4,0,.2,1);}}
.gauge-labels{{display:flex;justify-content:space-between;font-size:10px;color:#CBD5E1;font-family:'DM Mono',monospace;}}</style></head>
<body><div class="panel">
  <div class="top"><div><div class="label">Z-Score</div><div class="zscore" id="znum">0.00</div></div>
    <div class="zone-box"><div class="zone-dot"></div>
      <div><div class="zone-label">{zl}</div><div class="zone-sub">{zs}</div></div></div></div>
  <div class="gauge-wrap"><div class="gauge-track"></div><div class="gauge-marker" id="gmark"></div></div>
  <div class="gauge-labels"><span>Distress &lt;1.81</span><span>Grey 1.81–2.99</span><span>Safe &gt;2.99</span></div>
  {warn_html}
</div>
<script>
var target={z:.4f},gp={gp:.1f};
var el=document.getElementById('znum'),mk=document.getElementById('gmark');
var start=null,dur=1200;
function step(ts){{if(!start)start=ts;var p=Math.min((ts-start)/dur,1),e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(2);if(p<1)requestAnimationFrame(step);else el.textContent=target.toFixed(2);}}
requestAnimationFrame(step);setTimeout(function(){{mk.style.left=gp+'%';}},80);
</script></body></html>"""
    components.html(html, height=180)


def render_comparison_card(ticker, d, delay_ms=100):
    if not d["ok"]:
        components.html(f"""<div style="background:#FFFFFF;border:0.5px solid rgba(240,96,96,0.2);border-radius:14px;
                    padding:1.5rem;text-align:center;font-family:sans-serif;">
          <div style="font-family:monospace;font-size:14px;color:#5652D8;margin-bottom:8px;">{ticker}</div>
          <div style="color:#DC2626;font-size:12px;">Data unavailable</div></div>""", height=120)
        return
    z  = d["z"]
    zl, _, zc, zbg, zbd = zone_info(z)
    gp = gauge_pct(z)
    initials_c = ticker[:2]
    if d["logo"]:
        logo_tag = f'<div style="width:36px;height:36px;border-radius:8px;background:#F4F3FF;padding:4px;margin:0 auto 8px;display:flex;align-items:center;justify-content:center;"><img src="{d["logo"]}" style="width:28px;height:28px;object-fit:contain;"></div>'
    else:
        logo_tag = f'<div style="width:36px;height:36px;border-radius:8px;background:rgba(86,82,216,0.1);border:1.5px solid rgba(86,82,216,0.25);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:12px;font-weight:500;color:#5652D8;margin:0 auto 8px;">{initials_c}</div>'
    components.html(f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Inter',sans-serif;}}
.card{{background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;padding:1.2rem 1rem;text-align:center;}}
.ticker{{font-family:'DM Mono',monospace;font-size:12px;color:#5652D8;letter-spacing:1px;margin-bottom:3px;}}
.cname{{font-size:11px;color:#CBD5E1;margin-bottom:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
.znum{{font-family:'DM Mono',monospace;font-size:40px;font-weight:500;color:#5652D8;line-height:1;margin-bottom:10px;}}
.zbadge{{display:inline-block;background:{zbg};border:0.5px solid {zbd};border-radius:8px;padding:5px 12px;margin-bottom:14px;}}
.zbadge span{{font-size:12px;font-weight:700;color:{zc};}}
.gbar{{position:relative;height:4px;background:#242830;border-radius:2px;overflow:visible;margin-bottom:6px;}}
.gtrack{{position:absolute;left:0;top:0;height:100%;width:100%;background:linear-gradient(90deg,#F06060,#F0A030,#3FCF8E);border-radius:2px;}}
.gmarker{{position:absolute;top:-3px;left:5%;width:8px;height:8px;background:#F0EDE6;border-radius:50%;transform:translateX(-50%);border:1.5px solid #FFFFFF;transition:left 1.2s cubic-bezier(.4,0,.2,1);}}
.mcap{{font-size:11px;color:#CBD5E1;font-family:'DM Mono',monospace;}}</style></head>
<body><div class="card">
  {logo_tag}
  <div class="ticker">{ticker}</div><div class="cname">{d['name'][:26]}</div>
  <div class="znum" id="zn">0.00</div>
  <div class="zbadge"><span>{zl}</span></div>
  <div class="gbar"><div class="gtrack"></div><div class="gmarker" id="gm"></div></div>
  <div class="mcap">{fmt(d['mc'])}</div>
</div>
<script>
var target={z:.4f},gp={gp:.1f};
var el=document.getElementById('zn'),mk=document.getElementById('gm');
var start=null,dur=1100;
function step(ts){{if(!start)start=ts;var p=Math.min((ts-start)/dur,1),e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(2);if(p<1)requestAnimationFrame(step);else el.textContent=target.toFixed(2);}}
setTimeout(function(){{requestAnimationFrame(step);mk.style.left=gp+'%';}},{delay_ms});
</script></body></html>""", height=260)


def render_oscore_panel(prob, o_val):
    zl, pct_str, zc, zbg, zbd = o_zone(prob)
    pct = prob * 100
    html = f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Inter',sans-serif;}}
.panel{{background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:16px;padding:1.4rem 1.5rem;}}
.top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;}}
.label{{font-size:10px;color:#CBD5E1;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}}
.prob{{font-family:'DM Mono',monospace;font-size:56px;font-weight:500;color:#5652D8;line-height:1;}}
.zone-box{{display:flex;align-items:center;gap:8px;background:{zbg};border:0.5px solid {zbd};border-radius:10px;padding:10px 16px;}}
.zone-dot{{width:8px;height:8px;border-radius:50%;background:{zc};flex-shrink:0;}}
.zone-label{{font-size:14px;font-weight:700;color:{zc};}}.zone-sub{{font-size:11px;color:{zc};opacity:0.6;margin-top:1px;}}
.bar-wrap{{position:relative;height:6px;background:#242830;border-radius:3px;margin-bottom:8px;overflow:visible;}}
.bar-track{{position:absolute;left:0;top:0;height:100%;width:100%;border-radius:3px;background:linear-gradient(90deg,#3FCF8E 0%,#F0A030 50%,#F06060 100%);}}
.bar-marker{{position:absolute;top:-3px;left:5%;width:10px;height:10px;background:#F0EDE6;border-radius:50%;transform:translateX(-50%);border:2px solid #FFFFFF;transition:left 1.2s cubic-bezier(.4,0,.2,1);}}
.bar-labels{{display:flex;justify-content:space-between;font-size:10px;color:#CBD5E1;font-family:'DM Mono',monospace;}}
.o-val{{margin-top:10px;font-size:11px;color:#CBD5E1;font-family:'DM Mono',monospace;}}</style></head>
<body><div class="panel">
  <div class="top"><div><div class="label">Distress Probability</div><div class="prob" id="pnum">0.0%</div></div>
    <div class="zone-box"><div class="zone-dot"></div>
      <div><div class="zone-label">{zl}</div><div class="zone-sub">O-Score: {o_val:.3f}</div></div></div></div>
  <div class="bar-wrap"><div class="bar-track"></div><div class="bar-marker" id="bmark"></div></div>
  <div class="bar-labels"><span>Low &lt;20%</span><span>Medium 20–50%</span><span>High &gt;50%</span></div>
  <div class="o-val">Raw O-Score: {o_val:.4f} &nbsp;|&nbsp; P = 1/(1+e^-O)</div>
</div>
<script>
var target={pct:.4f},mk=document.getElementById('bmark'),el=document.getElementById('pnum');
var start=null,dur=1200;
function step(ts){{if(!start)start=ts;var p=Math.min((ts-start)/dur,1),e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(1)+'%';if(p<1)requestAnimationFrame(step);else el.textContent=target.toFixed(1)+'%';}}
requestAnimationFrame(step);setTimeout(function(){{mk.style.left=Math.min(94,target)+'%';}},80);
</script></body></html>"""
    components.html(html, height=190)


def render_mscore_panel(m):
    if m > -1.78:
        zone_lbl, zc, zbg, zbd = "Manipulator",     "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"
    elif m > -2.22:
        zone_lbl, zc, zbg, zbd = "Grey Zone",       "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
    else:
        zone_lbl, zc, zbg, zbd = "Non-Manipulator", "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"

    # Map m to gauge position: range -4.5 → +0.5 = 0%→100%
    gp = max(3.0, min(97.0, (m + 4.5) / 5.0 * 100.0))

    html = f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Inter',sans-serif;}}
.panel{{background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:16px;padding:1.4rem 1.5rem;}}
.top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;}}
.lbl{{font-size:10px;color:#CBD5E1;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}}
.mnum{{font-family:'DM Mono',monospace;font-size:56px;font-weight:500;color:#5652D8;line-height:1;}}
.zbox{{display:flex;align-items:center;gap:8px;background:{zbg};border:0.5px solid {zbd};border-radius:10px;padding:10px 16px;}}
.zdot{{width:8px;height:8px;border-radius:50%;background:{zc};flex-shrink:0;}}
.zlbl{{font-size:14px;font-weight:700;color:{zc};}}.zsub{{font-size:11px;color:{zc};opacity:0.6;margin-top:1px;}}
.gw{{position:relative;height:6px;background:#242830;border-radius:3px;margin-bottom:8px;overflow:visible;}}
.gt{{position:absolute;left:0;top:0;height:100%;width:100%;border-radius:3px;
     background:linear-gradient(90deg,#3FCF8E 0%,#3FCF8E 40%,#F0A030 50%,#F06060 100%);}}
.gm{{position:absolute;top:-3px;left:5%;width:10px;height:10px;background:#F0EDE6;border-radius:50%;
     transform:translateX(-50%);border:2px solid #FFFFFF;transition:left 1.2s cubic-bezier(.4,0,.2,1);}}
.gl{{display:flex;justify-content:space-between;font-size:10px;color:#CBD5E1;font-family:'DM Mono',monospace;}}
</style></head>
<body><div class="panel">
  <div class="top">
    <div><div class="lbl">M-Score</div><div class="mnum" id="mn">0.00</div></div>
    <div class="zbox"><div class="zdot"></div>
      <div><div class="zlbl">{zone_lbl}</div>
           <div class="zsub">Below −2.22: safe &middot; −2.22–−1.78: grey &middot; Above −1.78: risk</div></div>
    </div>
  </div>
  <div class="gw"><div class="gt"></div><div class="gm" id="gmark"></div></div>
  <div class="gl"><span>Non-Manipulator &lt;−2.22</span><span>Grey −2.22–−1.78</span><span>Manipulator &gt;−1.78</span></div>
</div>
<script>
var target={m:.4f},gp={gp:.1f};
var el=document.getElementById('mn'),mk=document.getElementById('gmark');
var start=null,dur=1200;
function step(ts){{if(!start)start=ts;var p=Math.min((ts-start)/dur,1),e=1-Math.pow(1-p,3);
  el.textContent=(target*e).toFixed(2);if(p<1)requestAnimationFrame(step);else el.textContent=target.toFixed(2);}}
requestAnimationFrame(step);
setTimeout(function(){{mk.style.left=gp+'%';}},80);
</script></body></html>"""
    components.html(html, height=185)


def render_criteria_group(group_title, keys, flags):
    rows = ""
    for k in keys:
        short, desc, passes, val = flags[k]
        ic     = "✓" if passes else "✗"
        ic_col = "#3FCF8E" if passes else "#F06060"
        ic_bg  = "rgba(63,207,142,0.1)" if passes else "rgba(240,96,96,0.08)"
        ic_bd  = "rgba(63,207,142,0.3)" if passes else "rgba(240,96,96,0.25)"
        rows += (
            f'<div style="display:flex;align-items:center;gap:10px;padding:9px 12px;'
            f'background:#F4F3FF;border-radius:8px;margin-bottom:5px;">'
            f'<div style="width:24px;height:24px;border-radius:50%;background:{ic_bg};'
            f'border:0.5px solid {ic_bd};display:flex;align-items:center;justify-content:center;'
            f'font-size:11px;color:{ic_col};flex-shrink:0;">{ic}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;font-weight:500;'
            f'color:#5652D8;width:22px;flex-shrink:0;">{k}</div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-size:12px;font-weight:600;color:#1E1B4B;line-height:1.2;">{short}</div>'
            f'<div style="font-size:10px;color:#CBD5E1;margin-top:1px;">{desc}</div>'
            f'</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#6B7280;'
            f'text-align:right;flex-shrink:0;max-width:95px;overflow:hidden;text-overflow:ellipsis;">{val}</div>'
            f'</div>'
        )
    return (
        f'<div style="margin-bottom:1rem;">'
        f'<div style="font-size:10px;font-weight:700;color:#CBD5E1;letter-spacing:1.5px;'
        f'text-transform:uppercase;margin-bottom:8px;padding:0 2px;">{group_title}</div>'
        f'{rows}</div>'
    )


def render_fscore_panel(score, flags):
    if score >= 7:
        zone_lbl, zc, zbg, zbd = "Strong",  "#3FCF8E", "#0D2B1F", "rgba(63,207,142,0.2)"
    elif score >= 3:
        zone_lbl, zc, zbg, zbd = "Neutral", "#F0A030", "#2B1A05", "rgba(240,160,48,0.2)"
    else:
        zone_lbl, zc, zbg, zbd = "Weak",    "#F06060", "#2B0D0D", "rgba(240,96,96,0.2)"

    dots_js = "[" + ",".join("1" if flags[f"F{i+1}"][2] else "0" for i in range(9)) + "]"
    dl_html = "".join(f'<div class="dl">F{i+1}</div>' for i in range(9))

    html = f"""<!DOCTYPE html><html>
<head><link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Inter',sans-serif;}}
.panel{{background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:16px;padding:1.4rem 1.5rem;}}
.top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.2rem;}}
.lbl{{font-size:10px;color:#CBD5E1;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}}
.snum{{font-family:'DM Mono',monospace;font-size:56px;font-weight:500;color:#5652D8;line-height:1;}}
.sden{{font-family:'DM Mono',monospace;font-size:28px;font-weight:500;color:#CBD5E1;line-height:1;vertical-align:bottom;padding-bottom:6px;}}
.zbox{{display:flex;align-items:center;gap:8px;background:{zbg};border:0.5px solid {zbd};border-radius:10px;padding:10px 16px;}}
.zdot{{width:8px;height:8px;border-radius:50%;background:{zc};flex-shrink:0;}}
.zlbl{{font-size:14px;font-weight:700;color:{zc};}}
.zsub{{font-size:11px;color:{zc};opacity:0.6;margin-top:1px;}}
.dots{{display:flex;gap:8px;align-items:center;margin-bottom:5px;flex-wrap:nowrap;}}
.dot{{width:30px;height:30px;border-radius:50%;border:1.5px solid #2A2D35;background:#F4F3FF;
      display:flex;align-items:center;justify-content:center;font-size:12px;
      opacity:0;transition:opacity 0.25s;flex-shrink:0;}}
.dot-labels{{display:flex;gap:8px;}}
.dl{{width:30px;text-align:center;font-size:9px;font-family:'DM Mono',monospace;color:#CBD5E1;flex-shrink:0;}}
</style></head>
<body><div class="panel">
  <div class="top">
    <div><div class="lbl">F-Score</div>
      <span class="snum" id="sn">0</span><span class="sden">&thinsp;/9</span>
    </div>
    <div class="zbox"><div class="zdot"></div>
      <div><div class="zlbl">{zone_lbl}</div>
           <div class="zsub">0–2 Weak &middot; 3–6 Neutral &middot; 7–9 Strong</div></div>
    </div>
  </div>
  <div class="dots" id="dr"></div>
  <div class="dot-labels">{dl_html}</div>
</div>
<script>
var d={dots_js},t={score};
var row=document.getElementById('dr');
for(var i=0;i<9;i++){{
  var el=document.createElement('div');el.className='dot';el.id='d'+i;
  if(d[i]){{el.style.background='rgba(63,207,142,0.15)';el.style.borderColor='#3FCF8E';
            el.style.color='#3FCF8E';el.textContent='✓';}}
  else{{el.style.background='rgba(240,96,96,0.08)';el.style.borderColor='rgba(240,96,96,0.3)';
        el.style.color='#F06060';el.textContent='✗';}}
  row.appendChild(el);
}}
for(var j=0;j<9;j++){{
  (function(idx){{setTimeout(function(){{document.getElementById('d'+idx).style.opacity='1';}},60+idx*70);}})(j);
}}
var sn=document.getElementById('sn'),st2=null,dur=700;
function step(ts){{if(!st2)st2=ts;var p=Math.min((ts-st2)/dur,1);
  sn.textContent=Math.round(p*t);if(p<1)requestAnimationFrame(step);else sn.textContent=t;}}
requestAnimationFrame(step);
</script></body></html>"""
    components.html(html, height=200)


# ── Sidebar (session-state based, no href links) ──────────────────────────────
NAV_GROUPS = [
    {
        "label": None,
        "items": [
            ("home", "Home", "#60A5FA", None, None, None),
        ]
    },
    {
        "label": "CORE MODELS",
        "items": [
            ("zscore",  "Altman Z-Score",    "#22C55E", "LIVE", "#22C55E", "rgba(34,197,94,0.15)"),
            ("oscore",  "Ohlson O-Score",    "#F59E0B", "LIVE", "#F59E0B", "rgba(245,158,11,0.15)"),
            ("zscore2", "Zmijewski Score",   "#06B6D4", "LIVE", "#06B6D4", "rgba(6,182,212,0.15)"),
            ("mlscore", "ML Distress Score", "#D946EF", "ML",   "#D946EF", "rgba(217,70,239,0.15)"),
        ]
    },
    {
        "label": "SUPPORTING SIGNALS",
        "items": [
            ("fscore", "Piotroski F-Score", "#22C55E", "LIVE", "#22C55E", "rgba(34,197,94,0.15)"),
            ("mscore",     "Beneish M-Score",     "#EF4444", "LIVE", "#EF4444",     "rgba(239,68,68,0.15)"),
            ("sentiment",  "Sentiment Analysis",  "#F97316", "BETA", "#F97316",  "rgba(249,115,22,0.15)"),
        ]
    },
    {
        "label": "TOOLS",
        "items": [
            ("comparison", "Comparison", "#3B82F6", "LIVE", "#3B82F6", "rgba(59,130,246,0.15)"),
        ]
    },
]
NAV_ITEMS = [item for g in NAV_GROUPS for item in g["items"]]

with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon"><div class="brand-icon-inner"></div></div>
      <div><div class="brand-name-1">Distress<span class="brand-name-2">IQ</span></div></div>
    </div>
    """, unsafe_allow_html=True)

    for g_idx, group in enumerate(NAV_GROUPS):
        if g_idx > 0:
            st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
        if group["label"]:
            st.markdown(f'<div class="nav-section-label">{group["label"]}</div>', unsafe_allow_html=True)
        for page_key, label, dot_color, badge, badge_color, badge_bg in group["items"]:
            if current_page == page_key:
                b_html = (f'<span style="font-size:9px;font-weight:700;font-family:DM Mono,monospace;'
                          f'padding:2px 7px;border-radius:4px;background:{badge_bg};color:{badge_color};">'
                          f'{badge}</span>') if badge else ""
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:9px;padding:7px 12px;border-radius:8px;
                            background:rgba(255,255,255,0.15);margin-bottom:2px;">
                  <span style="width:7px;height:7px;border-radius:50%;background:{dot_color};
                               flex-shrink:0;display:inline-block;"></span>
                  <span style="flex:1;font-size:13px;font-weight:600;color:#FFFFFF;
                               font-family:'Inter',sans-serif;">{label}</span>
                  {b_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                badge_suffix = f"  ·  {badge}" if badge else ""
                if st.button(f"● {label}{badge_suffix}", key=f"nav_{page_key}"):
                    go(page_key)

# ── Investor snapshot helpers ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def compute_investor_snapshot(ticker_str):
    """Fetch key investor metrics: ROE, ROA, margins, ratios, DuPont components."""
    try:
        stock = yf.Ticker(ticker_str)
        info  = stock.info
        bs    = stock.balance_sheet
        inc   = stock.income_stmt
        cf    = stock.cash_flow

        def _g(df, *keys):
            for k in keys:
                try:
                    col = df.columns[0]
                    if k in df.index:
                        v = df.at[k, col]
                        if v is not None and not (isinstance(v, float) and math.isnan(v)):
                            return float(v)
                except Exception:
                    pass
            return None

        ta   = _g(bs, "Total Assets", "TotalAssets")
        te   = _g(bs, "Stockholders Equity", "Total Stockholders Equity",
                  "StockholdersEquity", "TotalEquityGrossMinorityInterest")
        tl   = _g(bs, "Total Liabilities Net Minority Interest",
                  "Total Liabilities", "TotalLiabilities")
        ca   = _g(bs, "Current Assets", "CurrentAssets")
        cl   = _g(bs, "Current Liabilities", "CurrentLiabilities")
        inv  = _g(bs, "Inventory")
        rev  = _g(inc, "Total Revenue", "TotalRevenue", "Revenue")
        gp   = _g(inc, "Gross Profit", "GrossProfit")
        ebit = _g(inc, "EBIT", "Ebit", "Operating Income", "OperatingIncome")
        ni   = _g(inc, "Net Income", "NetIncome")
        dep  = _g(cf,  "Depreciation", "DepreciationAmortization",
                  "Depreciation And Amortization")
        ocf  = _g(cf,  "Operating Cash Flow", "OperatingCashFlow",
                  "Cash Flow From Operations")

        def _pct(num, den):
            try:
                if num is not None and den and den != 0:
                    return num / den * 100
            except Exception:
                pass
            return None

        # Core ratios
        roe          = _pct(ni, te)
        roa          = _pct(ni, ta)
        gross_margin = _pct(gp, rev)
        op_margin    = _pct(ebit, rev)
        net_margin   = _pct(ni,   rev)
        current_ratio = (ca / cl) if (ca and cl and cl != 0) else None
        quick_ratio   = ((ca - (inv or 0)) / cl) if (ca and cl and cl != 0) else None
        debt_equity   = (tl / te) if (te and te != 0 and tl) else None
        asset_turn    = (rev / ta) if (ta and ta != 0 and rev) else None
        eq_mult       = (ta / te) if (te and te != 0 and ta) else None
        fcf           = (ocf - abs(dep or 0)) if ocf is not None else None

        # DuPont decomposition: ROE = net_margin × asset_turn × eq_mult
        dupont_ok = all(v is not None for v in [net_margin, asset_turn, eq_mult])

        return dict(
            ok=True,
            roe=roe, roa=roa,
            gross_margin=gross_margin, op_margin=op_margin, net_margin=net_margin,
            current_ratio=current_ratio, quick_ratio=quick_ratio,
            debt_equity=debt_equity, asset_turn=asset_turn, eq_mult=eq_mult,
            fcf=fcf, rev=rev, ni=ni, ebit=ebit, ta=ta, te=te,
            dupont_ok=dupont_ok,
        )
    except Exception:
        return dict(ok=False)


def _fmt_pct(v, decimals=1):
    if v is None: return "N/A"
    return f"{v:.{decimals}f}%"

def _fmt_x(v, decimals=2):
    if v is None: return "N/A"
    return f"{v:.{decimals}f}x"

def _fmt_b(v):
    if v is None: return "N/A"
    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def _metric_color(v, good_above=None, bad_above=None, neutral=False):
    if v is None or neutral: return "#6B7280"
    if good_above is not None and bad_above is not None:
        if v >= good_above:  return "#16A34A"
        if v <= bad_above:   return "#DC2626"
        return "#D97706"
    if good_above is not None:
        return "#16A34A" if v >= good_above else "#DC2626"
    if bad_above is not None:
        return "#DC2626" if v >= bad_above else "#16A34A"
    return "#6B7280"


def render_investor_snapshot(snap):
    """Render the investor metrics snapshot card below model results."""
    if not snap or not snap.get("ok"):
        return

    s = snap
    rows_left = [
        ("ROE",           _fmt_pct(s.get("roe")),          _metric_color(s.get("roe"),  good_above=15,  bad_above=0)),
        ("ROA",           _fmt_pct(s.get("roa")),          _metric_color(s.get("roa"),  good_above=5,   bad_above=0)),
        ("Gross Margin",  _fmt_pct(s.get("gross_margin")), _metric_color(s.get("gross_margin"), good_above=30)),
        ("Op. Margin",    _fmt_pct(s.get("op_margin")),    _metric_color(s.get("op_margin"),    good_above=10, bad_above=0)),
        ("Net Margin",    _fmt_pct(s.get("net_margin")),   _metric_color(s.get("net_margin"),   good_above=5,  bad_above=0)),
    ]
    rows_right = [
        ("Current Ratio", _fmt_x(s.get("current_ratio")), _metric_color(s.get("current_ratio"), good_above=1.5, bad_above=1.0)),
        ("Quick Ratio",   _fmt_x(s.get("quick_ratio")),   _metric_color(s.get("quick_ratio"),   good_above=1.0, bad_above=0.7)),
        ("Debt / Equity", _fmt_x(s.get("debt_equity")),   _metric_color(s.get("debt_equity"),   bad_above=3.0)),
        ("Asset Turnover",_fmt_x(s.get("asset_turn")),    "#6B7280"),
        ("Eq. Multiplier",_fmt_x(s.get("eq_mult")),       "#6B7280"),
    ]

    def _rows_html(rows):
        out = ""
        for label, val, color in rows:
            out += (f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:5px 0;border-bottom:1px solid #EAE8F8;">'
                    f'<span style="font-size:12px;color:#6B7280;">{label}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:{color};">{val}</span>'
                    f'</div>')
        return out

    st.markdown(
        '<div style="margin-top:20px;padding:16px 18px;background:#FFFFFF;border-radius:12px;'
        'border:1px solid #E8E6F8;box-shadow:0 1px 4px rgba(86,82,216,0.06);">'
        '<div style="font-size:11px;font-weight:700;color:#5652D8;letter-spacing:.5px;'
        'text-transform:uppercase;margin-bottom:12px;">Investor Snapshot</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 24px;">'
        f'<div>{_rows_html(rows_left)}</div>'
        f'<div>{_rows_html(rows_right)}</div>'
        '</div></div>',
        unsafe_allow_html=True
    )
    if s.get("dupont_ok"):
        nm  = s.get("net_margin",  0) or 0
        at  = s.get("asset_turn",  0) or 0
        em  = s.get("eq_mult",     0) or 0
        roe_calc = (nm / 100) * at * em * 100
        st.markdown(
            '<div style="margin-top:10px;padding:12px 14px;background:#F5F3FF;border-radius:10px;'
            'border:1px solid #DDD8FA;">'
            '<div style="font-size:11px;font-weight:700;color:#5652D8;letter-spacing:.5px;'
            'text-transform:uppercase;margin-bottom:10px;">DuPont ROE Decomposition</div>'
            '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
            '<div style="text-align:center;background:#fff;border-radius:8px;padding:6px 10px;'
            'border:1px solid #DDD8FA;min-width:80px;">'
            '<div style="font-size:11px;color:#9CA3AF;margin-bottom:2px;">Net Margin</div>'
            f'<div style="font-weight:700;color:#1E1B4B;">{nm:.1f}%</div>'
            '</div>'
            '<span style="color:#9CA3AF;font-weight:700;">×</span>'
            '<div style="text-align:center;background:#fff;border-radius:8px;padding:6px 10px;'
            'border:1px solid #DDD8FA;min-width:80px;">'
            '<div style="font-size:11px;color:#9CA3AF;margin-bottom:2px;">Asset Turn.</div>'
            f'<div style="font-weight:700;color:#1E1B4B;">{at:.2f}x</div>'
            '</div>'
            '<span style="color:#9CA3AF;font-weight:700;">×</span>'
            '<div style="text-align:center;background:#fff;border-radius:8px;padding:6px 10px;'
            'border:1px solid #DDD8FA;min-width:80px;">'
            '<div style="font-size:11px;color:#9CA3AF;margin-bottom:2px;">Eq. Mult.</div>'
            f'<div style="font-weight:700;color:#1E1B4B;">{em:.2f}x</div>'
            '</div>'
            '<span style="color:#9CA3AF;font-weight:700;">=</span>'
            '<div style="text-align:center;background:#5652D8;border-radius:8px;padding:6px 10px;min-width:80px;">'
            '<div style="font-size:11px;color:rgba(255,255,255,0.7);margin-bottom:2px;">ROE</div>'
            f'<div style="font-weight:700;color:#fff;">{roe_calc:.1f}%</div>'
            '</div>'
            '</div></div>',
            unsafe_allow_html=True
        )


def render_financials_bar(snap):
    """Render a horizontal bar summary of key financials."""
    if not snap or not snap.get("ok"):
        return
    items = [
        ("Revenue",  snap.get("rev")),
        ("EBIT",     snap.get("ebit")),
        ("Net Inc.", snap.get("ni")),
        ("Assets",   snap.get("ta")),
        ("Equity",   snap.get("te")),
    ]
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        color = "#16A34A" if (val and val > 0) else ("#DC2626" if (val and val < 0) else "#6B7280")
        col.markdown(
            f'<div style="text-align:center;padding:10px 4px;background:#F9F8FF;border-radius:8px;'
            f'border:1px solid #EAE8F8;">'
            f'<div style="font-size:10px;color:#9CA3AF;margin-bottom:3px;">{label}</div>'
            f'<div style="font-size:13px;font-weight:700;color:{color};">{_fmt_b(val)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def render_zscore_gauge(z):
    """Return a Plotly gauge figure for Z-Score."""
    if not PLOTLY_AVAILABLE:
        return None
    color = "#16A34A" if z >= 2.99 else ("#EF4444" if z <= 1.81 else "#F59E0B")
    fig = pgo.Figure(pgo.Indicator(
        mode="gauge+number",
        value=z,
        number=dict(font=dict(size=32, color=color, family="Inter"), suffix=""),
        gauge=dict(
            axis=dict(range=[-1, 6], tickwidth=1, tickcolor="#D1D5DB",
                      tickvals=[0, 1.81, 2.99, 4, 6],
                      ticktext=["0", "1.81", "2.99", "4", "6"]),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#F5F3FF",
            borderwidth=0,
            steps=[
                dict(range=[-1, 1.81], color="#FEE2E2"),
                dict(range=[1.81, 2.99], color="#FEF3C7"),
                dict(range=[2.99, 6], color="#DCFCE7"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=z),
        ),
        domain=dict(x=[0, 1], y=[0, 1])
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="#FFFFFF", font=dict(family="Inter"),
    )
    return fig


def render_prob_gauge(prob, title="Distress Probability"):
    """Return a Plotly gauge figure for a probability (0–1)."""
    if not PLOTLY_AVAILABLE:
        return None
    pct   = prob * 100
    color = "#EF4444" if pct >= 50 else ("#F59E0B" if pct >= 30 else "#16A34A")
    fig = pgo.Figure(pgo.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(font=dict(size=32, color=color, family="Inter"), suffix="%"),
        title=dict(text=title, font=dict(size=11, color="#6B7280", family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#D1D5DB",
                      tickvals=[0, 30, 50, 70, 100],
                      ticktext=["0%", "30%", "50%", "70%", "100%"]),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#F5F3FF",
            borderwidth=0,
            steps=[
                dict(range=[0, 30],  color="#DCFCE7"),
                dict(range=[30, 50], color="#FEF3C7"),
                dict(range=[50, 100],color="#FEE2E2"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=pct),
        ),
        domain=dict(x=[0, 1], y=[0, 1])
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="#FFFFFF", font=dict(family="Inter"),
    )
    return fig


# ── Shared: back button ───────────────────────────────────────────────────────
def render_back_button():
    if st.session_state.history:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back", key=f"back_{current_page}"):
            go_back()
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ════════════════════════════════════════════════════════════════════════════
def page_home():
    # ── Intro ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="fu1" style="margin-bottom:2rem;">
      <div style="font-size:34px;font-weight:800;letter-spacing:-0.5px;line-height:1.1;
                  color:#1E1B4B;margin-bottom:10px;">
        Distress<span style="color:#5652D8;">IQ</span>
      </div>
      <div style="font-size:14px;color:#6B7280;line-height:1.7;max-width:600px;">
        No single model reliably predicts financial distress on its own. DistressIQ gives you
        six academic models — Altman, Ohlson, Zmijewski, ML, Piotroski, and Beneish — on any
        public company, so you can triangulate risk rather than rely on one signal.
        Built for analysts, investors, and researchers who need to see the full picture.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── HOW TO USE ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr fu2"><span class="sec-lbl">How to Use This App</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="qs-box">
      <div class="qs-step"><div class="qs-num">1</div>
        <div><b style="color:#1E1B4B;">Use Quick Overview below</b> to scan a company across all models at once — or pick a model from the sidebar for a deeper dive</div></div>
      <div class="qs-step"><div class="qs-num">2</div>
        <div><b style="color:#1E1B4B;">Enter a ticker</b> — e.g. <b style="color:#5652D8;">AAPL</b>,
             <b style="color:#5652D8;">MSFT</b>, <b style="color:#5652D8;">TSLA</b> — and click <b style="color:#1E1B4B;">Analyze →</b></div></div>
      <div class="qs-step"><div class="qs-num">3</div>
        <div><b style="color:#1E1B4B;">Compare results across models</b> — use the Comparison tool to run multiple models side by side and spot divergence</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── MODEL REFERENCE ────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr fu3" style="margin-top:2rem;"><span class="sec-lbl">Model Reference</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #EAE8F8;border-radius:12px;padding:16px 20px;">'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0;">'

        '<div style="padding:10px 12px 10px 0;border-bottom:1px solid #F3F2FC;border-right:1px solid #F3F2FC;">'
        '<div style="font-size:12px;font-weight:700;color:#1E1B4B;margin-bottom:3px;">Altman Z-Score</div>'
        '<div style="font-size:11px;color:#6B7280;line-height:1.5;">5 accounting ratios classify companies into Safe, Grey, or Distress zones. Scores below 1.81 signal elevated risk.</div>'
        '</div>'

        '<div style="padding:10px 0 10px 16px;border-bottom:1px solid #F3F2FC;">'
        '<div style="font-size:12px;font-weight:700;color:#1E1B4B;margin-bottom:3px;">Ohlson O-Score</div>'
        '<div style="font-size:11px;color:#6B7280;line-height:1.5;">9-variable logistic model outputting a direct distress probability (0–100%). Above 50% is a strong warning.</div>'
        '</div>'

        '<div style="padding:10px 12px 10px 0;border-bottom:1px solid #F3F2FC;border-right:1px solid #F3F2FC;">'
        '<div style="font-size:12px;font-weight:700;color:#1E1B4B;margin-bottom:3px;">Zmijewski Score</div>'
        '<div style="font-size:11px;color:#6B7280;line-height:1.5;">Probit model using ROA, leverage, and liquidity to estimate one-year distress probability.</div>'
        '</div>'

        '<div style="padding:10px 0 10px 16px;border-bottom:1px solid #F3F2FC;">'
        '<div style="font-size:12px;font-weight:700;color:#1E1B4B;margin-bottom:3px;">ML Distress Score</div>'
        '<div style="font-size:11px;color:#6B7280;line-height:1.5;">XGBoost classifier trained on 35 years of Compustat data, combining signals from all traditional models.</div>'
        '</div>'

        '<div style="padding:10px 12px 10px 0;border-right:1px solid #F3F2FC;">'
        '<div style="font-size:12px;font-weight:700;color:#1E1B4B;margin-bottom:3px;">Piotroski F-Score</div>'
        '<div style="font-size:11px;color:#6B7280;line-height:1.5;">0–9 score across profitability, leverage, and operating efficiency. 7–9 = strong, 0–2 = weak.</div>'
        '</div>'

        '<div style="padding:10px 0 10px 16px;">'
        '<div style="font-size:12px;font-weight:700;color:#1E1B4B;margin-bottom:3px;">Beneish M-Score</div>'
        '<div style="font-size:11px;color:#6B7280;line-height:1.5;">8 accounting indices detect earnings manipulation likelihood. Above −1.78 suggests possible manipulation.</div>'
        '</div>'

        '</div></div>',
        unsafe_allow_html=True
    )

    # ── QUICK OVERVIEW ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr fu4" style="margin-top:2rem;"><span class="sec-lbl">Quick Overview</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#6B7280;margin-bottom:10px;">Enter any ticker to run all six models and get a snapshot summary.</div>', unsafe_allow_html=True)
    _qo1, _qo2 = st.columns([4, 1])
    with _qo1:
        _qo_tick = st.text_input("qo_tick", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with _qo2:
        _qo_btn = st.button("Run Overview \u2192", key="qo_btn")

    if _qo_btn and _qo_tick.strip():
        _qt = _qo_tick.strip().upper()
        with st.spinner(f"Running all models for {_qt}\u2026"):
            _dz  = compute_zscore(_qt)
            _do  = compute_oscore(_qt)
            _dz2 = compute_zscore2(_qt)
            _df  = compute_fscore(_qt)
            _dm  = compute_mscore(_qt)
            _ml_ok = _ml_available()
            _dml = compute_mlscore(_qt) if _ml_ok else None

        # Company header
        _qo_name   = _dz.get("name",   _qt)  if _dz.get("ok") else _qt
        _qo_sector = _dz.get("sector", "\u2014") if _dz.get("ok") else "\u2014"
        _qo_mc     = _dz.get("mc")     if _dz.get("ok") else None
        _qo_logo   = _dz.get("logo")   if _dz.get("ok") else None
        if _qo_logo:
            _qo_logo_html = (
                '<div style="width:36px;height:36px;border-radius:8px;background:#F4F3FF;'
                'padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;">'
                f'<img src="{_qo_logo}" style="width:28px;height:28px;object-fit:contain;"></div>'
            )
        else:
            _qo_logo_html = (
                '<div style="width:36px;height:36px;border-radius:8px;background:rgba(86,82,216,0.1);'
                'border:1.5px solid rgba(86,82,216,0.25);display:flex;align-items:center;'
                'justify-content:center;font-family:monospace;font-size:12px;font-weight:500;'
                f'color:#5652D8;flex-shrink:0;">{_qt[:2]}</div>'
            )
        _mc_html = (
            '<div style="text-align:right;">'
            '<div style="font-size:10px;color:#CBD5E1;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:18px;font-weight:500;color:#1E1B4B;">{fmt(_qo_mc)}</div>'
            '</div>'
        ) if _qo_mc else ''
        st.markdown(
            '<div style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:12px;'
            'padding:1rem 1.4rem;display:flex;align-items:center;justify-content:space-between;'
            f'flex-wrap:wrap;gap:10px;margin:0.8rem 0;">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'{_qo_logo_html}'
            '<div style="background:rgba(86,82,216,0.08);border:1.5px solid #E8E5F8;border-radius:6px;'
            f'padding:3px 9px;font-family:DM Mono,monospace;font-size:12px;font-weight:500;color:#5652D8;letter-spacing:1px;">{_qt}</div>'
            f'<div><div style="font-size:15px;font-weight:700;color:#1E1B4B;">{_qo_name}</div>'
            f'<div style="font-size:11px;color:#6B7280;">{_qo_sector}</div></div>'
            f'</div>{_mc_html}</div>',
            unsafe_allow_html=True
        )

        # ── Model result cards ──────────────────────────────────────────────
        def _qo_card(title, metric_label, metric_val, zone_label, zone_color, bg_color, ok=True):
            if not ok:
                return (
                    '<div style="background:#F9F8FF;border:1px solid #EAE8F8;border-radius:12px;'
                    'padding:14px 16px;display:flex;flex-direction:column;gap:6px;">'
                    f'<div style="font-size:11px;font-weight:700;color:#9CA3AF;text-transform:uppercase;letter-spacing:.5px;">{title}</div>'
                    '<div style="font-size:12px;color:#9CA3AF;">Data unavailable</div>'
                    '</div>'
                )
            return (
                f'<div style="background:{bg_color};border:1px solid {zone_color}33;border-radius:12px;'
                f'padding:14px 16px;display:flex;flex-direction:column;gap:6px;">'
                f'<div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:.5px;">{title}</div>'
                f'<div style="font-family:DM Mono,monospace;font-size:28px;font-weight:700;color:{zone_color};line-height:1;">{metric_val}</div>'
                f'<div style="font-size:11px;color:#374151;">{metric_label}</div>'
                f'<div style="display:inline-block;background:{zone_color}18;border:1px solid {zone_color}44;'
                f'border-radius:6px;padding:2px 8px;font-size:10px;font-weight:700;color:{zone_color};'
                f'text-transform:uppercase;letter-spacing:.5px;width:fit-content;">{zone_label}</div>'
                '</div>'
            )

        # Z-Score card
        if _dz.get("ok"):
            _qz    = _dz["z"]
            _qz_c  = "#16A34A" if _qz >= 2.99 else ("#EF4444" if _qz <= 1.81 else "#F59E0B")
            _qz_z  = "Safe Zone" if _qz >= 2.99 else ("Grey Zone" if _qz > 1.81 else "Distress Zone")
            _qz_bg = "#F0FDF4"  if _qz >= 2.99 else ("#FFF7ED"  if _qz > 1.81 else "#FEF2F2")
            _z_card = _qo_card("Altman Z-Score", "5-ratio bankruptcy model", f"{_qz:.2f}", _qz_z, _qz_c, _qz_bg)
        else:
            _z_card = _qo_card("Altman Z-Score", "", "", "", "", "", ok=False)

        # O-Score card
        if _do.get("ok"):
            _qo_p  = _do["prob"] * 100
            _qo_c  = "#16A34A" if _qo_p < 30 else ("#EF4444" if _qo_p >= 50 else "#F59E0B")
            _qo_zz = "Low Risk" if _qo_p < 30 else ("High Risk" if _qo_p >= 50 else "Elevated Risk")
            _qo_b2 = "#F0FDF4" if _qo_p < 30 else ("#FEF2F2"  if _qo_p >= 50 else "#FFF7ED")
            _o_card = _qo_card("Ohlson O-Score", "Distress probability", f"{_qo_p:.1f}%", _qo_zz, _qo_c, _qo_b2)
        else:
            _o_card = _qo_card("Ohlson O-Score", "", "", "", "", "", ok=False)

        # Zmijewski card
        if _dz2.get("ok"):
            _qzm    = _dz2["prob"] * 100
            _qzm_c  = "#16A34A" if _qzm < 30 else ("#EF4444" if _qzm >= 50 else "#F59E0B")
            _qzm_zz = "Low Risk" if _qzm < 30 else ("High Risk" if _qzm >= 50 else "Elevated Risk")
            _qzm_bg = "#F0FDF4" if _qzm < 30 else ("#FEF2F2"  if _qzm >= 50 else "#FFF7ED")
            _zm_card = _qo_card("Zmijewski Score", "Probit distress probability", f"{_qzm:.1f}%", _qzm_zz, _qzm_c, _qzm_bg)
        else:
            _zm_card = _qo_card("Zmijewski Score", "", "", "", "", "", ok=False)

        # F-Score card
        if _df.get("ok"):
            _qf    = _df["score"]
            _qf_c  = "#16A34A" if _qf >= 7 else ("#EF4444" if _qf <= 2 else "#F59E0B")
            _qf_zz = "Strong" if _qf >= 7 else ("Weak" if _qf <= 2 else "Moderate")
            _qf_bg = "#F0FDF4" if _qf >= 7 else ("#FEF2F2" if _qf <= 2 else "#FFF7ED")
            _f_card = _qo_card("Piotroski F-Score", "Financial strength (0\u20139)", f"{_qf}/9", _qf_zz, _qf_c, _qf_bg)
        else:
            _f_card = _qo_card("Piotroski F-Score", "", "", "", "", "", ok=False)

        # M-Score card
        if _dm.get("ok"):
            _qm    = _dm["m"]
            _qm_c  = "#EF4444" if _qm > -1.78 else ("#F59E0B" if _qm > -2.22 else "#16A34A")
            _qm_zz = "Possible Manipulation" if _qm > -1.78 else ("Grey Area" if _qm > -2.22 else "Unlikely")
            _qm_bg = "#FEF2F2" if _qm > -1.78 else ("#FFF7ED"  if _qm > -2.22 else "#F0FDF4")
            _m_card = _qo_card("Beneish M-Score", "Earnings manipulation signal", f"{_qm:.2f}", _qm_zz, _qm_c, _qm_bg)
        else:
            _m_card = _qo_card("Beneish M-Score", "", "", "", "", "", ok=False)

        # ML card
        if _ml_ok and _dml and _dml.get("error") is None and _dml.get("probability") is not None:
            _qml    = _dml["probability"] * 100
            _qml_c  = "#16A34A" if _qml < 30 else ("#EF4444" if _qml >= 50 else "#F59E0B")
            _qml_zz = "Low Risk" if _qml < 30 else ("High Risk" if _qml >= 50 else "Elevated Risk")
            _qml_bg = "#F0FDF4" if _qml < 30 else ("#FEF2F2"  if _qml >= 50 else "#FFF7ED")
            _ml_card = _qo_card("ML Distress Score", "XGBoost distress probability", f"{_qml:.1f}%", _qml_zz, _qml_c, _qml_bg)
        else:
            _ml_card = _qo_card("ML Distress Score", "", "", "", "", "", ok=False)

        # Layout: 3 cards per row
        _r1c1, _r1c2, _r1c3 = st.columns(3, gap="small")
        with _r1c1: st.markdown(_z_card,  unsafe_allow_html=True)
        with _r1c2: st.markdown(_o_card,  unsafe_allow_html=True)
        with _r1c3: st.markdown(_zm_card, unsafe_allow_html=True)

        _r2c1, _r2c2, _r2c3 = st.columns(3, gap="small")
        with _r2c1: st.markdown(_f_card,  unsafe_allow_html=True)
        with _r2c2: st.markdown(_m_card,  unsafe_allow_html=True)
        with _r2c3: st.markdown(_ml_card, unsafe_allow_html=True)

        # Signal summary banner
        _safe_n = sum([
            1 if (_dz.get("ok")  and _dz["z"]      >= 2.99)  else 0,
            1 if (_do.get("ok")  and _do["prob"]    <  0.30)  else 0,
            1 if (_dz2.get("ok") and _dz2["prob"]   <  0.30)  else 0,
            1 if (_df.get("ok")  and _df["score"]   >= 7)     else 0,
            1 if (_dm.get("ok")  and _dm["m"]  <= -2.22) else 0,
        ])
        _risk_n = sum([
            1 if (_dz.get("ok")  and _dz["z"]      <= 1.81)  else 0,
            1 if (_do.get("ok")  and _do["prob"]    >= 0.50)  else 0,
            1 if (_dz2.get("ok") and _dz2["prob"]   >= 0.50)  else 0,
            1 if (_df.get("ok")  and _df["score"]   <= 2)     else 0,
            1 if (_dm.get("ok")  and _dm["m"]  >  -1.78) else 0,
        ])
        if _risk_n >= 3:
            _sc = "#DC2626"; _sb = "#FEF2F2"; _sbd = "#FECACA"
            _st = (f"<b>{_risk_n} of 5 models</b> flag elevated risk. The weight of evidence points toward "
                   "financial stress \u2014 corroborate with qualitative analysis before drawing conclusions.")
        elif _risk_n >= 1 and _safe_n >= 2:
            _sc = "#D97706"; _sb = "#FFFBEB"; _sbd = "#FDE68A"
            _st = (f"<b>{_safe_n} models</b> show safe signals while <b>{_risk_n}</b> flag concern. "
                   "The picture is mixed \u2014 use the Comparison page to explore the divergence.")
        elif _safe_n >= 3:
            _sc = "#16A34A"; _sb = "#F0FDF4"; _sbd = "#BBF7D0"
            _st = (f"<b>{_safe_n} of 5 models</b> signal low distress risk. "
                   "Overall the financial picture looks healthy under these academic frameworks.")
        else:
            _sc = "#6B7280"; _sb = "#F9F8FF"; _sbd = "#EAE8F8"
            _st = "Results are inconclusive \u2014 some models lack sufficient data. Open individual model pages for a deeper breakdown."
        st.markdown(
            f'<div style="margin-top:1rem;padding:12px 16px;background:{_sb};border-radius:10px;'
            f'border:1px solid {_sbd};font-size:13px;color:{_sc};line-height:1.7;">'
            '<span style="font-size:10px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;'
            f'display:block;margin-bottom:6px;color:{_sc};">Signal Summary</span>'
            f'{_st}</div>',
            unsafe_allow_html=True
        )
        if st.button("Open Comparison Page \u2192", key="qo_comparison_btn"):
            go("comparison")

    elif _qo_btn:
        st.markdown('<div style="background:rgba(245,158,11,0.06);border:1.5px solid rgba(245,158,11,0.2);border-radius:10px;padding:0.8rem 1.2rem;color:#D97706;font-size:13px;">Please enter a ticker symbol.</div>', unsafe_allow_html=True)

    # ── CORE MODELS ────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="margin-top:2.5rem;"><span class="sec-lbl">Core Models</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
    _cm1, _cm2 = st.columns(2, gap="medium")

    with _cm1:
        st.markdown("""
        <div class="model-card">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#22C55E;"></div>
              <div class="model-card-name">Altman Z-Score</div>
            </div>
            <span class="model-card-badge" style="background:rgba(34,197,94,0.1);color:#16A34A;">LIVE</span>
          </div>
          <div class="model-card-desc">5 financial ratios · Safe / Grey / Distress zones</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Altman Z-Score →", key="home_zscore"):
            go("zscore")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="model-card" style="margin-top:1rem;">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#06B6D4;"></div>
              <div class="model-card-name">Zmijewski Score</div>
            </div>
            <span class="model-card-badge" style="background:rgba(6,182,212,0.1);color:#0891B2;">LIVE</span>
          </div>
          <div class="model-card-desc">3-variable probit model · ROA, Leverage, Liquidity</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Zmijewski Score →", key="home_zscore2"):
            go("zscore2")
        st.markdown('</div>', unsafe_allow_html=True)

    with _cm2:
        st.markdown("""
        <div class="model-card">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#F59E0B;"></div>
              <div class="model-card-name">Ohlson O-Score</div>
            </div>
            <span class="model-card-badge" style="background:rgba(245,158,11,0.1);color:#D97706;">LIVE</span>
          </div>
          <div class="model-card-desc">9 variables · Distress probability as a percentage</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Ohlson O-Score →", key="home_oscore"):
            go("oscore")
        st.markdown('</div>', unsafe_allow_html=True)

        _ml_s  = "LIVE" if _ml_available() else "TRAIN FIRST"
        _ml_bc = "#A855F7" if _ml_available() else "#9CA3AF"
        _ml_bb = "rgba(168,85,247,0.1)" if _ml_available() else "rgba(156,163,175,0.1)"
        _ml_dc = "#D946EF" if _ml_available() else "#9CA3AF"
        st.markdown(f"""
        <div class="model-card" style="margin-top:1rem;">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:{_ml_dc};"></div>
              <div class="model-card-name">ML Distress Score</div>
            </div>
            <span class="model-card-badge" style="background:{_ml_bb};color:{_ml_bc};">{_ml_s}</span>
          </div>
          <div class="model-card-desc">XGBoost · Trained on WRDS/Compustat 1990–2025 · Calibrated probability</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open ML Score →", key="home_mlscore"):
            go("mlscore")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SUPPORTING SIGNALS ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="margin-top:2rem;"><span class="sec-lbl">Supporting Signals</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
    _ss1, _ss2 = st.columns(2, gap="medium")

    with _ss1:
        st.markdown("""
        <div class="model-card">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#22C55E;"></div>
              <div class="model-card-name">Piotroski F-Score</div>
            </div>
            <span class="model-card-badge" style="background:rgba(34,197,94,0.1);color:#16A34A;">LIVE</span>
          </div>
          <div class="model-card-desc">9 criteria · Financial strength score from 0 to 9</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Piotroski F-Score →", key="home_fscore"):
            go("fscore")
        st.markdown('</div>', unsafe_allow_html=True)

    with _ss2:
        st.markdown("""
        <div class="model-card">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#EF4444;"></div>
              <div class="model-card-name">Beneish M-Score</div>
            </div>
            <span class="model-card-badge" style="background:rgba(239,68,68,0.1);color:#DC2626;">LIVE</span>
          </div>
          <div class="model-card-desc">8 indices · Earnings manipulation detection</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Beneish M-Score →", key="home_mscore"):
            go("mscore")
        st.markdown('</div>', unsafe_allow_html=True)

    _sc1, _sc2 = st.columns(2, gap="medium")
    with _sc1:
        st.markdown("""
        <div class="model-card" style="margin-top:1rem;">
          <div class="model-card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="model-card-dot" style="background:#F97316;"></div>
              <div class="model-card-name">Sentiment Analysis</div>
            </div>
            <span class="model-card-badge" style="background:rgba(249,115,22,0.1);color:#EA580C;">BETA</span>
          </div>
          <div class="model-card-desc">News headline tone · Positive / Neutral / Negative signal</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-open-btn">', unsafe_allow_html=True)
        if st.button("Open Sentiment Analysis →", key="home_sentiment"):
            go("sentiment")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── DISCLAIMER ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:2rem;padding:12px 16px;background:#F9F8FF;border-radius:10px;'
        'border:1px solid #EAE8F8;display:flex;align-items:flex-start;gap:10px;">'
        '<div style="font-size:14px;color:#9CA3AF;flex-shrink:0;margin-top:1px;">\u2139</div>'
        '<div style="font-size:11px;color:#9CA3AF;line-height:1.6;">'
        '<b style="color:#6B7280;">Disclaimer:</b> DistressIQ is an analytical research tool for informational purposes only. '
        'It does not constitute investment advice, and its outputs should not be used as the sole basis for any financial decision. '
        'All data is sourced from public filings via yfinance. Always consult a qualified financial professional before investing.'
        '</div></div>',
        unsafe_allow_html=True
    )

def page_zscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#1E1B4B;">
            Altman <span style="color:#5652D8;">Z-Score</span>
          </div>
          <div style="font-size:12px;color:#9CA3AF;margin-top:3px;">5-variable bankruptcy prediction model</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#4440C9;
                    background:rgba(201,168,76,0.08);border:1.5px solid #E8E5F8;
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.2</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;font-weight:500;color:#6B7280;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1:
        t1 = st.text_input("t1", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with c2:
        btn1 = st.button("Analyze →", key="btn1")

    if btn1 and t1.strip():
        ticker = t1.strip().upper()
        with st.spinner("Fetching data..."):
            d = compute_zscore(ticker)

        if not d["ok"]:
            st.markdown('<div style="background:rgba(239,68,68,0.06);border:1.5px solid rgba(239,68,68,0.2);border-radius:10px;padding:1rem 1.2rem;color:#DC2626;font-size:13px;margin-top:1rem;">Required financial fields missing — Z-score could not be calculated.</div>', unsafe_allow_html=True)
        else:
            z = d["z"]
            initials = ticker[:2]
            if d["logo"]:
                logo_html = f'<div style="width:38px;height:38px;border-radius:8px;background:#F4F3FF;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{d["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_html = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(86,82,216,0.1);border:1.5px solid rgba(86,82,216,0.25);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#5652D8;flex-shrink:0;">{initials}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_html}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(86,82,216,0.08);border:1.5px solid #E8E5F8;
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#5652D8;letter-spacing:1px;">{ticker}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#1E1B4B;">{d['name']}</div>
                    <div style="font-size:12px;color:#6B7280;margin-top:1px;">{d['sector']} · {d['country']}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#CBD5E1;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#1E1B4B;margin-top:2px;">{fmt(d['mc'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            snap_z  = compute_investor_snapshot(ticker)
            trend_z = compute_revenue_ebit_trend(ticker)

            # ── Altman Model Inputs ────────────────────────────────────────────
            st.markdown('<div class="fu3 sec-hdr"><span class="sec-lbl">Model Input Variables</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            fin_rows = [("Working Capital", d['wc']), ("Total Assets", d['ta']),
                        ("Retained Earnings", d['re']), ("EBIT", d['eb']),
                        ("Total Liabilities", d['tl']), ("Sales / Revenue", d['rv'])]
            zcols = st.columns(2)
            for i, (lbl, val) in enumerate(fin_rows):
                with zcols[i % 2]:
                    st.markdown(f'<div class="fin-row"><span class="fin-row-label">{lbl}</span><span class="fin-row-val">{fmt(val)}</span></div>', unsafe_allow_html=True)

            st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Altman Z-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _zcolor = "#16A34A" if z >= 2.99 else ("#EF4444" if z <= 1.81 else "#F59E0B")
            _zone   = "Safe Zone" if z >= 2.99 else ("Grey Zone" if z > 1.81 else "Distress Zone")
            _zone_desc = "Z > 2.99 — Low probability of financial distress" if z >= 2.99 else ("1.81 < Z < 2.99 — Uncertain, monitor closely" if z > 1.81 else "Z < 1.81 — High probability of financial distress")
            _pct_pos = min(99, max(1, (z / 6) * 100))
            _x4w = '<div style="margin-top:10px;font-size:11px;color:#D97706;background:#FFFBEB;border:1px solid #FDE68A;border-radius:6px;padding:6px 10px;">⚠ X4 is very large — may inflate Z-score for high-cap firms</div>' if d.get("x4_warn") else ""
            st.markdown(
                '<div style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;padding:1.6rem 2rem;">' +
                '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;">' +
                f'<div style="font-family:DM Mono,monospace;font-size:56px;font-weight:700;color:{_zcolor};line-height:1;">{z:.2f}</div>' +
                f'<div style="text-align:right;"><div style="font-size:11px;color:#9CA3AF;margin-bottom:4px;letter-spacing:.5px;text-transform:uppercase;">Zone</div>' +
                f'<div style="font-size:18px;font-weight:700;color:{_zcolor};">{_zone}</div>' +
                f'<div style="font-size:11px;color:#9CA3AF;margin-top:3px;">{_zone_desc}</div></div>' +
                '</div>' +
                '<div style="position:relative;height:10px;border-radius:999px;background:linear-gradient(to right,#FEE2E2 0%,#FEF3C7 30%,#DCFCE7 50%,#DCFCE7 100%);margin-bottom:4px;">' +
                f'<div style="position:absolute;top:50%;left:{_pct_pos}%;transform:translate(-50%,-50%);width:16px;height:16px;border-radius:50%;background:{_zcolor};border:2px solid #fff;box-shadow:0 0 0 2px {_zcolor};"></div></div>' +
                '<div style="display:flex;justify-content:space-between;font-size:10px;color:#9CA3AF;margin-top:2px;">' +
                '<span>Distress &lt;1.81</span><span>Grey 1.81–2.99</span><span>Safe &gt;2.99</span></div>' +
                _x4w +
                '</div>',
                unsafe_allow_html=True
            )

            # ── Executive Summary ──────────────────────────────────────────────
            _name_short = d['name'].split()[0] if d.get('name') else ticker
            if z >= 2.99:
                _exec = (f"<b>{_name_short}</b> scores <b>{z:.2f}</b> on the Altman Z-Score, placing it firmly in the "
                         f"<b style='color:#16A34A;'>Safe Zone</b>. Across all five financial dimensions — liquidity, "
                         f"profitability, leverage, solvency, and asset efficiency — the company shows no meaningful signs "
                         f"of financial distress under this model.")
            elif z >= 1.81:
                _exec = (f"<b>{_name_short}</b> scores <b>{z:.2f}</b>, landing in the "
                         f"<b style='color:#F59E0B;'>Grey Zone</b>. This is an ambiguous range — financial stress "
                         f"is possible but not confirmed. Some ratios are strong while others show room for concern. "
                         f"This result calls for a closer look at trends and corroboration with other models.")
            else:
                _exec = (f"<b>{_name_short}</b> scores <b>{z:.2f}</b>, falling in the "
                         f"<b style='color:#EF4444;'>Distress Zone</b>. The model identifies significant financial strain "
                         f"across one or more key dimensions. Historically, scores below 1.81 have been associated with "
                         f"elevated bankruptcy risk. This result warrants careful analysis before any investment decision.")
            st.markdown(
                '<div style="margin-top:1rem;padding:14px 18px;background:#F9F8FF;border-radius:10px;'
                'border:1px solid #EAE8F8;font-size:13px;color:#374151;line-height:1.7;">'
                f'<span style="font-size:11px;font-weight:700;color:#5652D8;letter-spacing:.5px;'
                f'text-transform:uppercase;display:block;margin-bottom:8px;">Executive Summary</span>'
                f'{_exec}</div>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="fu5 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Ratio Breakdown</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            ratios = [("X1","Working Capital / Total Assets",d['x1'],"× 1.2"),
                      ("X2","Retained Earnings / Total Assets",d['x2'],"× 1.4"),
                      ("X3","EBIT / Total Assets",d['x3'],"× 3.3"),
                      ("X4","Market Value / Total Liabilities",d['x4'],"× 0.6"),
                      ("X5","Sales / Total Assets",d['x5'],"× 1.0")]
            cols5 = st.columns(5)
            for i, (nm, formula, val, wt) in enumerate(ratios):
                with cols5[i]:
                    st.markdown(f'<div style="background:#F4F3FF;border:1.5px solid #EAE8F8;border-radius:10px;padding:0.9rem 0.8rem;text-align:center;"><div style="font-size:12px;font-weight:700;color:#5652D8;margin-bottom:3px;">{nm}</div><div style="font-size:10px;color:#9CA3AF;margin-bottom:8px;line-height:1.4;">{formula}</div><div style="font-family:\'DM Mono\',monospace;font-size:20px;font-weight:600;color:#1E1B4B;">{val:.3f}</div><div style="font-size:10px;color:#9CA3AF;margin-top:4px;font-family:\'DM Mono\',monospace;">{wt}</div></div>', unsafe_allow_html=True)

            # ── What is driving this score? ────────────────────────────────────
            _contribs = [
                ("X1 · Working Capital / Assets",      d['x1'] * 1.2, d['x1']),
                ("X2 · Retained Earnings / Assets",    d['x2'] * 1.4, d['x2']),
                ("X3 · EBIT / Assets (profitability)", d['x3'] * 3.3, d['x3']),
                ("X4 · Market Value / Liabilities",    d['x4'] * 0.6, d['x4']),
                ("X5 · Sales / Assets (efficiency)",   d['x5'] * 1.0, d['x5']),
            ]
            _sorted_c = sorted(_contribs, key=lambda x: x[1], reverse=True)
            _best  = _sorted_c[0]
            _worst = _sorted_c[-1]
            # Caution: pick the most meaningful concern
            if d['x3'] < 0:
                _caution = "EBIT is negative — the company is not generating operating profit, which is the heaviest driver in the model (weight × 3.3)."
            elif d['x1'] < 0:
                _caution = "Working capital is negative — current liabilities exceed current assets, signalling short-term liquidity pressure."
            elif d['x2'] < 0:
                _caution = "Retained earnings are negative — accumulated losses outweigh profits, suggesting a history of deficit financing."
            elif d.get('x4_warn'):
                _caution = "X4 (Market Value / Liabilities) is unusually large due to the company's high market cap, which may inflate the Z-Score for mega-cap firms."
            else:
                _caution = f"All five ratios are positive. The weakest contributor is {_worst[0].split('·')[1].strip()}, contributing {_worst[1]:.3f} to the total score."

            _best_val_color  = "#16A34A" if _best[1]  > 0 else "#EF4444"
            _worst_val_color = "#EF4444" if _worst[1] < 0 else "#F59E0B"
            st.markdown(
                '<div style="margin-top:1rem;background:#FFFFFF;border:1px solid #EAE8F8;border-radius:12px;padding:14px 18px;">'
                '<div style="font-size:10px;font-weight:700;color:#5652D8;letter-spacing:.5px;text-transform:uppercase;margin-bottom:12px;">What Is Driving This Score?</div>'
                '<div style="display:flex;flex-direction:column;gap:8px;">'
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:#F0FDF4;border-radius:8px;border:1px solid #BBF7D0;">'
                f'<span style="font-size:13px;flex-shrink:0;">✅</span>'
                f'<div><div style="font-size:11px;font-weight:600;color:#15803D;margin-bottom:2px;">Strongest positive driver</div>'
                f'<div style="font-size:12px;color:#374151;">{_best[0]} — weighted contribution: '
                f'<b style="color:{_best_val_color};">{_best[1]:+.3f}</b></div></div></div>'
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:#FFF7ED;border-radius:8px;border:1px solid #FED7AA;">'
                f'<span style="font-size:13px;flex-shrink:0;">⚠️</span>'
                f'<div><div style="font-size:11px;font-weight:600;color:#C2410C;margin-bottom:2px;">Weakest driver</div>'
                f'<div style="font-size:12px;color:#374151;">{_worst[0]} — weighted contribution: '
                f'<b style="color:{_worst_val_color};">{_worst[1]:+.3f}</b></div></div></div>'
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:#F9F8FF;border-radius:8px;border:1px solid #EAE8F8;">'
                f'<span style="font-size:13px;flex-shrink:0;">ℹ️</span>'
                f'<div><div style="font-size:11px;font-weight:600;color:#4B5563;margin-bottom:2px;">One thing to note</div>'
                f'<div style="font-size:12px;color:#374151;">{_caution}</div></div></div>'
                '</div></div>',
                unsafe_allow_html=True
            )

            # ── Interpretation ─────────────────────────────────────────────────
            if z > 2.99:
                interp = (f"The Altman Z-Score of <b>{z:.2f}</b> places {_name_short} in the <b>Safe Zone (Z > 2.99)</b>. "
                          f"Companies in this range have historically shown a low probability of bankruptcy within two years. "
                          f"This does not mean the company is risk-free — it means no acute financial distress signal is present under this model.")
            elif z > 1.81:
                interp = (f"A Z-Score of <b>{z:.2f}</b> puts {_name_short} in the <b>Grey Zone (1.81–2.99)</b>. "
                          f"This range carries real uncertainty — about 20–30% of companies in the grey zone have historically experienced financial difficulty. "
                          f"It is advisable to run this alongside the Ohlson O-Score or Zmijewski model to get a fuller picture.")
            else:
                interp = (f"A Z-Score of <b>{z:.2f}</b> places {_name_short} in the <b>Distress Zone (Z < 1.81)</b>. "
                          f"Altman's original research found that companies with scores in this range had significantly elevated bankruptcy rates within two years. "
                          f"This is a warning signal, not a verdict — always corroborate with other models, qualitative analysis, and management commentary.")
            st.markdown(f'<div class="fu6 interp-box">{interp}</div>', unsafe_allow_html=True)
            render_investor_snapshot(snap_z)

            # ── Revenue & EBIT Trend ───────────────────────────────────────────
            if PLOTLY_AVAILABLE and trend_z:
                _tz_yrs = trend_z["years"]
                _tz_rev = trend_z["revenue"]
                _tz_ebt = trend_z["ebit"]
                # Filter out years where revenue is None
                _pairs = [(y, r, e) for y, r, e in zip(_tz_yrs, _tz_rev, _tz_ebt) if r is not None]
                if _pairs:
                    _f_yrs = [p[0] for p in _pairs]
                    _f_rev = [p[1] for p in _pairs]
                    _f_ebt = [p[2] for p in _pairs]
                    st.markdown('<div class="fu3 sec-hdr" style="margin-top:1.4rem;"><span class="sec-lbl">Revenue &amp; EBIT Trend</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                    _fig_zt = pgo.Figure()
                    _fig_zt.add_trace(pgo.Bar(
                        x=_f_yrs, y=_f_rev,
                        name="Revenue", marker_color="#DDD8FA",
                        marker_line=dict(color="#B8B0F5", width=1),
                        text=[f"${v:.1f}B" for v in _f_rev],
                        textposition="outside", textfont=dict(size=10, color="#6B7280"),
                    ))
                    _ebit_clean_z = [v for v in _f_ebt if v is not None]
                    if _ebit_clean_z:
                        _fig_zt.add_trace(pgo.Scatter(
                            x=_f_yrs, y=_f_ebt,
                            name="EBIT", mode="lines+markers",
                            line=dict(color="#5652D8", width=2.5),
                            marker=dict(size=7, color="#5652D8", line=dict(color="#fff", width=1.5)),
                        ))
                    _fig_zt.update_layout(
                        height=240, margin=dict(l=10, r=10, t=24, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", size=11, color="#374151"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                        yaxis=dict(title="$B", showgrid=True, gridcolor="#F3F2FC",
                                   zeroline=True, zerolinecolor="#E5E3F8", tickfont=dict(size=10)),
                        xaxis=dict(type="category", categoryarray=_f_yrs,
                                   showgrid=False, tickfont=dict(size=11)),
                        bargap=0.35,
                    )
                    st.plotly_chart(_fig_zt, use_container_width=True)

    elif btn1:
        st.markdown('<div style="background:rgba(245,158,11,0.06);border:1.5px solid rgba(245,158,11,0.2);border-radius:10px;padding:1rem 1.2rem;color:#D97706;font-size:13px;margin-top:1rem;">Please enter a ticker symbol.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: O-Score
# ════════════════════════════════════════════════════════════════════════════
def page_oscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#1E1B4B;">
            Ohlson <span style="color:#5652D8;">O-Score</span>
          </div>
          <div style="font-size:12px;color:#9CA3AF;margin-top:3px;">9-variable logistic regression distress model</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#4440C9;
                    background:rgba(201,168,76,0.08);border:1.5px solid #E8E5F8;
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;font-weight:500;color:#6B7280;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    o1, o2 = st.columns([4, 1])
    with o1:
        t_o = st.text_input("to", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with o2:
        btn3 = st.button("Analyze →", key="btn3")

    if btn3 and t_o.strip():
        ticker_o = t_o.strip().upper()
        with st.spinner("Fetching data..."):
            od = compute_oscore(ticker_o)

        if not od["ok"]:
            st.markdown('<div style="background:rgba(239,68,68,0.06);border:1.5px solid rgba(239,68,68,0.2);border-radius:10px;padding:1rem 1.2rem;color:#DC2626;font-size:13px;margin-top:1rem;">Required financial fields missing — O-score could not be calculated.</div>', unsafe_allow_html=True)
        else:
            prob = od["prob"]
            initials_o = ticker_o[:2]
            if od["logo"]:
                logo_o = f'<div style="width:38px;height:38px;border-radius:8px;background:#F4F3FF;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{od["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_o = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(86,82,216,0.1);border:1.5px solid rgba(86,82,216,0.25);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#5652D8;flex-shrink:0;">{initials_o}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_o}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(86,82,216,0.08);border:1.5px solid #E8E5F8;
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#5652D8;letter-spacing:1px;">{ticker_o}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#1E1B4B;">{od['name']}</div>
                    <div style="font-size:12px;color:#6B7280;margin-top:1px;">{od['sector']} · {od['country']}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#CBD5E1;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#1E1B4B;margin-top:2px;">{fmt(od['mc'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            snap_o  = compute_investor_snapshot(ticker_o)
            trend_o = compute_oscore_trend(ticker_o)

            # ── Model Input Variables ─────────────────────────────────────
            st.markdown('<div class="sec-hdr"><span class="sec-lbl">Model Input Variables</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            o_fin_rows = [
                ("Total Assets", od['ta']), ("Total Liabilities", od['tl']),
                ("Working Capital", od['wc']), ("Current Assets", od['ca']),
                ("Current Liabilities", od['cl']), ("Net Income", od['ni']),
                ("Net Income (prev)", od['ni_prev']), ("Operating Cash Flow", od['cfo']),
            ]
            ocols2 = st.columns(2)
            for i, (lbl, val) in enumerate(o_fin_rows):
                with ocols2[i % 2]:
                    st.markdown(f'<div class="fin-row"><span class="fin-row-label">{lbl}</span><span class="fin-row-val">{fmt(val) if val is not None else "N/A"}</span></div>', unsafe_allow_html=True)

            # ── O-Score card ──────────────────────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Ohlson O-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _opct  = od["prob"] * 100
            _ocol  = "#EF4444" if _opct >= 50 else ("#F59E0B" if _opct >= 30 else "#16A34A")
            _olbl  = "High Risk" if _opct >= 50 else ("Moderate Risk" if _opct >= 30 else "Low Risk")
            _odesc = "Probability ≥50% — high likelihood of distress" if _opct >= 50 else ("Probability 30–50% — elevated risk, monitor closely" if _opct >= 30 else "Probability <30% — low distress signal")
            _oppos = min(99, max(1, _opct))
            st.markdown(
                '<div style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;padding:1.6rem 2rem;">' +
                '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;">' +
                f'<div style="font-family:DM Mono,monospace;font-size:56px;font-weight:700;color:{_ocol};line-height:1;">{_opct:.1f}%</div>' +
                f'<div style="text-align:right;"><div style="font-size:11px;color:#9CA3AF;margin-bottom:4px;letter-spacing:.5px;text-transform:uppercase;">Risk Level</div>' +
                f'<div style="font-size:18px;font-weight:700;color:{_ocol};">{_olbl}</div>' +
                f'<div style="font-size:11px;color:#9CA3AF;margin-top:3px;">{_odesc}</div></div>' +
                '</div>' +
                '<div style="position:relative;height:10px;border-radius:999px;background:linear-gradient(to right,#DCFCE7 0%,#FEF3C7 30%,#FEE2E2 50%,#FEE2E2 100%);margin-bottom:4px;">' +
                f'<div style="position:absolute;top:50%;left:{_oppos}%;transform:translate(-50%,-50%);width:16px;height:16px;border-radius:50%;background:{_ocol};border:2px solid #fff;box-shadow:0 0 0 2px {_ocol};"></div></div>' +
                '<div style="display:flex;justify-content:space-between;font-size:10px;color:#9CA3AF;margin-top:2px;">' +
                '<span>0% · Low</span><span>30% · Moderate</span><span>50% · High</span><span>100%</span></div>' +
                '</div>',
                unsafe_allow_html=True
            )

            # ── Executive summary ─────────────────────────────────────────
            if _opct < 20:
                _exec_sum = (
                    f"The O-Score model assigns <b>{od['name']}</b> a distress probability of <b>{_opct:.1f}%</b> — "
                    "well below the 30% watch threshold. Leverage, liquidity, and profitability are all reading in "
                    "healthy ranges at this point in time. This does not rule out future stress if conditions change, "
                    "but the current signal is reassuring."
                )
            elif _opct < 50:
                _exec_sum = (
                    f"The O-Score model flags <b>{od['name']}</b> with a distress probability of <b>{_opct:.1f}%</b> — "
                    "in the elevated watch zone (30–50%). One or more of the model's key drivers — leverage, "
                    "profitability, or cash flow coverage — is showing stress. The company is not in acute danger, "
                    "but the signal warrants closer monitoring."
                )
            else:
                _exec_sum = (
                    f"The O-Score model assigns <b>{od['name']}</b> a distress probability of <b>{_opct:.1f}%</b> — "
                    "above the 50% high-risk threshold. This indicates significant financial stress across multiple "
                    "model dimensions. Ohlson (1980) found companies above this threshold have a substantially higher "
                    "probability of bankruptcy within two years."
                )
            st.markdown(
                '<div style="background:rgba(86,82,216,0.04);border-left:3px solid #5652D8;border-radius:0 8px 8px 0;'
                'padding:1rem 1.2rem;margin:1rem 0;font-size:13px;color:#374151;line-height:1.7;">'
                + _exec_sum + '</div>',
                unsafe_allow_html=True
            )

            # ── Variable Breakdown (improved fonts) ───────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Variable Breakdown</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            o_vars = [
                ("X1","log(Total Assets)",od['x1'],"−0.407"),
                ("X2","Total Liab / Assets",od['x2'],"+6.03"),
                ("X3","Working Capital / Assets",od['x3'],"−1.43"),
                ("X4","CL / Current Assets",od['x4'],"+0.076"),
                ("X5","Insolvent (0/1)",od['x5'],"−1.72"),
            ]
            o_vars2 = [
                ("X6","Net Income / Assets",od['x6'],"−2.37"),
                ("X7","CFO / Total Liab",od['x7'],"−1.83"),
                ("X8","2yr Loss (0/1)",od['x8'],"+0.285"),
                ("X9","NI Change Ratio",od['x9'],"−0.521"),
            ]
            _vc_style_a = "background:#F4F3FF;border:1.5px solid #EAE8F8;border-radius:10px;padding:0.9rem 0.8rem;text-align:center;"
            _vc_style_b = "background:#F4F3FF;border:1.5px solid #EAE8F8;border-radius:10px;padding:0.9rem 0.8rem;text-align:center;margin-top:8px;"
            cols5o = st.columns(5)
            for i, (nm, formula, val, wt) in enumerate(o_vars):
                with cols5o[i]:
                    st.markdown(
                        f'<div style="{_vc_style_a}">'
                        f'<div style="font-size:11px;font-weight:700;color:#5652D8;margin-bottom:3px;">{nm}</div>'
                        f'<div style="font-size:10px;color:#9CA3AF;margin-bottom:8px;line-height:1.3;">{formula}</div>'
                        f'<div style="font-family:DM Mono,monospace;font-size:20px;font-weight:500;color:#1E1B4B;">{val:.3f}</div>'
                        f'<div style="font-size:10px;color:#CBD5E1;margin-top:3px;font-family:DM Mono,monospace;">{wt}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            cols4o = st.columns(4)
            for i, (nm, formula, val, wt) in enumerate(o_vars2):
                with cols4o[i]:
                    st.markdown(
                        f'<div style="{_vc_style_b}">'
                        f'<div style="font-size:11px;font-weight:700;color:#5652D8;margin-bottom:3px;">{nm}</div>'
                        f'<div style="font-size:10px;color:#9CA3AF;margin-bottom:8px;line-height:1.3;">{formula}</div>'
                        f'<div style="font-family:DM Mono,monospace;font-size:20px;font-weight:500;color:#1E1B4B;">{val:.3f}</div>'
                        f'<div style="font-size:10px;color:#CBD5E1;margin-top:3px;font-family:DM Mono,monospace;">{wt}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # ── What Is Driving This Score? ───────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">What Is Driving This Score?</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _o_contribs = [
                ("X2 · Leverage (TL/TA)",       6.03  * od['x2'],  od['x2'],  "High leverage raises distress probability significantly."),
                ("X6 · Profitability (NI/TA)",  -2.37  * od['x6'], od['x6'],  "Losses increase distress probability; profit reduces it."),
                ("X7 · Cash Coverage (CFO/TL)", -1.83  * od['x7'], od['x7'],  "Poor cash flow relative to liabilities signals stress."),
                ("X3 · Working Capital (WC/TA)",-1.43  * od['x3'], od['x3'],  "Low working capital reduces ability to meet short-term obligations."),
                ("X5 · Insolvency Flag",        -1.72  * od['x5'], od['x5'],  "Liabilities exceeding assets is a direct insolvency signal."),
            ]
            _o_sorted  = sorted(_o_contribs, key=lambda d: d[1], reverse=True)
            _o_top     = _o_sorted[0]
            _o_best    = _o_sorted[-1]
            _o_caution = max(_o_contribs, key=lambda d: abs(d[1]))
            st.markdown(
                '<div style="display:grid;gap:8px;margin-top:4px;">'
                + '<div style="background:#FFF9F9;border:1.5px solid rgba(239,68,68,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">'
                + f'<span style="color:#EF4444;font-weight:700;">&#8593; Most risk-elevating: </span><b>{_o_top[0]}</b> — {_o_top[3]}'
                + '</div>'
                + '<div style="background:#F0FDF4;border:1.5px solid rgba(22,163,74,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">'
                + f'<span style="color:#16A34A;font-weight:700;">&#10003; Least concerning: </span><b>{_o_best[0]}</b> — current value: {_o_best[2]:.3f}'
                + '</div>'
                + '<div style="background:#FFFBEB;border:1.5px solid rgba(245,158,11,0.2);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">'
                + f'<span style="color:#D97706;font-weight:700;">&#9432; Largest impact: </span><b>{_o_caution[0]}</b> — contributes {_o_caution[1]:+.3f} to the O-Score logit.'
                + '</div>'
                + '</div>',
                unsafe_allow_html=True
            )

            # ── Interpretation ────────────────────────────────────────────
            _pct = _opct
            if _pct < 20:
                _interp = (
                    f"At <b>{_pct:.1f}%</b> distress probability, the Ohlson model places this company in a <b>low-risk</b> range. "
                    "Ohlson (1980) validated his model on NYSE/AMEX firms, finding the 50% threshold as the primary decision boundary. "
                    "A sub-20% reading suggests no current financial stress — but leverage or profitability deterioration "
                    "could change the signal within one or two reporting periods."
                )
            elif _pct < 50:
                _interp = (
                    f"At <b>{_pct:.1f}%</b>, the O-Score places this company in a <b>moderate-risk watch zone</b>. "
                    "Historical research shows firms in the 30–50% band face elevated creditor scrutiny and are more "
                    "likely to experience covenant stress. This is an early warning signal — "
                    "monitoring leverage trends and cash generation over the next 1–2 reporting periods is advisable."
                )
            else:
                _interp = (
                    f"At <b>{_pct:.1f}%</b>, the O-Score model flags this company as <b>high risk</b>. "
                    "In Ohlson's original study, companies above the 50% threshold had a significantly elevated rate of "
                    "bankruptcy within two years. This reading typically reflects a combination of high leverage, "
                    "poor cash generation, and sustained losses — the signal demands immediate analysis of "
                    "refinancing options and liquidity runway."
                )
            st.markdown(f'<div class="interp-box">{_interp}</div>', unsafe_allow_html=True)

            # ── Investor Snapshot ─────────────────────────────────────────
            render_investor_snapshot(snap_o)

            # ── Key Driver Trends chart (after investor snapshot) ─────────
            if trend_o and PLOTLY_AVAILABLE and len(trend_o["years"]) >= 2:
                st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Key Driver Trends</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:12px;color:#6B7280;margin-bottom:0.8rem;">'
                    'Leverage (TL/TA) and profitability (NI/TA) are the two highest-weight O-Score drivers. '
                    'Rising leverage or falling profitability over time raises distress probability.</div>',
                    unsafe_allow_html=True
                )
                _yrs = trend_o["years"]
                fig_o = pgo.Figure()
                fig_o.add_trace(pgo.Bar(
                    name="Leverage (TL/TA)",
                    x=_yrs,
                    y=trend_o["leverage"],
                    marker_color="#5652D8",
                    opacity=0.85,
                    yaxis="y1",
                ))
                fig_o.add_trace(pgo.Scatter(
                    name="Profitability (NI/TA)",
                    x=_yrs,
                    y=trend_o["profitability"],
                    mode="lines+markers",
                    line=dict(color="#F59E0B", width=2.5),
                    marker=dict(size=7, color="#F59E0B"),
                    yaxis="y2",
                ))
                fig_o.update_layout(
                    height=280,
                    margin=dict(l=8, r=8, t=20, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=11, color="#6B7280"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                    xaxis=dict(type="category", categoryorder="array", categoryarray=_yrs, showgrid=False, tickfont=dict(size=11)),
                    yaxis=dict(title="TL/TA", showgrid=True, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10), zeroline=True, zerolinecolor="rgba(0,0,0,0.1)"),
                    yaxis2=dict(title="NI/TA", overlaying="y", side="right", showgrid=False, tickfont=dict(size=10), zeroline=True, zerolinecolor="rgba(245,158,11,0.3)"),
                    bargap=0.35,
                )
                st.plotly_chart(fig_o, use_container_width=True)

    elif btn3:
        st.markdown('<div class="alert-warn">Please enter a ticker symbol.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Comparison
# ════════════════════════════════════════════════════════════════════════════
def page_comparison():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div>
        <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#1E1B4B;">
          Competitor <span style="color:#5652D8;">Comparison</span>
        </div>
        <div style="font-size:12px;color:#9CA3AF;margin-top:3px;">Side-by-side analysis across all 4 models for up to 3 companies</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;font-weight:500;color:#6B7280;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Enter up to 3 tickers to compare</div>', unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 1])
    with cc1: t_a = st.text_input("ta", placeholder="AAPL", label_visibility="collapsed")
    with cc2: t_b = st.text_input("tb", placeholder="MSFT", label_visibility="collapsed")
    with cc3: t_c = st.text_input("tc", placeholder="GOOGL", label_visibility="collapsed")
    with cc4: btn2 = st.button("Compare →", key="btn2")

    if btn2:
        tickers = [t.strip().upper() for t in [t_a, t_b, t_c] if t.strip()]
        if not tickers:
            st.markdown('<div style="background:rgba(245,158,11,0.06);border:1.5px solid rgba(245,158,11,0.2);border-radius:10px;padding:1rem;color:#D97706;font-size:13px;margin-top:1rem;">Please enter at least one ticker.</div>', unsafe_allow_html=True)
        else:
            all_data = {}
            with st.spinner("Fetching all model data..."):
                for t in tickers:
                    all_data[t] = {
                        "z": compute_zscore(t),
                        "o": compute_oscore(t),
                        "f": compute_fscore(t),
                        "m": compute_mscore(t),
                    }

            n = len(tickers)

            # ─── helpers ──────────────────────────────────────────────────
            def _logo_tag(d, ticker):
                if d.get("logo"):
                    return (f'<div style="width:32px;height:32px;border-radius:7px;background:#F4F3FF;'
                            f'padding:3px;margin:0 auto 6px;display:flex;align-items:center;justify-content:center;">'
                            f'<img src="{d["logo"]}" style="width:26px;height:26px;object-fit:contain;"></div>')
                return (f'<div style="width:32px;height:32px;border-radius:7px;background:rgba(86,82,216,0.1);'
                        f'border:1.5px solid rgba(86,82,216,0.2);display:flex;align-items:center;'
                        f'justify-content:center;font-family:monospace;font-size:11px;font-weight:500;'
                        f'color:#5652D8;margin:0 auto 6px;">{ticker[:2]}</div>')

            def _mini_card(ticker, d, value_str, zone_label, zone_color, zone_bg, zone_bd, highlight=None):
                if not d["ok"]:
                    return (f'<div style="background:#FFFFFF;border:1.5px solid #E8E5F8;'
                            f'border-radius:14px;padding:1.2rem 1rem;text-align:center;">'
                            f'<div style="font-family:DM Mono,monospace;font-size:12px;color:#5652D8;margin-bottom:8px;">{ticker}</div>'
                            f'<div style="color:#9CA3AF;font-size:11px;">Data unavailable</div></div>')
                logo  = _logo_tag(d, ticker)
                cname = d.get("name", ticker)[:24]
                # border/badge for best/worst
                if highlight == "best":
                    outer_border = "border:2px solid #16A34A;"
                    badge = '<div style="font-size:9px;font-weight:700;color:#16A34A;letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px;">&#9733; Best</div>'
                elif highlight == "worst":
                    outer_border = "border:2px solid #EF4444;"
                    badge = '<div style="font-size:9px;font-weight:700;color:#EF4444;letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px;">&#9660; Weakest</div>'
                else:
                    outer_border = "border:1.5px solid #E8E5F8;"
                    badge = '<div style="margin-bottom:6px;height:14px;"></div>'
                return (
                    f'<div style="background:#FFFFFF;{outer_border}'
                    f'border-radius:14px;padding:1.2rem 1rem;text-align:center;">'
                    f'{badge}'
                    f'{logo}'
                    f'<div style="font-family:DM Mono,monospace;font-size:11px;color:#5652D8;'
                    f'letter-spacing:1px;margin-bottom:2px;">{ticker}</div>'
                    f'<div style="font-size:10px;color:#9CA3AF;margin-bottom:10px;overflow:hidden;'
                    f'text-overflow:ellipsis;white-space:nowrap;">{cname}</div>'
                    f'<div style="font-family:DM Mono,monospace;font-size:34px;font-weight:600;'
                    f'color:#1E1B4B;line-height:1;margin-bottom:10px;">{value_str}</div>'
                    f'<div style="display:inline-block;background:{zone_bg};border:1px solid {zone_bd};'
                    f'border-radius:8px;padding:5px 12px;">'
                    f'<span style="font-size:11px;font-weight:700;color:{zone_color};">{zone_label}</span>'
                    f'</div></div>'
                )

            # ─── compute safe/risk signals for overall summary ─────────────
            def _overall_signals(t):
                safe, risk = 0, 0
                zd = all_data[t]["z"]
                if zd["ok"]:
                    z = zd["z"]
                    if z > 2.99: safe += 1
                    elif z < 1.81: risk += 1
                od = all_data[t]["o"]
                if od["ok"]:
                    if od["prob"] < 0.30: safe += 1
                    elif od["prob"] >= 0.50: risk += 1
                fd = all_data[t]["f"]
                if fd["ok"]:
                    if fd["score"] >= 7: safe += 1
                    elif fd["score"] < 3: risk += 1
                md = all_data[t]["m"]
                if md["ok"]:
                    if md["m"] <= -2.22: safe += 1
                    elif md["m"] > -1.78: risk += 1
                return safe, risk

            _signals = {t: _overall_signals(t) for t in tickers}
            _scores  = {t: s - r for t, (s, r) in _signals.items()}
            _best_t  = max(_scores, key=lambda t: _scores[t]) if len(tickers) > 1 else tickers[0]
            _worst_t = min(_scores, key=lambda t: _scores[t]) if len(tickers) > 1 else tickers[0]
            _all_agree = len(set("safe" if s > r else ("risk" if r > s else "mixed") for s, r in _signals.values())) == 1

            # ─── comparison summary banner ─────────────────────────────────
            if len(tickers) > 1:
                _best_safe,  _best_risk  = _signals[_best_t]
                _worst_safe, _worst_risk = _signals[_worst_t]
                _agree_txt = (
                    "The models broadly <b>agree</b> — all companies are pointing in the same direction across most signals."
                    if _all_agree else
                    "The models show <b>mixed signals</b> — different models emphasise different aspects, so cross-model context matters."
                )
                _summ_html = (
                    '<div style="background:rgba(86,82,216,0.04);border:1.5px solid #E8E5F8;'
                    'border-radius:12px;padding:1rem 1.4rem;margin:1rem 0 0.5rem;display:grid;gap:6px;">'
                    f'<div style="font-size:13px;color:#374151;line-height:1.7;">'
                    f'<b style="color:#16A34A;">&#9733; Strongest overall: {_best_t}</b> — {_best_safe} safe signal{"s" if _best_safe!=1 else ""}, {_best_risk} risk signal{"s" if _best_risk!=1 else ""} across the four models.'
                    f'</div>'
                )
                if _best_t != _worst_t:
                    _summ_html += (
                        f'<div style="font-size:13px;color:#374151;line-height:1.7;">'
                        f'<b style="color:#EF4444;">&#9660; Weakest overall: {_worst_t}</b> — {_worst_safe} safe signal{"s" if _worst_safe!=1 else ""}, {_worst_risk} risk signal{"s" if _worst_risk!=1 else ""}.'
                        f'</div>'
                    )
                _summ_html += (
                    f'<div style="font-size:12px;color:#6B7280;border-top:1px solid rgba(86,82,216,0.1);padding-top:6px;margin-top:2px;">{_agree_txt}</div>'
                    '</div>'
                )
                st.markdown(_summ_html, unsafe_allow_html=True)

            # ─── Z-Score comparison ────────────────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Altman Z-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _z_vals = {t: all_data[t]["z"]["z"] for t in tickers if all_data[t]["z"]["ok"]}
            _z_best  = max(_z_vals, key=lambda t: _z_vals[t]) if _z_vals else None
            _z_worst = min(_z_vals, key=lambda t: _z_vals[t]) if _z_vals else None
            zcols = st.columns(n)
            for i, t in enumerate(tickers):
                with zcols[i]:
                    _hl = ("best" if t == _z_best else "worst" if t == _z_worst else None) if len(_z_vals) > 1 else None
                    render_comparison_card(t, all_data[t]["z"], delay_ms=100 + i * 200)

            # ─── O-Score comparison ────────────────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.8rem;"><span class="sec-lbl">Ohlson O-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _o_vals  = {t: all_data[t]["o"]["prob"] for t in tickers if all_data[t]["o"]["ok"]}
            _o_best  = min(_o_vals, key=lambda t: _o_vals[t]) if _o_vals else None  # lower prob = better
            _o_worst = max(_o_vals, key=lambda t: _o_vals[t]) if _o_vals else None
            ocols = st.columns(n)
            for i, t in enumerate(tickers):
                with ocols[i]:
                    od = all_data[t]["o"]
                    _hl = ("best" if t == _o_best else "worst" if t == _o_worst else None) if len(_o_vals) > 1 else None
                    if od["ok"]:
                        zl, _, zc, zbg, zbd = o_zone(od["prob"])
                        card = _mini_card(t, od, f"{od['prob']*100:.1f}%", zl, zc, zbg, zbd, highlight=_hl)
                    else:
                        card = _mini_card(t, od, "N/A", "N/A", "#9CA3AF", "rgba(156,163,175,0.08)", "rgba(156,163,175,0.2)")
                    st.markdown(card, unsafe_allow_html=True)

            # ─── F-Score comparison ────────────────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.8rem;"><span class="sec-lbl">Piotroski F-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _f_vals  = {t: all_data[t]["f"]["score"] for t in tickers if all_data[t]["f"]["ok"]}
            _f_best  = max(_f_vals, key=lambda t: _f_vals[t]) if _f_vals else None
            _f_worst = min(_f_vals, key=lambda t: _f_vals[t]) if _f_vals else None
            fcols = st.columns(n)
            for i, t in enumerate(tickers):
                with fcols[i]:
                    fd = all_data[t]["f"]
                    _hl = ("best" if t == _f_best else "worst" if t == _f_worst else None) if len(_f_vals) > 1 else None
                    if fd["ok"]:
                        sc = fd["score"]
                        if sc >= 7:   fzl, fzc, fzbg, fzbd = "Strong",  "#16A34A", "rgba(22,163,74,0.08)",  "rgba(22,163,74,0.2)"
                        elif sc >= 3: fzl, fzc, fzbg, fzbd = "Neutral", "#D97706", "rgba(217,119,6,0.08)",  "rgba(217,119,6,0.2)"
                        else:         fzl, fzc, fzbg, fzbd = "Weak",    "#EF4444", "rgba(239,68,68,0.08)",  "rgba(239,68,68,0.2)"
                        card = _mini_card(t, fd, f"{sc}/9", fzl, fzc, fzbg, fzbd, highlight=_hl)
                    else:
                        card = _mini_card(t, fd, "N/A", "N/A", "#9CA3AF", "rgba(156,163,175,0.08)", "rgba(156,163,175,0.2)")
                    st.markdown(card, unsafe_allow_html=True)

            # ─── M-Score comparison ────────────────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.8rem;"><span class="sec-lbl">Beneish M-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _m_vals  = {t: all_data[t]["m"]["m"] for t in tickers if all_data[t]["m"]["ok"]}
            _m_best  = min(_m_vals, key=lambda t: _m_vals[t]) if _m_vals else None  # more negative = cleaner
            _m_worst = max(_m_vals, key=lambda t: _m_vals[t]) if _m_vals else None
            mcols = st.columns(n)
            for i, t in enumerate(tickers):
                with mcols[i]:
                    mrd = all_data[t]["m"]
                    _hl = ("best" if t == _m_best else "worst" if t == _m_worst else None) if len(_m_vals) > 1 else None
                    if mrd["ok"]:
                        mv = mrd["m"]
                        if mv > -1.78:   mzl, mzc, mzbg, mzbd = "Manipulator",     "#EF4444", "rgba(239,68,68,0.08)",  "rgba(239,68,68,0.2)"
                        elif mv > -2.22: mzl, mzc, mzbg, mzbd = "Grey Zone",       "#D97706", "rgba(217,119,6,0.08)",  "rgba(217,119,6,0.2)"
                        else:            mzl, mzc, mzbg, mzbd = "Non-Manipulator", "#16A34A", "rgba(22,163,74,0.08)",  "rgba(22,163,74,0.2)"
                        card = _mini_card(t, mrd, f"{mv:.2f}", mzl, mzc, mzbg, mzbd, highlight=_hl)
                    else:
                        card = _mini_card(t, mrd, "N/A", "N/A", "#9CA3AF", "rgba(156,163,175,0.08)", "rgba(156,163,175,0.2)")
                    st.markdown(card, unsafe_allow_html=True)

            # ─── Summary Table ─────────────────────────────────────────────
            if n > 1:
                st.markdown('<div class="sec-hdr" style="margin-top:1.8rem;"><span class="sec-lbl">Summary Table</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                fr = f"180px {' '.join(['1fr']*n)}"

                def cell(content, accent=False, header=False, center=False, clr=None, bold=False):
                    c     = clr if clr else ("#5652D8" if accent else ("#6B7280" if header else "#1E1B4B"))
                    mono  = "font-family:DM Mono,monospace;" if not header else ""
                    align = "center" if center else "left"
                    fw    = "font-weight:600;" if bold else ""
                    bg    = "#F9F8FF" if header else "#FFFFFF"
                    return f'<div style="padding:9px 14px;background:{bg};color:{c};{mono}{fw}font-size:12px;text-align:{align};">{content}</div>'

                def sep_row(lbl, dot_col):
                    row  = (f'<div style="padding:6px 14px;background:rgba(86,82,216,0.07);color:{dot_col};'
                            f'font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;'
                            f'border-top:1px solid rgba(86,82,216,0.1);">{lbl}</div>')
                    row += f'<div style="background:rgba(86,82,216,0.07);border-top:1px solid rgba(86,82,216,0.1);"></div>' * n
                    return row

                # header row
                html = (f'<div style="display:grid;grid-template-columns:{fr};gap:1px;'
                        f'background:rgba(86,82,216,0.1);border-radius:12px 12px 0 0;overflow:hidden;">')
                html += cell("Model / Metric", header=True)
                for t in tickers:
                    html += cell(t, accent=True, center=True)
                html += "</div>"

                # body
                html += (f'<div style="display:grid;grid-template-columns:{fr};gap:1px;'
                         f'background:rgba(86,82,216,0.06);border-radius:0 0 12px 12px;overflow:hidden;">')

                # Z-Score block
                html += sep_row("Altman Z-Score", "#5652D8")
                html += cell("Z-Score", header=True)
                for t in tickers:
                    zd = all_data[t]["z"]
                    _bold = zd["ok"] and t == _z_best
                    html += cell(f"{zd['z']:.2f}" if zd["ok"] else "N/A", center=True, bold=_bold)
                html += cell("Zone", header=True)
                for t in tickers:
                    zd = all_data[t]["z"]
                    if zd["ok"]:
                        zl2, _, zc2, _, _ = zone_info(zd["z"])
                        html += cell(zl2, center=True, clr=zc2)
                    else:
                        html += cell("N/A", center=True)
                html += cell("Market Cap", header=True)
                for t in tickers:
                    zd = all_data[t]["z"]
                    html += cell(fmt(zd["mc"]) if zd["ok"] else "N/A", center=True)

                # O-Score block
                html += sep_row("Ohlson O-Score", "#5652D8")
                html += cell("Distress Prob.", header=True)
                for t in tickers:
                    od = all_data[t]["o"]
                    _bold = od["ok"] and t == _o_best
                    html += cell(f"{od['prob']*100:.1f}%" if od["ok"] else "N/A", center=True, bold=_bold)
                html += cell("Zone", header=True)
                for t in tickers:
                    od = all_data[t]["o"]
                    if od["ok"]:
                        zl2, _, zc2, _, _ = o_zone(od["prob"])
                        html += cell(zl2, center=True, clr=zc2)
                    else:
                        html += cell("N/A", center=True)

                # F-Score block
                html += sep_row("Piotroski F-Score", "#5652D8")
                html += cell("F-Score", header=True)
                for t in tickers:
                    fd = all_data[t]["f"]
                    _bold = fd["ok"] and t == _f_best
                    html += cell(f"{fd['score']}/9" if fd["ok"] else "N/A", center=True, bold=_bold)
                html += cell("Zone", header=True)
                for t in tickers:
                    fd = all_data[t]["f"]
                    if fd["ok"]:
                        sc = fd["score"]
                        fzl2 = "Strong" if sc >= 7 else ("Neutral" if sc >= 3 else "Weak")
                        fzc2 = "#16A34A" if sc >= 7 else ("#D97706" if sc >= 3 else "#EF4444")
                        html += cell(fzl2, center=True, clr=fzc2)
                    else:
                        html += cell("N/A", center=True)

                # M-Score block
                html += sep_row("Beneish M-Score", "#5652D8")
                html += cell("M-Score", header=True)
                for t in tickers:
                    mrd = all_data[t]["m"]
                    _bold = mrd["ok"] and t == _m_best
                    html += cell(f"{mrd['m']:.2f}" if mrd["ok"] else "N/A", center=True, bold=_bold)
                html += cell("Zone", header=True)
                for t in tickers:
                    mrd = all_data[t]["m"]
                    if mrd["ok"]:
                        mv = mrd["m"]
                        mzl2 = "Manipulator" if mv > -1.78 else ("Grey Zone" if mv > -2.22 else "Non-Manip.")
                        mzc2 = "#EF4444"     if mv > -1.78 else ("#D97706"   if mv > -2.22 else "#16A34A")
                        html += cell(mzl2, center=True, clr=mzc2)
                    else:
                        html += cell("N/A", center=True)

                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

                # ─── plain-English note ─────────────────────────────────────
                st.markdown(
                    '<div style="background:rgba(86,82,216,0.03);border:1px solid rgba(86,82,216,0.12);'
                    'border-radius:10px;padding:0.9rem 1.2rem;margin-top:1rem;font-size:12px;color:#6B7280;line-height:1.7;">'
                    '<b style="color:#5652D8;">How to read this comparison:</b> Each model measures a different dimension — '
                    'Altman Z-Score and Ohlson O-Score assess financial distress probability, '
                    'Piotroski F-Score measures operating quality and financial strength, '
                    'and Beneish M-Score screens for earnings manipulation risk. '
                    'A company can score well on one dimension and poorly on another. '
                    'The best-highlighted value in each section reflects the strongest result for that specific model — '
                    'not necessarily the strongest company overall.</div>',
                    unsafe_allow_html=True
                )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: M-Score
# ════════════════════════════════════════════════════════════════════════════
def page_mscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#1E1B4B;">
            Beneish <span style="color:#5652D8;">M-Score</span>
          </div>
          <div style="font-size:12px;color:#9CA3AF;margin-top:3px;">8-index earnings manipulation detection model</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#4440C9;
                    background:rgba(201,168,76,0.08);border:1.5px solid #E8E5F8;
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;font-weight:500;color:#6B7280;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    pm1, pm2 = st.columns([4, 1])
    with pm1:
        t_m = st.text_input("tm", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with pm2:
        btn_m = st.button("Analyze →", key="btn_m")

    if btn_m and t_m.strip():
        ticker_m = t_m.strip().upper()
        with st.spinner("Fetching data..."):
            md = compute_mscore(ticker_m)

        if not md["ok"]:
            st.markdown('<div style="background:rgba(239,68,68,0.06);border:1.5px solid rgba(239,68,68,0.2);border-radius:10px;padding:1rem 1.2rem;color:#DC2626;font-size:13px;margin-top:1rem;">Required financial fields missing — M-score could not be calculated.</div>', unsafe_allow_html=True)
        else:
            m    = md["m"]
            idx  = md["idx"]
            initials_m = ticker_m[:2]
            if md["logo"]:
                logo_m = f'<div style="width:38px;height:38px;border-radius:8px;background:#F4F3FF;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{md["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_m = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(86,82,216,0.1);border:1.5px solid rgba(86,82,216,0.25);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#5652D8;flex-shrink:0;">{initials_m}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_m}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(86,82,216,0.08);border:1.5px solid #E8E5F8;
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#5652D8;letter-spacing:1px;">{ticker_m}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#1E1B4B;">{md['name']}</div>
                    <div style="font-size:12px;color:#6B7280;margin-top:1px;">{md['sector']} · {md['country']}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#CBD5E1;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#1E1B4B;margin-top:2px;">{fmt(md['mc'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            snap_m = compute_investor_snapshot(ticker_m)

            # ── M-Score card (no render_financials_bar) ───────────────────
            st.markdown('<div class="fu3 sec-hdr"><span class="sec-lbl">Beneish M-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _m_col  = "#EF4444" if m > -1.78 else ("#F59E0B" if m > -2.22 else "#16A34A")
            _m_lbl  = "Likely Manipulator" if m > -1.78 else ("Grey Zone" if m > -2.22 else "Non-Manipulator")
            _m_desc = "M > −1.78 — High probability of earnings manipulation" if m > -1.78 else ("−2.22 < M < −1.78 — Inconclusive, review closely" if m > -2.22 else "M < −2.22 — Low probability of earnings manipulation")
            _m_pct_pos = min(99, max(1, (m + 5) / 10 * 100))
            st.markdown(
                '<div style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;padding:1.6rem 2rem;">' +
                '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;">' +
                f'<div style="font-family:DM Mono,monospace;font-size:56px;font-weight:700;color:{_m_col};line-height:1;">{m:.2f}</div>' +
                f'<div style="text-align:right;"><div style="font-size:11px;color:#9CA3AF;margin-bottom:4px;letter-spacing:.5px;text-transform:uppercase;">Signal</div>' +
                f'<div style="font-size:18px;font-weight:700;color:{_m_col};">{_m_lbl}</div>' +
                f'<div style="font-size:11px;color:#9CA3AF;margin-top:3px;">{_m_desc}</div></div>' +
                '</div>' +
                '<div style="position:relative;height:10px;border-radius:999px;background:linear-gradient(to right,#DCFCE7 0%,#FEF3C7 60%,#FEE2E2 100%);margin-bottom:4px;">' +
                f'<div style="position:absolute;top:50%;left:{_m_pct_pos}%;transform:translate(-50%,-50%);width:16px;height:16px;border-radius:50%;background:{_m_col};border:2px solid #fff;box-shadow:0 0 0 2px {_m_col};"></div></div>' +
                '<div style="display:flex;justify-content:space-between;font-size:10px;color:#9CA3AF;margin-top:2px;">' +
                '<span>Non-Manip. &lt;−2.22</span><span>Grey −2.22 to −1.78</span><span>Manip. &gt;−1.78</span></div>' +
                '</div>',
                unsafe_allow_html=True
            )

            # ── Executive summary ─────────────────────────────────────────
            if m > -1.78:
                _m_exec = (
                    f"The Beneish model assigns <b>{md['name']}</b> an M-Score of <b>{m:.2f}</b> — "
                    "above the −1.78 threshold that Beneish (1999) identified as the boundary for likely earnings manipulation. "
                    "One or more of the eight financial indices is signalling abnormal patterns in receivables, margins, accruals, "
                    "or asset quality. This does not confirm fraud — it flags elevated manipulation risk that warrants "
                    "closer scrutiny of the financial statements."
                )
            elif m > -2.22:
                _m_exec = (
                    f"The Beneish model assigns <b>{md['name']}</b> an M-Score of <b>{m:.2f}</b> — "
                    "in the grey zone between −2.22 and −1.78. Some indices are showing elevated readings "
                    "but the overall signal is inconclusive. The model cannot clearly classify this as "
                    "manipulator or non-manipulator. Monitoring the flagged indices closely is advisable."
                )
            else:
                _m_exec = (
                    f"The Beneish model assigns <b>{md['name']}</b> an M-Score of <b>{m:.2f}</b> — "
                    "below the −2.22 non-manipulator threshold. The eight financial indices do not show "
                    "the patterns typically associated with earnings manipulation. "
                    "This is a reassuring signal, though it does not guarantee perfect accounting quality."
                )
            st.markdown(
                '<div style="background:rgba(86,82,216,0.04);border-left:3px solid #5652D8;'
                'border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin:1rem 0;'
                'font-size:13px;color:#374151;line-height:1.7;">' + _m_exec + '</div>',
                unsafe_allow_html=True
            )

            # ── Index Breakdown (preserved, improved font sizes) ──────────
            st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Index Breakdown</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            INDEX_ORDER = ["DSRI","GMI","AQI","SGI","DEPI","SGAI","TATA","LVGI"]
            COEFF = {"DSRI":"+0.920","GMI":"+0.528","AQI":"+0.404","SGI":"+0.892",
                     "DEPI":"+0.115","SGAI":"−0.172","TATA":"+4.679","LVGI":"−0.327"}
            COEFF_VALS = {"DSRI":0.920,"GMI":0.528,"AQI":0.404,"SGI":0.892,
                          "DEPI":0.115,"SGAI":-0.172,"TATA":4.679,"LVGI":-0.327}
            icols = st.columns(4)
            for i, key in enumerate(INDEX_ORDER):
                full_name, val, elevated = idx[key]
                with icols[i % 4]:
                    val_str  = f"{val:.3f}" if val is not None else "N/A"
                    dot_col  = "#EF4444" if elevated else "#16A34A"
                    dot_bg   = "rgba(239,68,68,0.08)" if elevated else "rgba(22,163,74,0.08)"
                    dot_bd   = "rgba(239,68,68,0.25)" if elevated else "rgba(22,163,74,0.2)"
                    flag_lbl = "Elevated" if elevated else "Normal"
                    card_bg  = "rgba(239,68,68,0.03)" if elevated else "#F4F3FF"
                    card_bd  = "rgba(239,68,68,0.2)" if elevated else "#EAE8F8"
                    st.markdown(
                        f'<div style="background:{card_bg};border:1.5px solid {card_bd};'
                        f'border-radius:10px;padding:0.85rem 0.9rem;margin-bottom:8px;">'
                        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">'
                        f'<span style="font-family:DM Mono,monospace;font-size:12px;font-weight:700;color:#5652D8;">{key}</span>'
                        f'<span style="font-family:DM Mono,monospace;font-size:9px;color:#CBD5E1;">{COEFF[key]}</span>'
                        f'</div>'
                        f'<div style="font-size:10px;color:#9CA3AF;margin-bottom:6px;line-height:1.3;">{full_name}</div>'
                        f'<div style="font-family:DM Mono,monospace;font-size:20px;font-weight:600;color:#1E1B4B;margin-bottom:6px;">{val_str}</div>'
                        f'<div style="display:inline-flex;align-items:center;gap:5px;background:{dot_bg};'
                        f'border:0.5px solid {dot_bd};border-radius:5px;padding:2px 8px;">'
                        f'<div style="width:5px;height:5px;border-radius:50%;background:{dot_col};"></div>'
                        f'<span style="font-size:9px;font-weight:700;color:{dot_col};letter-spacing:.3px;">{flag_lbl}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

            # ── What Is Driving This Score? ───────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">What Is Driving This Score?</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            # Weighted contribution = coefficient × (value - 1.0 neutral baseline), signed
            _m_contribs = []
            for _k in INDEX_ORDER:
                _fn, _v, _el = idx[_k]
                if _v is not None:
                    _contrib = COEFF_VALS[_k] * (_v - 1.0)
                    _m_contribs.append((_k, _fn, _contrib, _el))
            if _m_contribs:
                _m_sorted   = sorted(_m_contribs, key=lambda x: x[2], reverse=True)
                _m_top      = _m_sorted[0]   # most risk-elevating
                _m_best     = _m_sorted[-1]  # most risk-reducing / reassuring
                _INDEX_NOTES = {
                    "DSRI": "Rising receivables relative to revenue can signal premature revenue recognition.",
                    "GMI":  "A deteriorating gross margin may indicate revenue or cost manipulation.",
                    "AQI":  "High asset quality index suggests growing off-balance-sheet or intangible assets.",
                    "SGI":  "Rapid sales growth creates pressure and opportunity for earnings management.",
                    "DEPI": "Slowing depreciation rates may indicate intentional asset life extension.",
                    "SGAI": "Rising SG&A relative to sales can signal operational deterioration.",
                    "TATA": "High total accruals relative to assets is the strongest manipulation signal in this model.",
                    "LVGI": "Rising leverage increases incentive to manage earnings to meet covenants.",
                }
                _caution_txt = (
                    "TATA (total accruals) carries the highest model weight (+4.679). Even a small rise "
                    "in accruals relative to assets can meaningfully push the M-Score toward the risk zone."
                    if _m_top[0] != "TATA"
                    else "TATA is already the most elevated driver here — it has the single largest coefficient in the model (+4.679) and its effect dominates the overall score."
                )
                st.markdown(
                    '<div style="display:grid;gap:8px;margin-top:4px;">'
                    + '<div style="background:#FFF9F9;border:1.5px solid rgba(239,68,68,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">'
                    + f'<span style="color:#EF4444;font-weight:700;">&#8593; Most concerning: </span>'
                    + f'<b>{_m_top[0]} — {_m_top[1]}</b><br><span style="color:#6B7280;">{_INDEX_NOTES.get(_m_top[0], "")}</span>'
                    + '</div>'
                    + '<div style="background:#F0FDF4;border:1.5px solid rgba(22,163,74,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">'
                    + f'<span style="color:#16A34A;font-weight:700;">&#10003; Most reassuring: </span>'
                    + f'<b>{_m_best[0]} — {_m_best[1]}</b><br><span style="color:#6B7280;">{_INDEX_NOTES.get(_m_best[0], "")}</span>'
                    + '</div>'
                    + '<div style="background:#FFFBEB;border:1.5px solid rgba(245,158,11,0.2);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">'
                    + f'<span style="color:#D97706;font-weight:700;">&#9432; Note: </span>{_caution_txt}'
                    + '</div>'
                    + '</div>',
                    unsafe_allow_html=True
                )

            # ── Interpretation ────────────────────────────────────────────
            if m > -1.78:
                _m_interp = (
                    f"At <b>{m:.2f}</b>, the Beneish M-Score signals a <b>high likelihood of earnings manipulation</b>. "
                    "The model was designed specifically to detect earnings management — not bankruptcy or general financial distress. "
                    "Beneish (1999) found this threshold correctly identifies manipulators roughly 76% of the time. "
                    "A high M-Score means the accounting patterns are unusual, not that fraud is confirmed. "
                    "Scrutinise the elevated indices above, especially TATA, DSRI, and SGI."
                )
            elif m > -2.22:
                _m_interp = (
                    f"At <b>{m:.2f}</b>, the M-Score is in the <b>grey zone</b> — neither clearly clean nor clearly risky. "
                    "The Beneish model targets earnings manipulation specifically: it looks at receivables growth, margin changes, "
                    "accruals, and asset quality shifts. A grey-zone reading means some of these signals are elevated "
                    "but not enough to reach the manipulator threshold. Monitor the flagged indices in the next reporting period."
                )
            else:
                _m_interp = (
                    f"At <b>{m:.2f}</b>, the M-Score places this company in the <b>non-manipulator range</b>. "
                    "The Beneish model is designed to detect earnings manipulation, not general financial health or bankruptcy risk — "
                    "a clean M-Score does not mean the company is financially strong, only that its accounting patterns "
                    "do not currently resemble those seen in earnings manipulation cases. "
                    "Beneish (1999) estimated a false-negative rate of roughly 24%, so some genuine manipulators do score below −2.22."
                )
            st.markdown(f'<div class="fu5 interp-box">{_m_interp}</div>', unsafe_allow_html=True)

            # ── Investor Snapshot ─────────────────────────────────────────
            render_investor_snapshot(snap_m)

            # ── Index chart (after investor snapshot, refined styling) ────
            _idx_k = list(md["idx"].keys())
            _idx_v = [md["idx"][k][1] if md["idx"][k][1] is not None else 1.0 for k in _idx_k]
            _idx_f = [md["idx"][k][2] for k in _idx_k]
            if PLOTLY_AVAILABLE:
                _bar_colors = ["#EF4444" if f else "#5652D8" for f in _idx_f]
                _fig_m = pgo.Figure(pgo.Bar(
                    x=_idx_k,
                    y=_idx_v,
                    marker=dict(color=_bar_colors, opacity=0.85),
                    text=[f"{v:.3f}" for v in _idx_v],
                    textposition="outside",
                    textfont=dict(size=10, family="DM Mono, monospace", color="#6B7280"),
                ))
                _fig_m.add_hline(y=1.0, line_dash="dot", line_color="rgba(0,0,0,0.2)", line_width=1,
                                 annotation_text="Neutral = 1.0", annotation_font_size=9,
                                 annotation_font_color="#9CA3AF")
                _fig_m.update_layout(
                    height=250,
                    margin=dict(l=8, r=8, t=28, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=11, color="#6B7280"),
                    showlegend=False,
                    title=dict(text="Index Values — red bars are elevated above neutral threshold",
                               font=dict(size=11, color="#9CA3AF"), x=0),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                               zeroline=False, tickfont=dict(size=10)),
                    xaxis=dict(showgrid=False, tickfont=dict(size=11, family="DM Mono, monospace")),
                    bargap=0.35,
                )
                st.plotly_chart(_fig_m, use_container_width=True)

    elif btn_m:
        st.markdown('<div class="alert-warn">Please enter a ticker symbol.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# SENTIMENT: keyword-based headline scorer
# ════════════════════════════════════════════════════════════════════════════
_SENT_POS = {
    "beat","beats","beating","surge","surges","surging","record","records","growth",
    "profit","profits","profitability","gain","gains","upgrade","upgraded","strong",
    "strength","rally","rallies","bullish","rise","rises","rising","soar","soars",
    "outperform","outperforms","positive","success","expand","expands","expanding",
    "dividend","dividends","revenue","revenues","acquisition","partnership","launch",
    "innovation","breakthrough","improvement","rebound","recovery","upbeat","optimistic",
    "confident","raised","raise","exceeded","exceed","exceeds","opportunity","award",
    "approved","approval","boost","boosts","boosting","record-high","all-time",
}
_SENT_NEG = {
    "miss","misses","missed","missing","drop","drops","dropping","loss","losses",
    "decline","declines","declining","weak","weakness","cut","cuts","cutting",
    "downgrade","downgraded","risk","risks","debt","lawsuit","lawsuits","fraud",
    "bankrupt","bankruptcy","sell","sells","bear","falls","fall","crash","crashes",
    "warning","warnings","concern","concerns","default","layoffs","layoff",
    "restructure","restructuring","investigation","investigated","probe","fine",
    "fined","penalty","penalties","recall","recalls","suspended","suspension",
    "withdrawal","withdrew","charges","charged","accusation","accused","scandal",
    "writedown","write-down","impairment","shortfall","below","missed","disappoints",
    "disappointing","disappointed","slump","slumps","plunge","plunges","sink","sinks",
    "tumble","tumbles","slide","slides","slid","overvalued","headwinds","pressure",
}

@st.cache_data(ttl=1800, show_spinner=False)
def compute_sentiment(ticker_str):
    """Fetch recent news headlines and score sentiment via keyword matching."""
    import re
    try:
        stock = yf.Ticker(ticker_str)
        info  = stock.info
        name    = info.get("longName", ticker_str)
        sector  = info.get("sector",  "N/A")
        country = info.get("country", "N/A")
        mc      = info.get("marketCap", 0) or 0
        website = info.get("website", "") or ""
        domain  = website.replace("https://","").replace("http://","").split("/")[0]
        logo    = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""

        raw_news = stock.news or []
        if not raw_news:
            return dict(ok=False, name=name, sector=sector, country=country,
                        logo=logo, mc=mc, reason="No recent news found for this ticker.")

        import time
        now_ts = time.time()
        results = []
        for item in raw_news[:15]:
            # Support both old yfinance flat format and new nested content format
            _cb   = item.get("content", {}) or {}
            title = item.get("title", "") or _cb.get("title", "")
            pub   = (item.get("publisher", "")
                     or _cb.get("provider", {}).get("displayName", ""))
            link  = (item.get("link", "")
                     or _cb.get("canonicalUrl", {}).get("url", "")
                     or _cb.get("clickThroughUrl", {}).get("url", ""))
            ts    = item.get("providerPublishTime", 0)
            if not ts:
                _pd = _cb.get("pubDate", "")
                if _pd:
                    try:
                        import datetime as _dt
                        ts = int(_dt.datetime.fromisoformat(
                            _pd.replace("Z", "+00:00")).timestamp())
                    except Exception:
                        ts = 0
            if not title:
                continue
            words = re.findall(r"[a-z]+", title.lower())
            pos = sum(1 for w in words if w in _SENT_POS)
            neg = sum(1 for w in words if w in _SENT_NEG)
            total = max(len(words), 1)
            raw_score = (pos - neg) / total
            if raw_score > 0.04:
                label, color, icon = "Positive", "#16A34A", "🟢"
            elif raw_score < -0.04:
                label, color, icon = "Negative", "#EF4444", "🔴"
            else:
                label, color, icon = "Neutral",  "#6B7280", "⚪"
            age_days = (now_ts - ts) / 86400 if ts else 999
            results.append(dict(title=title, publisher=pub, link=link,
                                score=raw_score, label=label, color=color,
                                icon=icon, age_days=age_days))

        if not results:
            return dict(ok=False, name=name, sector=sector, country=country,
                        logo=logo, mc=mc, reason="Could not parse headline data.")

        avg   = sum(r["score"] for r in results) / len(results)
        pos_n = sum(1 for r in results if r["label"] == "Positive")
        neu_n = sum(1 for r in results if r["label"] == "Neutral")
        neg_n = sum(1 for r in results if r["label"] == "Negative")

        if avg > 0.04:
            overall_label = "Positive"
            overall_color = "#16A34A"
            overall_bg    = "#F0FDF4"
        elif avg < -0.04:
            overall_label = "Negative"
            overall_color = "#EF4444"
            overall_bg    = "#FEF2F2"
        else:
            overall_label = "Neutral"
            overall_color = "#6B7280"
            overall_bg    = "#F9FAFB"

        return dict(
            ok=True, name=name, sector=sector, country=country, logo=logo, mc=mc,
            avg_score=avg, label=overall_label, color=overall_color, bg=overall_bg,
            pos_n=pos_n, neu_n=neu_n, neg_n=neg_n,
            count=len(results), items=results,
        )
    except Exception as ex:
        return dict(ok=False, name=ticker_str, sector="N/A", country="N/A",
                    logo="", mc=0, reason=str(ex))


# PAGE: F-Score
# ════════════════════════════════════════════════════════════════════════════
def page_fscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#1E1B4B;">
            Piotroski <span style="color:#5652D8;">F-Score</span>
          </div>
          <div style="font-size:12px;color:#9CA3AF;margin-top:3px;">9-point financial strength scoring model</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#4440C9;
                    background:rgba(201,168,76,0.08);border:1.5px solid #E8E5F8;
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;font-weight:500;color:#6B7280;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    pf1, pf2 = st.columns([4, 1])
    with pf1:
        t_f = st.text_input("tf", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with pf2:
        btn_f = st.button("Analyze →", key="btn_f")

    if btn_f and t_f.strip():
        ticker_f = t_f.strip().upper()
        with st.spinner("Fetching data..."):
            fd = compute_fscore(ticker_f)

        if not fd["ok"]:
            st.markdown('<div style="background:rgba(239,68,68,0.06);border:1.5px solid rgba(239,68,68,0.2);border-radius:10px;padding:1rem 1.2rem;color:#DC2626;font-size:13px;margin-top:1rem;">Required financial fields missing — F-score could not be calculated.</div>', unsafe_allow_html=True)
        else:
            score = fd["score"]
            flags = fd["flags"]
            initials_f = ticker_f[:2]
            if fd["logo"]:
                logo_f = f'<div style="width:38px;height:38px;border-radius:8px;background:#F4F3FF;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{fd["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_f = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(86,82,216,0.1);border:1.5px solid rgba(86,82,216,0.25);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#5652D8;flex-shrink:0;">{initials_f}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_f}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(86,82,216,0.08);border:1.5px solid #E8E5F8;
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#5652D8;letter-spacing:1px;">{ticker_f}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#1E1B4B;">{fd['name']}</div>
                    <div style="font-size:12px;color:#6B7280;margin-top:1px;">{fd['sector']} · {fd['country']}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#CBD5E1;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#1E1B4B;margin-top:2px;">{fmt(fd['mc'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Model Input Variables (no render_financials_bar) ──────────
            st.markdown('<div class="fu3 sec-hdr"><span class="sec-lbl">Model Input Variables</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            f_fin_rows = [
                ("Total Assets",        fd['ta0']),
                ("Net Income",          fd['ni0']),
                ("Operating Cash Flow", fd['cfo0']),
                ("Long-term Debt",      fd.get('ltd0')),
                ("Current Assets",      fd.get('ca0')),
                ("Current Liabilities", fd.get('cl0')),
                ("Gross Profit",        fd.get('gp0')),
                ("Revenue",             fd.get('rv0')),
            ]
            fcols = st.columns(2)
            for i, (lbl, val) in enumerate(f_fin_rows):
                with fcols[i % 2]:
                    disp = fmt(val) if val is not None else "N/A"
                    st.markdown(f'<div class="fin-row"><span class="fin-row-label">{lbl}</span><span class="fin-row-val">{disp}</span></div>', unsafe_allow_html=True)

            snap_f = compute_investor_snapshot(ticker_f)

            # ── F-Score card (preserve existing layout) ───────────────────
            st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Piotroski F-Score</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            ff1, ff2 = st.columns([1, 2])
            with ff1:
                _sc_col = '#22C55E' if score >= 7 else ('#F59E0B' if score >= 4 else '#EF4444')
                _sc_lbl = 'Strong' if score >= 7 else ('Average' if score >= 4 else 'Weak')
                st.markdown(
                    '<div style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;'
                    'padding:2rem;text-align:center;box-shadow:0 2px 8px rgba(86,82,216,0.06);">'
                    '<div style="font-size:10px;font-weight:700;color:#9CA3AF;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;">F-Score</div>'
                    f'<div style="font-family:DM Mono,monospace;font-size:64px;font-weight:800;color:{_sc_col};line-height:1;">{score}</div>'
                    '<div style="font-size:12px;color:#9CA3AF;margin-top:6px;">out of 9</div>'
                    f'<div style="margin-top:12px;background:{_sc_col}22;border-radius:8px;padding:6px 12px;">'
                    f'<span style="font-size:13px;font-weight:600;color:{_sc_col};">{_sc_lbl}</span></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            with ff2:
                render_fscore_panel(score, flags)

            # ── Executive summary ─────────────────────────────────────────
            if score >= 7:
                _f_exec = (
                    f"<b>{fd['name']}</b> scores <b>{score}/9</b> on the Piotroski F-Score — a <b>strong</b> reading. "
                    "The model awards points for positive profitability signals, improving leverage and liquidity, "
                    "and expanding operating efficiency. A score in this range has historically been associated with "
                    "above-average stock performance in the year following publication of financial results."
                )
            elif score >= 4:
                _f_exec = (
                    f"<b>{fd['name']}</b> scores <b>{score}/9</b> on the Piotroski F-Score — a <b>neutral</b> reading. "
                    "The company is passing some financial strength criteria but failing others. "
                    "This is not a distress signal — it reflects a mixed picture of financial quality "
                    "that warrants monitoring over the next one to two reporting periods."
                )
            else:
                _f_exec = (
                    f"<b>{fd['name']}</b> scores <b>{score}/9</b> on the Piotroski F-Score — a <b>weak</b> reading. "
                    "The model is detecting deteriorating signals across multiple financial dimensions. "
                    "Note that a low F-Score measures declining financial quality, not direct bankruptcy risk — "
                    "but it historically preceded weaker stock performance and warrants closer scrutiny."
                )
            st.markdown(
                '<div style="background:rgba(86,82,216,0.04);border-left:3px solid #5652D8;'
                'border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin:1rem 0;'
                'font-size:13px;color:#374151;line-height:1.7;">' + _f_exec + '</div>',
                unsafe_allow_html=True
            )

            # ── Criteria Breakdown (preserved) ────────────────────────────
            st.markdown('<div class="fu5 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Criteria Breakdown</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            ga, gb, gc = st.columns(3, gap="medium")
            with ga:
                st.markdown(render_criteria_group("A · Profitability", ["F1","F2","F3","F4"], flags), unsafe_allow_html=True)
            with gb:
                st.markdown(render_criteria_group("B · Leverage &amp; Liquidity", ["F5","F6","F7"], flags), unsafe_allow_html=True)
            with gc:
                st.markdown(render_criteria_group("C · Operating Efficiency", ["F8","F9"], flags), unsafe_allow_html=True)

            # ── What Is Driving This Score? ───────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">What Is Driving This Score?</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _grp_a = sum(1 for k in ["F1","F2","F3","F4"] if flags[k][2])
            _grp_b = sum(1 for k in ["F5","F6","F7"]       if flags[k][2])
            _grp_c = sum(1 for k in ["F8","F9"]             if flags[k][2])
            _grp_ratios = [
                ("Profitability",         _grp_a, 4, "Covers earnings quality, cash generation, and ROA trend."),
                ("Leverage & Liquidity",  _grp_b, 3, "Covers debt reduction, current ratio, and share dilution."),
                ("Operating Efficiency",  _grp_c, 2, "Covers gross margin improvement and asset turnover trend."),
            ]
            _grp_sorted = sorted(_grp_ratios, key=lambda x: x[1]/x[2], reverse=True)
            _best_grp   = _grp_sorted[0]
            _worst_grp  = _grp_sorted[-1]
            # Caution: find the first failing criterion in the worst group
            _caution_map = {
                "Profitability":        "Profitability flags (F1–F4) drive 4 of the 9 points. Failure here typically reflects poor earnings quality or declining ROA.",
                "Leverage & Liquidity": "Leverage and liquidity flags (F5–F7) are year-over-year change signals. One missed year can flip all three.",
                "Operating Efficiency": "Efficiency flags (F8–F9) require both gross margin and asset turnover to improve. Both are sensitive to revenue mix shifts.",
            }
            st.markdown(
                '<div style="display:grid;gap:8px;margin-top:4px;">'
                + '<div style="background:#F0FDF4;border:1.5px solid rgba(22,163,74,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">'
                + f'<span style="color:#16A34A;font-weight:700;">&#10003; Strongest area: </span>'
                + f'<b>{_best_grp[0]}</b> — {_best_grp[1]}/{_best_grp[2]} criteria passed. {_best_grp[3]}'
                + '</div>'
                + '<div style="background:#FFF9F9;border:1.5px solid rgba(239,68,68,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">'
                + f'<span style="color:#EF4444;font-weight:700;">&#8593; Weakest area: </span>'
                + f'<b>{_worst_grp[0]}</b> — only {_worst_grp[1]}/{_worst_grp[2]} criteria passed. {_worst_grp[3]}'
                + '</div>'
                + '<div style="background:#FFFBEB;border:1.5px solid rgba(245,158,11,0.2);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">'
                + f'<span style="color:#D97706;font-weight:700;">&#9432; Note: </span>'
                + _caution_map.get(_worst_grp[0], "Review the failing criteria in detail to understand the direction of change.")
                + '</div>'
                + '</div>',
                unsafe_allow_html=True
            )

            # ── Interpretation ────────────────────────────────────────────
            if score >= 7:
                _f_interp = (
                    f"A score of <b>{score}/9</b> places this company in the <b>strong</b> tier (7–9). "
                    "The Piotroski F-Score is a financial strength diagnostic, not a bankruptcy probability model — "
                    "it measures whether profitability, leverage, and operating efficiency are improving or deteriorating. "
                    "Piotroski (2000) found that high-score firms significantly outperformed low-score firms in the following year. "
                    "A strong reading means the business is improving on most of the nine measurable financial dimensions."
                )
            elif score >= 4:
                _f_interp = (
                    f"A score of <b>{score}/9</b> sits in the <b>neutral</b> range (4–6). "
                    "The F-Score captures year-over-year change in financial quality — not absolute values. "
                    "A neutral score means some signals are improving while others are flat or worsening. "
                    "This is a common reading for stable, mature businesses going through a mixed year. "
                    "Watch the weakest category over the next reporting period for directional confirmation."
                )
            else:
                _f_interp = (
                    f"A score of <b>{score}/9</b> is in the <b>weak</b> range (0–3). "
                    "The F-Score is detecting deterioration across multiple financial dimensions simultaneously. "
                    "This does not directly predict bankruptcy — it signals declining operating quality and financial flexibility. "
                    "Piotroski (2000) found low-score firms underperformed the market substantially in the year following scoring. "
                    "The weakest category above is the most productive place to begin further analysis."
                )
            st.markdown(f'<div class="fu6 interp-box">{_f_interp}</div>', unsafe_allow_html=True)

            # ── Investor Snapshot ─────────────────────────────────────────
            render_investor_snapshot(snap_f)

            # ── Category pass chart (after investor snapshot) ─────────────
            if PLOTLY_AVAILABLE:
                st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Score by Category</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:12px;color:#6B7280;margin-bottom:0.8rem;">'
                    'Criteria passed in each of the three Piotroski groups. '
                    'The F-Score rewards improvement, not absolute levels — each bar shows how many signals fired.</div>',
                    unsafe_allow_html=True
                )
                _cats     = ["A · Profitability", "B · Leverage &<br>Liquidity", "C · Efficiency"]
                _passed   = [_grp_a, _grp_b, _grp_c]
                _maxes    = [4, 3, 2]
                _pct_list = [p/m for p,m in zip(_passed, _maxes)]
                _bcolors  = ["#22C55E" if r >= 0.75 else ("#F59E0B" if r >= 0.5 else "#EF4444") for r in _pct_list]
                fig_f = pgo.Figure()
                fig_f.add_trace(pgo.Bar(
                    x=_cats,
                    y=_passed,
                    text=[f"{p}/{m}" for p,m in zip(_passed, _maxes)],
                    textposition="outside",
                    textfont=dict(size=13, family="DM Mono, monospace", color="#1E1B4B"),
                    marker_color=_bcolors,
                    marker_line_width=0,
                    opacity=0.88,
                    width=0.45,
                ))
                # Max-possible reference bars (ghost)
                fig_f.add_trace(pgo.Bar(
                    x=_cats,
                    y=_maxes,
                    marker_color="rgba(0,0,0,0.04)",
                    marker_line_width=0,
                    width=0.45,
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig_f.update_layout(
                    barmode="overlay",
                    height=240,
                    margin=dict(l=8, r=8, t=28, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=11, color="#6B7280"),
                    xaxis=dict(showgrid=False, tickfont=dict(size=12)),
                    yaxis=dict(
                        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                        tickvals=[0,1,2,3,4], tickfont=dict(size=10),
                        range=[0, 4.8], zeroline=False,
                        title="Criteria Passed",
                    ),
                    showlegend=False,
                    bargap=0.4,
                )
                st.plotly_chart(fig_f, use_container_width=True)

    elif btn_f:
        st.markdown('<div class="alert-warn">Please enter a ticker symbol.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ML Score
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def compute_mlscore(ticker_str):
    """Wrapper around the trained XGBoost model — cacheable via Streamlit."""
    return _ml_predict(ticker_str)


def render_mlscore_panel(result: dict):
    """Animated probability gauge for the ML distress score."""
    prob       = result["probability"]
    risk_label = result["risk_label"]
    risk_color = result["risk_color"]
    threshold  = result.get("threshold", 0.5)
    missing    = result.get("missing_pct", 0.0)
    pct        = round(prob * 100, 1)
    # Gauge: 0 = left (low risk), 100 = right (high risk)
    gp = max(3.0, min(97.0, pct))

    html = f"""
<div id="mlpanel" style="background:#FFFFFF;border:0.5px solid rgba(201,107,232,0.2);
     border-radius:14px;padding:1.8rem 2rem 1.5rem;margin-bottom:1rem;">

  <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.4rem;">
    <div style="font-size:11px;font-weight:600;letter-spacing:2px;color:#CBD5E1;text-transform:uppercase;">ML Distress Probability</div>
    <div style="flex:1;height:0.5px;background:rgba(201,107,232,0.1);"></div>
    <div style="font-size:10px;font-weight:600;letter-spacing:1px;
                background:rgba(201,107,232,0.1);color:#C96BE8;
                padding:3px 9px;border-radius:5px;font-family:'DM Mono',monospace;">XGBoost</div>
  </div>

  <!-- Big probability number -->
  <div style="text-align:center;margin-bottom:1.6rem;">
    <div id="ml-pct" style="font-family:'DM Mono',monospace;font-size:52px;font-weight:500;
                              color:{risk_color};line-height:1;">0.0%</div>
    <div style="font-size:13px;font-weight:600;color:{risk_color};margin-top:6px;
                letter-spacing:0.5px;">{risk_label}</div>
    <div style="font-size:11px;color:#CBD5E1;margin-top:4px;">
      Decision threshold: {threshold:.0%}
    </div>
  </div>

  <!-- Gradient bar gauge -->
  <div style="position:relative;height:10px;border-radius:999px;margin:0 0.5rem 0.4rem;
              background:linear-gradient(to right,#3FCF8E 0%,#F0A030 40%,#E85555 80%,#B22020 100%);
              box-shadow:0 0 12px rgba(201,107,232,0.15);">
    <div id="ml-marker" style="position:absolute;top:50%;transform:translate(-50%,-50%);
                                left:0%;transition:left 1s cubic-bezier(.22,1,.36,1);
                                width:16px;height:16px;border-radius:50%;
                                background:{risk_color};
                                box-shadow:0 0 8px {risk_color},0 0 0 3px rgba(0,0,0,0.6);"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;
              font-size:10px;color:#CBD5E1;padding:0 0.5rem;margin-bottom:1.4rem;">
    <span>0% · Low</span>
    <span>{threshold:.0%} · Threshold</span>
    <span>100% · High</span>
  </div>

  <!-- Interpretation bar -->
  <div style="background:#F4F3FF;border-left:2px solid {risk_color};border-radius:0 8px 8px 0;
              padding:0.7rem 1rem;font-size:12px;color:#6B7280;line-height:1.6;">
    {"<b style='color:" + risk_color + ";'>⚠ High distress probability.</b> The model assigns a &gt;50% chance of financial distress within 1 year." if prob >= 0.5
    else "<b style='color:" + risk_color + ";'>⚡ Elevated risk.</b> Probability between 20–50% — monitor closely and corroborate with other models." if prob >= 0.2
    else "<b style='color:" + risk_color + ";'>✓ Low distress risk.</b> The ML model finds no strong signals of near-term financial distress."}
    {"<span style='color:#CBD5E1;'> &nbsp;·&nbsp; " + f"{missing:.0%} of features imputed from training-set medians.</span>" if missing > 0.1 else ""}
  </div>

</div>

<script>
(function(){{
  var target = {pct};
  var marker = document.getElementById('ml-marker');
  var numEl  = document.getElementById('ml-pct');
  if (!marker || !numEl) return;
  var start = null;
  var dur   = 1000;
  function ease(t){{ return t<0.5 ? 4*t*t*t : 1-Math.pow(-2*t+2,3)/2; }}
  function step(ts){{
    if (!start) start = ts;
    var p = Math.min((ts - start) / dur, 1);
    var v = ease(p) * target;
    numEl.textContent  = v.toFixed(1) + '%';
    marker.style.left  = Math.max(1, Math.min(99, v)) + '%';
    if (p < 1) requestAnimationFrame(step);
    else {{ numEl.textContent = target.toFixed(1) + '%'; marker.style.left = '{gp}%'; }}
  }}
  requestAnimationFrame(step);
}})();
</script>
"""
    components.html(html, height=320, scrolling=False)


def page_mlscore():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(201,168,76,0.12);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#1E1B4B;">
            ML <span style="color:#5652D8;">Distress Score</span>
          </div>
          <div style="font-size:12px;color:#9CA3AF;margin-top:3px;">
            XGBoost classifier trained on WRDS/Compustat 1990–2025 data
          </div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8A4FBF;
                    background:rgba(201,107,232,0.08);border:0.5px solid rgba(201,107,232,0.2);
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0 · XGBoost</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not _ml_available():
        st.markdown("""
        <div style="background:#1A1025;border:0.5px solid rgba(201,107,232,0.2);border-radius:14px;
                    padding:1.5rem 1.8rem;margin-top:1rem;">
          <div style="font-size:16px;font-weight:700;color:#C96BE8;margin-bottom:0.7rem;">
            ⚙ Model Not Trained Yet
          </div>
          <div style="font-size:13px;color:#6B7280;line-height:1.7;">
            To enable this feature, run the ML training pipeline from the
            <code style="background:#0E0F12;padding:2px 6px;border-radius:4px;color:#5652D8;">ml_model/</code> directory:
          </div>
          <div style="background:#0E0F12;border-radius:8px;padding:1rem 1.2rem;margin:0.8rem 0;
                      font-family:'DM Mono',monospace;font-size:12px;color:#6B7280;line-height:2;">
            <span style="color:#CBD5E1;"># Step 1 — Pull WRDS data (requires WRDS account)</span><br>
            <span style="color:#5652D8;">python</span> ml_model/01_pull_wrds_data.py<br><br>
            <span style="color:#CBD5E1;"># Step 2 — Engineer features</span><br>
            <span style="color:#5652D8;">python</span> ml_model/02_prepare_features.py<br><br>
            <span style="color:#CBD5E1;"># Step 3 — Train the model (~5 min)</span><br>
            <span style="color:#5652D8;">python</span> ml_model/03_train_model.py
          </div>
          <div style="font-size:12px;color:#CBD5E1;line-height:1.5;">
            After training, restart the app and this page will show live predictions.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<div style="font-size:11px;font-weight:500;color:#6B7280;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    pml1, pml2 = st.columns([4, 1])
    with pml1:
        t_ml = st.text_input("t_ml", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with pml2:
        btn_ml = st.button("Analyze →", key="btn_ml")

    if btn_ml and t_ml.strip():
        ticker_ml = t_ml.strip().upper()
        with st.spinner("Running ML model..."):
            result = compute_mlscore(ticker_ml)
            snap_ml = compute_investor_snapshot(ticker_ml)

        if result.get("error") or result.get("probability") is None:
            err_msg = result.get("error", "Unknown error")
            st.markdown(f'<div style="background:rgba(239,68,68,0.06);border:1.5px solid rgba(239,68,68,0.2);border-radius:10px;padding:1rem 1.2rem;color:#DC2626;font-size:13px;margin-top:1rem;">{err_msg}</div>', unsafe_allow_html=True)
        else:
            prob       = result["probability"]
            risk_color = result["risk_color"]
            risk_label = result["risk_label"]

            # ── Company header (from yfinance info) ──────────────────────────
            try:
                info   = yf.Ticker(ticker_ml).info
                name   = info.get("longName", ticker_ml)
                sector = info.get("sector", "N/A")
                country= info.get("country", "N/A")
                mc     = info.get("marketCap", 0) or 0
                website= info.get("website", "") or ""
                domain = website.replace("https://","").replace("http://","").split("/")[0]
                logo   = f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""
            except Exception:
                name = ticker_ml; sector = "N/A"; country = "N/A"; mc = 0; logo = ""

            initials_ml = ticker_ml[:2]
            if logo:
                logo_ml = f'<div style="width:38px;height:38px;border-radius:8px;background:#F4F3FF;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{logo}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_ml = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(201,107,232,0.15);border:0.5px solid rgba(201,107,232,0.3);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#C96BE8;flex-shrink:0;">{initials_ml}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_ml}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(86,82,216,0.08);border:1.5px solid #E8E5F8;
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#5652D8;letter-spacing:1px;">{ticker_ml}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#1E1B4B;">{name}</div>
                    <div style="font-size:12px;color:#6B7280;margin-top:1px;">{sector} · {country}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#CBD5E1;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#1E1B4B;margin-top:2px;">{fmt(mc)}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="fu3 sec-hdr"><span class="sec-lbl">ML Distress Probability</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _ml_p   = result.get("probability") or 0
            _mlpct  = _ml_p * 100
            _mlcol  = "#EF4444" if _mlpct >= 50 else ("#F59E0B" if _mlpct >= 20 else "#16A34A")
            _mllbl  = result.get("risk_label", "Unknown")
            _mldesc = f"Decision threshold: {result.get('threshold', 0.5):.0%}"
            _mlpos  = min(99, max(1, _mlpct))
            st.markdown(
                '<div style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;padding:1.6rem 2rem;">' +
                '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;">' +
                f'<div style="font-family:DM Mono,monospace;font-size:56px;font-weight:700;color:{_mlcol};line-height:1;">{_mlpct:.1f}%</div>' +
                f'<div style="text-align:right;"><div style="font-size:11px;color:#9CA3AF;margin-bottom:4px;letter-spacing:.5px;text-transform:uppercase;">ML Signal</div>' +
                f'<div style="font-size:18px;font-weight:700;color:{_mlcol};">{_mllbl}</div>' +
                f'<div style="font-size:11px;color:#9CA3AF;margin-top:3px;">{_mldesc}</div></div>' +
                '</div>' +
                '<div style="position:relative;height:10px;border-radius:999px;background:linear-gradient(to right,#DCFCE7 0%,#FEF3C7 20%,#FEE2E2 50%,#FEE2E2 100%);margin-bottom:4px;">' +
                f'<div style="position:absolute;top:50%;left:{_mlpos}%;transform:translate(-50%,-50%);width:16px;height:16px;border-radius:50%;background:{_mlcol};border:2px solid #fff;box-shadow:0 0 0 2px {_mlcol};"></div></div>' +
                '<div style="display:flex;justify-content:space-between;font-size:10px;color:#9CA3AF;margin-top:2px;">' +
                '<span>0% · Low</span><span>20% · Elevated</span><span>50% · High</span><span>100%</span></div>' +
                '</div>',
                unsafe_allow_html=True
            )
            # ── Executive summary ─────────────────────────────────────────
            if _mlpct < 20:
                _ml_exec = (
                    f"The XGBoost classifier assigns <b>{name}</b> a distress probability of <b>{_mlpct:.1f}%</b>. "
                    "The model detects no strong signals of near-term financial stress across the features it was trained on. "
                    "This is a low-risk reading — but it reflects the current data snapshot, not a permanent guarantee."
                )
            elif _mlpct < 50:
                _ml_exec = (
                    f"The XGBoost classifier assigns <b>{name}</b> a distress probability of <b>{_mlpct:.1f}%</b> — "
                    "in the elevated watch zone. The model has identified stress in one or more of its input features. "
                    "This is a decision-support signal: corroborate it with the traditional model results below before drawing conclusions."
                )
            else:
                _ml_exec = (
                    f"The XGBoost classifier assigns <b>{name}</b> a distress probability of <b>{_mlpct:.1f}%</b> — "
                    "above the classification threshold. The model has learned patterns across hundreds of historical distress cases "
                    "and is flagging meaningful combined stress here. Treat this as a high-priority signal requiring cross-model validation."
                )
            st.markdown(
                '<div style="background:rgba(86,82,216,0.04);border-left:3px solid #5652D8;'
                'border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin:1rem 0;'
                'font-size:13px;color:#374151;line-height:1.7;">' + _ml_exec + '</div>',
                unsafe_allow_html=True
            )

            # ── Feature inputs + all improvements ────────────────────────
            feats = result.get("features", {})
            missing_pct = result.get("missing_pct", 0.0)
            n_feats_available = sum(1 for v in feats.values() if v is not None) if feats else 0
            n_feats_total     = len(feats) if feats else 0

            if feats:
                # ── Model Feature Inputs (improved hierarchy) ────────────
                st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Model Feature Inputs</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                _feat_defs = [
                    ("z_score", "Altman Z-Score",      "{:.2f}",  lambda v: "safe"   if v > 2.99 else ("grey" if v > 1.81 else "stress")),
                    ("o_prob",  "Ohlson Prob",          "{:.1%}",  lambda v: "stress" if v > 0.50 else ("grey" if v > 0.30 else "safe")),
                    ("f_score", "Piotroski F-Score",    "{:.0f}/9",lambda v: "safe"   if v >= 6   else ("grey" if v >= 3   else "stress")),
                    ("m_score", "Beneish M-Score",      "{:.3f}",  lambda v: "stress" if v > -2.22 else "safe"),
                    ("roa",     "ROA",                  "{:.1%}",  lambda v: "safe"   if v > 0.05 else ("grey" if v > 0    else "stress")),
                    ("cfo_ta",  "CFO / Assets",         "{:.1%}",  lambda v: "safe"   if v > 0.05 else ("grey" if v > 0    else "stress")),
                    ("lev",     "Leverage (LT/TA)",     "{:.2f}",  lambda v: "safe"   if v < 0.35 else ("grey" if v < 0.6  else "stress")),
                    ("cr",      "Current Ratio",        "{:.2f}",  lambda v: "safe"   if v > 2.0  else ("grey" if v > 1.0  else "stress")),
                ]
                _zone_colors = {"safe": "#16A34A", "grey": "#D97706", "stress": "#EF4444"}
                _zone_bg     = {"safe": "#F0FDF4", "grey": "#FFFBEB", "stress": "#FFF5F5"}
                _zone_border = {"safe": "rgba(22,163,74,0.2)", "grey": "rgba(217,119,6,0.25)", "stress": "rgba(239,68,68,0.2)"}
                feat_cols2 = st.columns(4)
                for i, (key, label, fmt_str, zone_fn) in enumerate(_feat_defs):
                    val = feats.get(key)
                    with feat_cols2[i % 4]:
                        val_disp = (fmt_str.format(val) if val is not None else "N/A")
                        zone     = zone_fn(val) if val is not None else "grey"
                        zcol     = _zone_colors[zone]
                        zbg      = _zone_bg[zone]
                        zborder  = _zone_border[zone]
                        st.markdown(
                            f'<div style="background:{zbg};border:1.5px solid {zborder};border-radius:10px;'
                            f'padding:12px;margin-bottom:8px;text-align:center;">'
                            f'<div style="font-size:10px;color:#9CA3AF;margin-bottom:4px;">{label}</div>'
                            f'<div style="font-family:DM Mono,monospace;font-size:18px;font-weight:700;color:#1E1B4B;">{val_disp}</div>'
                            f'<div style="font-size:9px;color:{zcol};font-weight:600;margin-top:3px;text-transform:uppercase;letter-spacing:.5px;">{zone}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # ── What Is Driving This Prediction? ─────────────────────
                st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">What Is Driving This Prediction?</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                # Proxy contribution: signed signal per feature (-1 = very safe, +1 = very risky)
                def _clip(v, lo, hi):
                    return max(lo, min(hi, v))
                _signals = []
                if feats.get("z_score")   is not None: _signals.append(("Altman Z-Score",      _clip((2.4 - feats["z_score"]) / 2.4, -1, 1),       "Lower Z-Score signals more distress risk."))
                if feats.get("o_prob")    is not None: _signals.append(("Ohlson Probability",   _clip((feats["o_prob"] - 0.25) / 0.5, -1, 1),        "Higher Ohlson probability signals distress."))
                if feats.get("f_score")   is not None: _signals.append(("Piotroski F-Score",    _clip((4.5 - feats["f_score"]) / 4.5, -1, 1),        "Lower F-Score signals weaker fundamentals."))
                if feats.get("m_score")   is not None: _signals.append(("Beneish M-Score",      _clip((feats["m_score"] + 2.22) / 2.0, -1, 1),       "M-Score above -2.22 signals earnings risk."))
                if feats.get("roa")       is not None: _signals.append(("Return on Assets",     _clip(-feats["roa"] / 0.08, -1, 1),                  "Negative or low ROA elevates distress risk."))
                if feats.get("cfo_ta")    is not None: _signals.append(("Cash Flow / Assets",   _clip(-feats["cfo_ta"] / 0.08, -1, 1),               "Low or negative operating cash flow is a stress signal."))
                if feats.get("lev")       is not None: _signals.append(("Long-term Leverage",   _clip((feats["lev"] - 0.4) / 0.4, -1, 1),            "Higher leverage raises distress probability."))
                if feats.get("cr")        is not None: _signals.append(("Current Ratio",        _clip((1.5 - feats["cr"]) / 1.5, -1, 1),             "Lower current ratio signals liquidity stress."))
                if _signals:
                    _signals_sorted = sorted(_signals, key=lambda x: x[1], reverse=True)
                    _top_risk  = _signals_sorted[0]
                    _top_safe  = _signals_sorted[-1]
                    st.markdown(
                        '<div style="display:grid;gap:8px;margin-top:4px;">' +
                        '<div style="background:#FFF9F9;border:1.5px solid rgba(239,68,68,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">' +
                        f'<span style="color:#EF4444;font-weight:700;">&#8593; Most risk-elevating: </span><b>{_top_risk[0]}</b> — {_top_risk[2]}' +
                        '</div>' +
                        '<div style="background:#F0FDF4;border:1.5px solid rgba(22,163,74,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">' +
                        f'<span style="color:#16A34A;font-weight:700;">&#10003; Most risk-reducing: </span><b>{_top_safe[0]}</b> — {_top_safe[2]}' +
                        '</div>' +
                        '</div>',
                        unsafe_allow_html=True
                    )

                # ── Traditional Model Comparison ──────────────────────────
                st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Traditional Model Signals</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:12px;color:#6B7280;margin-bottom:0.8rem;">' +
                    'Comparing the ML result against the four traditional models that feed into its features. Disagreement between models is normal — use it as context.</div>',
                    unsafe_allow_html=True
                )
                def _trad_zone(model, val):
                    if val is None: return "N/A", "#9CA3AF", "rgba(156,163,175,0.1)"
                    if model == "z":
                        lbl = "Safe Zone" if val > 2.99 else ("Grey Zone" if val > 1.81 else "Distress Zone")
                        col = "#16A34A" if val > 2.99 else ("#D97706" if val > 1.81 else "#EF4444")
                        bg  = "#F0FDF4" if val > 2.99 else ("#FFFBEB" if val > 1.81 else "#FFF5F5")
                    elif model == "o":
                        lbl = "High Risk" if val >= 0.5 else ("Watch" if val >= 0.3 else "Low Risk")
                        col = "#EF4444" if val >= 0.5 else ("#D97706" if val >= 0.3 else "#16A34A")
                        bg  = "#FFF5F5" if val >= 0.5 else ("#FFFBEB" if val >= 0.3 else "#F0FDF4")
                    elif model == "f":
                        lbl = "Strong" if val >= 6 else ("Neutral" if val >= 3 else "Weak")
                        col = "#16A34A" if val >= 6 else ("#D97706" if val >= 3 else "#EF4444")
                        bg  = "#F0FDF4" if val >= 6 else ("#FFFBEB" if val >= 3 else "#FFF5F5")
                    elif model == "m":
                        lbl = "Manipulation Risk" if val > -2.22 else "Likely Clean"
                        col = "#EF4444" if val > -2.22 else "#16A34A"
                        bg  = "#FFF5F5" if val > -2.22 else "#F0FDF4"
                    else:
                        lbl, col, bg = "N/A", "#9CA3AF", "#F9F8FF"
                    return lbl, col, bg
                _z_val  = feats.get("z_score")
                _o_val  = feats.get("o_prob")
                _f_val  = feats.get("f_score")
                _m_val  = feats.get("m_score")
                _trad_items = [
                    ("ML Score",         f"{_mlpct:.1f}%",    _mllbl,   _mlcol,   "rgba(86,82,216,0.06)"),
                    ("Altman Z-Score",   f"{_z_val:.2f}" if _z_val is not None else "N/A",  *(_trad_zone("z", _z_val)[:2] + (_trad_zone("z", _z_val)[2],))),
                    ("Ohlson O-Score",   f"{_o_val:.1%}" if _o_val is not None else "N/A",  *(_trad_zone("o", _o_val)[:2] + (_trad_zone("o", _o_val)[2],))),
                    ("Piotroski F",      f"{_f_val:.0f}/9" if _f_val is not None else "N/A",*(_trad_zone("f", _f_val)[:2] + (_trad_zone("f", _f_val)[2],))),
                    ("Beneish M",        f"{_m_val:.3f}" if _m_val is not None else "N/A",  *(_trad_zone("m", _m_val)[:2] + (_trad_zone("m", _m_val)[2],))),
                ]
                _trad_html = '<div style="display:grid;gap:6px;margin-top:4px;">'
                for _tm, _tv, _tl, _tc, _tbg in _trad_items:
                    _trad_html += (
                        f'<div style="background:{_tbg};border-radius:8px;padding:0.7rem 1rem;' +
                        f'display:flex;align-items:center;justify-content:space-between;font-size:12px;">' +
                        f'<span style="color:#6B7280;font-weight:500;">{_tm}</span>' +
                        f'<div style="display:flex;align-items:center;gap:10px;">' +
                        f'<span style="font-family:DM Mono,monospace;font-size:13px;font-weight:700;color:#1E1B4B;">{_tv}</span>' +
                        f'<span style="font-size:10px;font-weight:700;color:{_tc};text-transform:uppercase;letter-spacing:.5px;">{_tl}</span>' +
                        f'</div></div>'
                    )
                _trad_html += '</div>'
                st.markdown(_trad_html, unsafe_allow_html=True)

            # ── Interpretation ────────────────────────────────────────────
            if _mlpct < 20:
                _ml_interp = (
                    f"At <b>{_mlpct:.1f}%</b>, the XGBoost model classifies this company as <b>low distress risk</b>. "
                    "The model was trained on historical patterns from WRDS/Compustat data spanning 1990–2025 — it is recognising "
                    "a financial profile that does not resemble companies that went on to experience distress. "
                    "A low ML score does not mean zero future risk. It means the current inputs do not match known distress patterns."
                )
            elif _mlpct < 50:
                _ml_interp = (
                    f"At <b>{_mlpct:.1f}%</b>, the XGBoost model places this company in a <b>moderate elevated-risk zone</b>. "
                    "The model is detecting feature combinations associated with financial stress in its training data, "
                    "but below the classification threshold. Treat this as a decision-support signal — check the traditional model "
                    "signals above and monitor the most risk-elevating features over the next reporting period."
                )
            else:
                _ml_interp = (
                    f"At <b>{_mlpct:.1f}%</b>, the XGBoost model classifies this company as <b>high distress risk</b>. "
                    "The model is recognising a pattern that historically preceded financial distress in its training data. "
                    "ML-based predictions should always be treated as decision-support tools, not definitive forecasts — "
                    "cross-check with the traditional model signals above and consider whether any features may be imputed or lagged."
                )
            # Confidence note
            if missing_pct > 0.25:
                _conf_note = f"&#9432; Reliability note: {missing_pct:.0%} of model features were imputed from training-set medians. Prediction confidence is reduced — interpret with caution."
                _conf_bg   = "rgba(245,158,11,0.06)"
                _conf_bdr  = "rgba(245,158,11,0.25)"
            elif missing_pct > 0.1:
                _conf_note = f"&#9432; Note: {missing_pct:.0%} of features used median imputation. The core signals are available; reliability is adequate."
                _conf_bg   = "rgba(86,82,216,0.04)"
                _conf_bdr  = "rgba(86,82,216,0.15)"
            else:
                _conf_note = "&#10003; All key features were available directly from financial data — prediction reliability is high."
                _conf_bg   = "rgba(22,163,74,0.04)"
                _conf_bdr  = "rgba(22,163,74,0.2)"
            st.markdown(
                f'<div class="fu5 interp-box" style="margin-top:1.2rem;">{_ml_interp}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="background:{_conf_bg};border:1px solid {_conf_bdr};border-radius:8px;'
                f'padding:0.7rem 1rem;font-size:11px;color:#6B7280;margin-top:8px;line-height:1.6;">{_conf_note}</div>',
                unsafe_allow_html=True
            )

            # ── Investor Snapshot ─────────────────────────────────────────
            render_investor_snapshot(snap_ml)

            # ── Feature Signal Chart (after investor snapshot) ────────────
            if feats and PLOTLY_AVAILABLE and _signals:
                st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Feature Risk Signals</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:12px;color:#6B7280;margin-bottom:0.8rem;">' +
                    'Each bar shows whether a feature is pushing the ML prediction toward risk (red) or safety (green). ' +
                    'Values are normalised proxies — not exact SHAP values.</div>',
                    unsafe_allow_html=True
                )
                _chart_signals = sorted(_signals, key=lambda x: x[1], reverse=True)
                _feat_names = [s[0] for s in _chart_signals]
                _feat_vals  = [s[1] for s in _chart_signals]
                _bar_colors = ["#EF4444" if v > 0 else "#16A34A" for v in _feat_vals]
                fig_ml = pgo.Figure()
                fig_ml.add_trace(pgo.Bar(
                    x=_feat_vals,
                    y=_feat_names,
                    orientation="h",
                    marker_color=_bar_colors,
                    marker_line_width=0,
                    opacity=0.85,
                ))
                fig_ml.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.15)")
                fig_ml.update_layout(
                    height=280,
                    margin=dict(l=8, r=8, t=12, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=11, color="#6B7280"),
                    xaxis=dict(
                        range=[-1.1, 1.1],
                        showgrid=True,
                        gridcolor="rgba(0,0,0,0.05)",
                        zeroline=False,
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=["−1<br>Very Safe", "−0.5", "0<br>Neutral", "+0.5", "+1<br>Very Risky"],
                        tickfont=dict(size=9),
                    ),
                    yaxis=dict(showgrid=False, tickfont=dict(size=11)),
                    showlegend=False,
                    bargap=0.3,
                )
                st.plotly_chart(fig_ml, use_container_width=True)

    elif btn_ml:
        st.markdown('<div class="alert-warn">Please enter a ticker symbol.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Zmijewski Score
# ════════════════════════════════════════════════════════════════════════════
def page_zscore2():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(86,82,216,0.1);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#1E1B4B;">
            Zmijewski <span style="color:#5652D8;">Score</span>
          </div>
          <div style="font-size:12px;color:#9CA3AF;margin-top:3px;">Probit model of financial distress (1984)</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#4440C9;
                    background:#F0EFFE;border:1.5px solid #E8E5F8;
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">v1.0 · Probit</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;font-weight:500;color:#6B7280;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    zz1, zz2 = st.columns([4, 1])
    with zz1:
        t_z2 = st.text_input("tz2", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with zz2:
        btn_z2 = st.button("Analyze →", key="btn_z2")

    if btn_z2 and t_z2.strip():
        ticker = t_z2.strip().upper()
        with st.spinner("Fetching data..."):
            zd = compute_zscore2(ticker)
            snap_z2 = compute_investor_snapshot(ticker)

        if not zd["ok"]:
            st.markdown('<div style="background:rgba(239,68,68,0.06);border:1.5px solid rgba(239,68,68,0.2);border-radius:10px;padding:1rem 1.2rem;color:#DC2626;font-size:13px;margin-top:1rem;">Required financial fields missing — Zmijewski Score could not be calculated.</div>', unsafe_allow_html=True)
        else:
            prob = zd["prob"]
            x_val = zd["x"]
            pct  = prob * 100

            initials_z2 = ticker[:2]
            if zd.get("logo"):
                logo_z2 = f'<div style="width:38px;height:38px;border-radius:8px;background:#F4F3FF;padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;"><img src="{zd["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>'
            else:
                logo_z2 = f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(86,82,216,0.1);border:1.5px solid rgba(86,82,216,0.25);display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#5652D8;flex-shrink:0;">{initials_z2}</div>'

            st.markdown(f"""
            <div class="fu2" style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;
                        padding:1.2rem 1.5rem;display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">
              <div style="display:flex;align-items:center;gap:12px;">
                {logo_z2}
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="background:rgba(86,82,216,0.08);border:1.5px solid #E8E5F8;
                              border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;
                              font-size:13px;font-weight:500;color:#5652D8;letter-spacing:1px;">{ticker}</div>
                  <div>
                    <div style="font-size:17px;font-weight:700;color:#1E1B4B;">{zd['name']}</div>
                    <div style="font-size:12px;color:#6B7280;margin-top:1px;">{zd['sector']} · {zd['country']}</div>
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px;color:#CBD5E1;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>
                <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:#1E1B4B;margin-top:2px;">{fmt(zd['mc'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            trend_z2 = compute_zscore2_trend(ticker)

            # ── Score card (keep existing gauge + risk panel layout) ──────
            st.markdown('<div class="fu3 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Distress Probability</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            zp1, zp2 = st.columns([1, 1])
            with zp1:
                if PLOTLY_AVAILABLE:
                    st.plotly_chart(render_prob_gauge(prob, "Zmijewski Distress Probability"), use_container_width=True)
            with zp2:
                risk_color = "#EF4444" if prob >= 0.5 else ("#F59E0B" if prob >= 0.3 else "#16A34A")
                risk_label = "High Risk" if prob >= 0.5 else ("Moderate Risk" if prob >= 0.3 else "Low Risk")
                st.markdown(
                    f'<div style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;'
                    f'padding:1.8rem;text-align:center;height:100%;display:flex;flex-direction:column;'
                    f'align-items:center;justify-content:center;gap:10px;">'
                    f'<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9CA3AF;">Distress Probability</div>'
                    f'<div style="font-family:DM Mono,monospace;font-size:52px;font-weight:700;color:{risk_color};line-height:1;">{pct:.1f}%</div>'
                    f'<div style="font-size:13px;font-weight:600;color:{risk_color};">{risk_label}</div>'
                    f'<div style="font-size:11px;color:#9CA3AF;">X = {x_val:.4f}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── Executive summary ─────────────────────────────────────────
            if pct < 20:
                _z2_exec = (
                    f"The Zmijewski model assigns <b>{zd['name']}</b> a distress probability of <b>{pct:.1f}%</b>. "
                    "All three model inputs — profitability, leverage, and liquidity — are reading within healthy ranges. "
                    "This is a low-signal result, meaning no acute financial stress is detected at this time."
                )
            elif pct < 50:
                _z2_exec = (
                    f"The Zmijewski model assigns <b>{zd['name']}</b> a distress probability of <b>{pct:.1f}%</b> — "
                    "in the moderate watch zone. At least one of the three inputs is showing weakness: "
                    "either profitability is thin, leverage is elevated, or liquidity is tighter than typical. "
                    "This is worth monitoring over the next reporting period."
                )
            else:
                _z2_exec = (
                    f"The Zmijewski model assigns <b>{zd['name']}</b> a distress probability of <b>{pct:.1f}%</b> — "
                    "above the 50% high-risk threshold. The probit model detects meaningful stress across its three "
                    "core dimensions: profitability, leverage, and liquidity. This warrants immediate attention."
                )
            st.markdown(
                '<div style="background:rgba(86,82,216,0.04);border-left:3px solid #5652D8;'
                'border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin:1rem 0;'
                'font-size:13px;color:#374151;line-height:1.7;">' + _z2_exec + '</div>',
                unsafe_allow_html=True
            )

            # ── Model Inputs (improved hierarchy) ────────────────────────
            st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Model Inputs</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _roa  = zd.get("roa")
            _lev  = zd.get("lev")
            _liq  = zd.get("liq")
            _coef = [("ROA", "Net Income / Assets", _roa, "−4.513", "Higher ROA lowers distress risk."),
                     ("Leverage", "Total Liab / Assets", _lev, "+5.679", "Higher leverage raises distress risk."),
                     ("Liquidity", "Current Assets / CL",  _liq, "+0.004", "Higher liquidity mildly reduces risk.")]
            z2cols = st.columns(3)
            for col, (nm, formula, val, wt, note) in zip(z2cols, _coef):
                val_disp = f"{val:.3f}" if val is not None else "N/A"
                col.markdown(
                    f'<div style="background:#F9F8FF;border:1.5px solid #EAE8F8;border-radius:10px;padding:1rem;text-align:center;">'
                    f'<div style="font-size:11px;font-weight:700;color:#5652D8;margin-bottom:2px;">{nm}</div>'
                    f'<div style="font-size:10px;color:#9CA3AF;margin-bottom:8px;line-height:1.3;">{formula}</div>'
                    f'<div style="font-family:DM Mono,monospace;font-size:24px;font-weight:700;color:#1E1B4B;line-height:1;">{val_disp}</div>'
                    f'<div style="font-size:10px;color:#CBD5E1;margin-top:4px;font-family:DM Mono,monospace;">{wt}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── What Is Driving This Score? ───────────────────────────────
            st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">What Is Driving This Score?</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            # Weighted contributions (coefficient × value, sign indicates direction)
            _z2_contribs = []
            if _roa  is not None: _z2_contribs.append(("ROA (profitability)",   -4.513 * _roa,  "Negative ROA dramatically raises distress probability."))
            if _lev  is not None: _z2_contribs.append(("Leverage (TL/TA)",      +5.679 * _lev,  "High leverage is the strongest risk-elevating force in this model."))
            if _liq  is not None: _z2_contribs.append(("Liquidity (CA/CL)",     +0.004 * _liq,  "Liquidity has a small but positive effect on the probit score."))
            if _z2_contribs:
                _z2_sorted  = sorted(_z2_contribs, key=lambda d: d[1], reverse=True)
                _z2_top     = _z2_sorted[0]
                _z2_best    = _z2_sorted[-1]
                st.markdown(
                    '<div style="display:grid;gap:8px;margin-top:4px;">' +
                    '<div style="background:#FFF9F9;border:1.5px solid rgba(239,68,68,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">' +
                    f'<span style="color:#EF4444;font-weight:700;">&#8593; Most risk-elevating: </span><b>{_z2_top[0]}</b> — {_z2_top[2]}' +
                    '</div>' +
                    '<div style="background:#F0FDF4;border:1.5px solid rgba(22,163,74,0.15);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">' +
                    f'<span style="color:#16A34A;font-weight:700;">&#10003; Least concerning: </span><b>{_z2_best[0]}</b> — {_z2_best[2]}' +
                    '</div>' +
                    '<div style="background:#FFFBEB;border:1.5px solid rgba(245,158,11,0.2);border-radius:8px;padding:0.8rem 1rem;font-size:12px;color:#374151;">' +
                    '<span style="color:#D97706;font-weight:700;">&#9432; Note: </span>The Zmijewski model uses only three inputs. ' +
                    'A good score here does not guarantee financial health — it reflects low current stress on these three dimensions only.' +
                    '</div>' +
                    '</div>',
                    unsafe_allow_html=True
                )

            # ── Interpretation ────────────────────────────────────────────
            if pct < 20:
                _z2_interp = (
                    f"At <b>{pct:.1f}%</b>, the Zmijewski probit model places this company firmly in the <b>low-risk</b> range. "
                    "The model was developed using ROA, financial leverage, and current liquidity as its only three inputs — "
                    "all of which are reading positively here. A low score does not eliminate all future risk, but it signals "
                    "no current distress under this framework."
                )
            elif pct < 50:
                _z2_interp = (
                    f"At <b>{pct:.1f}%</b>, the Zmijewski model places this company in a <b>moderate watch zone</b>. "
                    "The model is driven by ROA, leverage, and liquidity — and at least one of these is under pressure. "
                    "Zmijewski (1984) found firms in this range have elevated but not certain default risk over a one-to-two year horizon. "
                    "Monitoring the weakest driver closely is advisable."
                )
            else:
                _z2_interp = (
                    f"At <b>{pct:.1f}%</b>, the Zmijewski model places this company in the <b>high-risk</b> category. "
                    "The probit model identifies combined stress across profitability, leverage, and liquidity — "
                    "the three variables that Zmijewski (1984) found most predictive of financial distress. "
                    "Companies above the 50% threshold showed materially higher bankruptcy rates in the original study."
                )
            st.markdown(f'<div class="fu5 interp-box" style="margin-top:1.2rem;">{_z2_interp}</div>', unsafe_allow_html=True)

            # ── Investor Snapshot ─────────────────────────────────────────
            render_investor_snapshot(snap_z2)

            # ── ROA & Leverage Trend chart (after investor snapshot) ──────
            if trend_z2 and PLOTLY_AVAILABLE and len(trend_z2["years"]) >= 2:
                st.markdown('<div class="sec-hdr" style="margin-top:1.5rem;"><span class="sec-lbl">Key Driver Trends</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:12px;color:#6B7280;margin-bottom:0.8rem;">' +
                    'ROA and leverage are the two dominant Zmijewski inputs. Falling profitability or rising leverage ' +
                    'directly increases distress probability.</div>',
                    unsafe_allow_html=True
                )
                _yrs_z2 = trend_z2["years"]
                fig_z2 = pgo.Figure()
                fig_z2.add_trace(pgo.Bar(
                    name="Leverage (TL/TA)",
                    x=_yrs_z2,
                    y=trend_z2["leverage"],
                    marker_color="#5652D8",
                    opacity=0.85,
                    yaxis="y1",
                ))
                fig_z2.add_trace(pgo.Scatter(
                    name="ROA (NI/TA)",
                    x=_yrs_z2,
                    y=trend_z2["roa"],
                    mode="lines+markers",
                    line=dict(color="#F59E0B", width=2.5),
                    marker=dict(size=7, color="#F59E0B"),
                    yaxis="y2",
                ))
                fig_z2.update_layout(
                    height=260,
                    margin=dict(l=8, r=8, t=20, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=11, color="#6B7280"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                    xaxis=dict(type="category", categoryorder="array", categoryarray=_yrs_z2, showgrid=False, tickfont=dict(size=11)),
                    yaxis=dict(title="TL/TA", showgrid=True, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10), zeroline=True, zerolinecolor="rgba(0,0,0,0.1)"),
                    yaxis2=dict(title="ROA", overlaying="y", side="right", showgrid=False, tickfont=dict(size=10), zeroline=True, zerolinecolor="rgba(245,158,11,0.3)"),
                    bargap=0.35,
                )
                st.plotly_chart(fig_z2, use_container_width=True)

    elif btn_z2:
        st.markdown('<div class="alert-warn">Please enter a ticker symbol.</div>', unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════════
# PAGE: Sentiment Analysis
# ════════════════════════════════════════════════════════════════════════════
def page_sentiment():
    render_back_button()
    st.markdown("""
    <div class="fu1" style="margin-bottom:1.5rem;padding-bottom:1.2rem;
                            border-bottom:0.5px solid rgba(249,115,22,0.15);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;color:#1E1B4B;">
            Sentiment <span style="color:#F97316;">Analysis</span>
          </div>
          <div style="font-size:12px;color:#9CA3AF;margin-top:3px;">Supporting signal · Recent news headline tone</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#EA580C;
                    background:rgba(249,115,22,0.08);border:1.5px solid rgba(249,115,22,0.25);
                    padding:4px 10px;border-radius:4px;letter-spacing:1px;">BETA</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Supporting signal notice
    st.markdown(
        '<div style="margin-bottom:1.2rem;padding:10px 14px;background:rgba(249,115,22,0.06);' +
        'border:1px solid rgba(249,115,22,0.2);border-radius:8px;display:flex;align-items:flex-start;gap:10px;">' +
        '<div style="font-size:13px;color:#EA580C;flex-shrink:0;margin-top:1px;">⚠️</div>' +
        '<div style="font-size:12px;color:#6B7280;line-height:1.6;">' +
        '<b style="color:#374151;">Supporting signal only.</b> Sentiment Analysis scans recent news headlines ' +
        'to assess the prevailing tone around a company. It is <b>not</b> a financial distress model and ' +
        'should not replace quantitative analysis. Use it alongside the core models for additional context.' +
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div style="font-size:11px;font-weight:500;color:#6B7280;letter-spacing:1.5px;text-transform:uppercase;margin:1rem 0 6px;">Company Ticker</div>', unsafe_allow_html=True)
    _s1, _s2 = st.columns([4, 1])
    with _s1:
        t_s = st.text_input("t_s", placeholder="e.g. AAPL, TSLA, MSFT", label_visibility="collapsed")
    with _s2:
        btn_s = st.button("Analyze →", key="btn_s")

    if btn_s and t_s.strip():
        ticker_s = t_s.strip().upper()
        with st.spinner("Fetching recent headlines…"):
            sd = compute_sentiment(ticker_s)

        if not sd["ok"]:
            st.markdown(
                '<div style="background:rgba(239,68,68,0.06);border:1.5px solid rgba(239,68,68,0.2);' +
                f'border-radius:10px;padding:1rem 1.2rem;color:#DC2626;font-size:13px;margin-top:1rem;">' +
                f'⚠ {sd.get("reason","Could not retrieve headline data.")}</div>',
                unsafe_allow_html=True
            )
        else:
            # Company header
            _s_init = ticker_s[:2]
            if sd["logo"]:
                _s_logo = ('<div style="width:38px;height:38px;border-radius:8px;background:#FFF7ED;' +
                           'padding:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;">' +
                           f'<img src="{sd["logo"]}" style="width:30px;height:30px;object-fit:contain;"></div>')
            else:
                _s_logo = (f'<div style="width:38px;height:38px;border-radius:8px;background:rgba(249,115,22,0.1);' +
                           'border:1.5px solid rgba(249,115,22,0.25);display:flex;align-items:center;' +
                           f'justify-content:center;font-family:monospace;font-size:13px;font-weight:500;color:#F97316;flex-shrink:0;">{_s_init}</div>')
            st.markdown(
                '<div class="fu2" style="background:#FFFFFF;border:1.5px solid #E8E5F8;border-radius:14px;' +
                'padding:1.2rem 1.5rem;display:flex;align-items:center;justify-content:space-between;' +
                'flex-wrap:wrap;gap:12px;margin:1rem 0 1.5rem;">' +
                f'<div style="display:flex;align-items:center;gap:12px;">{_s_logo}' +
                '<div style="display:flex;align-items:center;gap:10px;">' +
                '<div style="background:rgba(249,115,22,0.08);border:1.5px solid rgba(249,115,22,0.2);' +
                f'border-radius:6px;padding:4px 10px;font-family:DM Mono,monospace;font-size:13px;font-weight:500;color:#F97316;letter-spacing:1px;">{ticker_s}</div>' +
                '<div>' +
                f'<div style="font-size:17px;font-weight:700;color:#1E1B4B;">{sd["name"]}</div>' +
                f'<div style="font-size:12px;color:#6B7280;margin-top:1px;">{sd["sector"]} · {sd["country"]}</div>' +
                '</div></div></div>' +
                ('<div style="text-align:right;">' +
                 '<div style="font-size:10px;color:#CBD5E1;letter-spacing:1px;text-transform:uppercase;">Market Cap</div>' +
                 f'<div style="font-family:DM Mono,monospace;font-size:20px;font-weight:500;color:#1E1B4B;">{fmt(sd["mc"])}</div>' +
                 '</div>' if sd["mc"] else '') +
                '</div>',
                unsafe_allow_html=True
            )

            # ── Sentiment result card ──────────────────────────────────────
            st.markdown('<div class="fu3 sec-hdr"><span class="sec-lbl">Sentiment Signal</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            _s_pct = round((sd["avg_score"] + 0.20) / 0.40 * 100, 1)
            _s_pct = max(2.0, min(98.0, _s_pct))
            _s_pct_bar = max(2.0, min(97.0, _s_pct))
            st.markdown(
                f'<div style="background:{sd["bg"]};border:1.5px solid {sd["color"]}33;' +
                'border-radius:14px;padding:1.6rem 2rem;">' +
                '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.4rem;">' +
                f'<div style="font-family:DM Mono,monospace;font-size:52px;font-weight:700;color:{sd["color"]};line-height:1;">{sd["label"]}</div>' +
                '<div style="text-align:right;">' +
                f'<div style="font-size:11px;color:#9CA3AF;margin-bottom:4px;letter-spacing:.5px;text-transform:uppercase;">Based on {sd["count"]} headlines</div>' +
                f'<div style="font-size:13px;color:#6B7280;">{sd["pos_n"]} positive · {sd["neu_n"]} neutral · {sd["neg_n"]} negative</div>' +
                '</div></div>' +
                '<div style="position:relative;height:10px;border-radius:999px;background:linear-gradient(to right,#FEE2E2 0%,#F3F4F6 50%,#DCFCE7 100%);margin-bottom:4px;">' +
                f'<div style="position:absolute;top:50%;left:{_s_pct_bar}%;transform:translate(-50%,-50%);width:16px;height:16px;border-radius:50%;background:{sd["color"]};border:2px solid #fff;box-shadow:0 0 0 2px {sd["color"]};"></div></div>' +
                '<div style="display:flex;justify-content:space-between;font-size:10px;color:#9CA3AF;margin-top:2px;">' +
                '<span>Negative</span><span>Neutral</span><span>Positive</span></div>' +
                '</div>',
                unsafe_allow_html=True
            )

            # ── Executive Summary ──────────────────────────────────────────
            _s_name = sd["name"].split()[0] if sd.get("name") else ticker_s
            if sd["label"] == "Positive":
                _s_exec = (f"Recent news coverage of <b>{_s_name}</b> is trending <b style='color:#16A34A;'>positive</b>. "
                           f"Across {sd['count']} recent headlines, the tone is broadly constructive, with more "
                           f"favourable references ({sd['pos_n']}) than negative ones ({sd['neg_n']}). "
                           f"This does not guarantee financial health, but suggests the current news cycle is not a headwind.")
            elif sd["label"] == "Negative":
                _s_exec = (f"Recent news coverage of <b>{_s_name}</b> is trending <b style='color:#EF4444;'>negative</b>. "
                           f"Across {sd['count']} recent headlines, the tone is broadly cautious, with more "
                           f"unfavourable references ({sd['neg_n']}) than positive ones ({sd['pos_n']}). "
                           f"This is a supporting signal worth noting, particularly when core distress models also show elevated risk.")
            else:
                _s_exec = (f"Recent news coverage of <b>{_s_name}</b> is largely <b style='color:#6B7280;'>neutral</b>. "
                           f"Across {sd['count']} recent headlines, positive and negative tones are roughly balanced. "
                           f"No clear directional sentiment signal is present in the current news cycle.")
            st.markdown(
                '<div style="margin-top:1rem;padding:14px 18px;background:#F9F8FF;border-radius:10px;' +
                'border:1px solid #EAE8F8;font-size:13px;color:#374151;line-height:1.7;">' +
                '<span style="font-size:11px;font-weight:700;color:#F97316;letter-spacing:.5px;' +
                'text-transform:uppercase;display:block;margin-bottom:8px;">Executive Summary</span>' +
                f'{_s_exec}</div>',
                unsafe_allow_html=True
            )

            # ── Headline Breakdown ─────────────────────────────────────────
            st.markdown('<div class="fu4 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Recent Headlines</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
            for item in sd["items"][:8]:
                _age_str = f'{item["age_days"]:.0f}d ago' if item["age_days"] < 30 else ""
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;padding:8px 12px;' +
                    f'background:#FFFFFF;border:1px solid #EAE8F8;border-radius:8px;margin-bottom:6px;">' +
                    f'<span style="font-size:14px;flex-shrink:0;margin-top:1px;">{item["icon"]}</span>' +
                    f'<div style="flex:1;">' +
                    f'<div style="font-size:12px;color:#1E1B4B;font-weight:500;line-height:1.4;">{item["title"]}</div>' +
                    f'<div style="font-size:10px;color:#9CA3AF;margin-top:3px;">{item["publisher"]}' +
                    (f' · {_age_str}' if _age_str else '') +
                    f' · <span style="color:{item["color"]};font-weight:600;">{item["label"]}</span></div></div></div>',
                    unsafe_allow_html=True
                )

            # ── Distribution chart ─────────────────────────────────────────
            if PLOTLY_AVAILABLE and sd["count"] > 0:
                st.markdown('<div class="fu5 sec-hdr" style="margin-top:1.2rem;"><span class="sec-lbl">Sentiment Distribution</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                _fig_s = pgo.Figure()
                _fig_s.add_trace(pgo.Bar(
                    x=[sd["pos_n"]], y=["Headlines"],
                    name="Positive", orientation="h",
                    marker_color="#BBF7D0", marker_line=dict(color="#16A34A", width=1.5),
                    text=[f'{sd["pos_n"]} Positive'] if sd["pos_n"] else [""],
                    textposition="inside", textfont=dict(color="#15803D", size=11),
                ))
                _fig_s.add_trace(pgo.Bar(
                    x=[sd["neu_n"]], y=["Headlines"],
                    name="Neutral", orientation="h",
                    marker_color="#E5E7EB", marker_line=dict(color="#9CA3AF", width=1),
                    text=[f'{sd["neu_n"]} Neutral'] if sd["neu_n"] else [""],
                    textposition="inside", textfont=dict(color="#374151", size=11),
                ))
                _fig_s.add_trace(pgo.Bar(
                    x=[sd["neg_n"]], y=["Headlines"],
                    name="Negative", orientation="h",
                    marker_color="#FECACA", marker_line=dict(color="#EF4444", width=1.5),
                    text=[f'{sd["neg_n"]} Negative'] if sd["neg_n"] else [""],
                    textposition="inside", textfont=dict(color="#DC2626", size=11),
                ))
                _fig_s.update_layout(
                    barmode="stack", height=100,
                    margin=dict(l=0, r=0, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", size=11),
                    showlegend=False,
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(showgrid=False, showticklabels=False),
                )
                st.plotly_chart(_fig_s, use_container_width=True)

            # ── Interpretation ─────────────────────────────────────────────
            if sd["label"] == "Positive":
                _s_interp = (f"The news cycle around <b>{_s_name}</b> carries a broadly constructive tone. "
                             f"Positive sentiment in the press can reflect earnings beats, product launches, "
                             f"analyst upgrades, or broader sector tailwinds. However, headline sentiment "
                             f"can shift quickly and is not a substitute for fundamental financial analysis.")
            elif sd["label"] == "Negative":
                _s_interp = (f"The news cycle around <b>{_s_name}</b> is cautionary. "
                             f"Negative coverage can reflect earnings misses, regulatory issues, management changes, "
                             f"or macro headwinds. When negative sentiment coincides with weak distress model scores, "
                             f"the combined signal warrants closer attention.")
            else:
                _s_interp = (f"The news cycle around <b>{_s_name}</b> does not show a clear directional bias. "
                             f"Mixed or neutral coverage typically means the company is not in the spotlight for "
                             f"either sharply positive or negative reasons. Treat this as a non-signal and focus "
                             f"on the quantitative models for risk assessment.")
            st.markdown(f'<div class="fu6 interp-box">{_s_interp}</div>', unsafe_allow_html=True)

            # ── Disclaimer ────────────────────────────────────────────────
            st.markdown(
                '<div style="margin-top:1.5rem;padding:10px 14px;background:#FFF7ED;border-radius:8px;' +
                'border:1px solid rgba(249,115,22,0.2);display:flex;align-items:flex-start;gap:10px;">' +
                '<div style="font-size:13px;color:#F97316;flex-shrink:0;">⚠️</div>' +
                '<div style="font-size:11px;color:#9CA3AF;line-height:1.6;">' +
                '<b style="color:#6B7280;">Sentiment disclaimer:</b> This signal is derived from keyword matching ' +
                'on recent news headlines via public data sources. It is a supplementary, qualitative indicator only. ' +
                'Sentiment can be volatile, lag events, or reflect noise rather than fundamentals. ' +
                'Do not use sentiment as a standalone basis for any investment or credit decision.' +
                '</div></div>',
                unsafe_allow_html=True
            )

    elif btn_s:
        st.markdown('<div style="background:rgba(245,158,11,0.06);border:1.5px solid rgba(245,158,11,0.2);border-radius:10px;padding:1rem 1.2rem;color:#D97706;font-size:13px;margin-top:1rem;">Please enter a ticker symbol.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════
PAGE_MAP = {
    "home":       page_home,
    "zscore":     page_zscore,
    "oscore":     page_oscore,
    "zscore2":    page_zscore2,
    "fscore":     page_fscore,
    "mscore":     page_mscore,
    "mlscore":    page_mlscore,
    "comparison": page_comparison,
    "sentiment":   page_sentiment,
}

PAGE_MAP.get(current_page, page_home)()
