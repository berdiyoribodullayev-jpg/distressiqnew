"""
Script 3 of 4 — Model Training (Pure NumPy)
============================================
Trains a regularised logistic regression classifier using only
NumPy and Pandas — no external ML libraries required.

The model is trained on WRDS/Compustat historical data (1989-2024)
to predict 3-year forward financial distress, then saved as JSON
for use in the Streamlit app.

Run:
    python 03_train_model.py

Output:
    model/distress_model.json     — trained weights + metadata
    model/threshold.json          — optimal decision threshold
    model/eval_report.txt         — evaluation metrics
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("  Logistic Regression Training — Financial Distress Model")
print("  Trained on WRDS/Compustat 1989–2024 data")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")

df = pd.read_csv(DATA_DIR / "ml_dataset.csv", low_memory=False)
with open(DATA_DIR / "feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

# Use 3-year distress target — more cases, better signal
TARGET = "distress_3yr"

print(f"  Observations : {len(df):,}")
print(f"  Companies    : {df['gvkey'].nunique():,}")
print(f"  Features     : {len(FEATURE_COLS)}")
print(f"  Target       : {TARGET}")
print(f"  Distress rate: {df[TARGET].mean():.3%}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train / test split (temporal)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Splitting train/test...")

train = df[df["fyear"] <= 2015]
test  = df[df["fyear"] >  2015]

X_tr = train[FEATURE_COLS].values.astype(np.float64)
y_tr = train[TARGET].values.astype(np.float64)
X_te = test[FEATURE_COLS].values.astype(np.float64)
y_te = test[TARGET].values.astype(np.float64)

print(f"  Train: {len(train):,} obs  (≤2015) | positives: {int(y_tr.sum())}")
print(f"  Test : {len(test):,} obs  (>2015) | positives: {int(y_te.sum())}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Standardise features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Training model...")

mu  = X_tr.mean(axis=0)
std = X_tr.std(axis=0) + 1e-8
X_tr_s = (X_tr - mu) / std
X_te_s = (X_te - mu) / std

# ── Weighted Logistic Regression via mini-batch Adam ─────────────────────────
np.random.seed(42)
n_feat   = X_tr_s.shape[1]
w        = np.zeros(n_feat, dtype=np.float64)
b        = 0.0
lr       = 0.01
l2       = 0.01
n_epochs = 300
batch_sz = 4096

# Class weight: weight positive class by inverse frequency
n_pos = y_tr.sum()
n_neg = len(y_tr) - n_pos
pos_w = n_neg / (n_pos + 1e-9)   # ~390 for 3yr target
print(f"  pos_weight = {pos_w:.1f}  ({int(n_neg)} neg / {int(n_pos)} pos)")

# Adam parameters
m_w = np.zeros_like(w); v_w = np.zeros_like(w)
m_b = 0.0;              v_b = 0.0
beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
t = 0

idx = np.arange(len(X_tr_s))
best_val_loss = np.inf
best_w, best_b = w.copy(), b

for epoch in range(1, n_epochs + 1):
    np.random.shuffle(idx)
    for start in range(0, len(idx), batch_sz):
        batch = idx[start:start+batch_sz]
        Xb, yb = X_tr_s[batch], y_tr[batch]
        t += 1
        # Forward
        logit = Xb @ w + b
        logit = np.clip(logit, -50, 50)
        prob  = 1 / (1 + np.exp(-logit))
        # Weighted loss gradient
        sample_w = np.where(yb == 1, pos_w, 1.0)
        err = (prob - yb) * sample_w
        grad_w = Xb.T @ err / len(batch) + l2 * w
        grad_b = err.mean()
        # Adam update
        m_w = beta1*m_w + (1-beta1)*grad_w
        v_w = beta2*v_w + (1-beta2)*grad_w**2
        m_b = beta1*m_b + (1-beta1)*grad_b
        v_b = beta2*v_b + (1-beta2)*grad_b**2
        mw_hat = m_w / (1 - beta1**t)
        vw_hat = v_w / (1 - beta2**t)
        mb_hat = m_b / (1 - beta1**t)
        vb_hat = v_b / (1 - beta2**t)
        w -= lr * mw_hat / (np.sqrt(vw_hat) + eps_adam)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + eps_adam)

    # Validation loss every 50 epochs
    if epoch % 50 == 0:
        logit_te = np.clip(X_te_s @ w + b, -50, 50)
        p_te = 1 / (1 + np.exp(-logit_te))
        sw_te = np.where(y_te == 1, pos_w, 1.0)
        val_loss = -(sw_te * (y_te * np.log(p_te+1e-9) +
                              (1-y_te) * np.log(1-p_te+1e-9))).mean()
        print(f"  Epoch {epoch:3d} | val_loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_w, best_b = w.copy(), b

w, b = best_w, best_b

# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating...")

logit_te = np.clip(X_te_s @ w + b, -50, 50)
y_prob   = 1 / (1 + np.exp(-logit_te))

# AUC-ROC (manual)
pos_probs = y_prob[y_te == 1]
neg_probs = y_prob[y_te == 0]
n_pos_te, n_neg_te = len(pos_probs), len(neg_probs)
auc_roc = (pos_probs[:, None] > neg_probs[None, :]).mean()
print(f"  AUC-ROC : {auc_roc:.4f}")

# Find best F1 threshold
best_f1, best_thr = 0, 0.5
for thr in np.linspace(0.01, 0.99, 200):
    y_pred = (y_prob >= thr).astype(float)
    tp = ((y_pred==1) & (y_te==1)).sum()
    fp = ((y_pred==1) & (y_te==0)).sum()
    fn = ((y_pred==0) & (y_te==1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

y_pred_final = (y_prob >= best_thr).astype(float)
tp = ((y_pred_final==1) & (y_te==1)).sum()
fp = ((y_pred_final==1) & (y_te==0)).sum()
fn = ((y_pred_final==0) & (y_te==1)).sum()
tn = ((y_pred_final==0) & (y_te==0)).sum()
prec = tp / (tp + fp + 1e-9)
rec  = tp / (tp + fn + 1e-9)

print(f"  Best threshold : {best_thr:.4f}")
print(f"  Precision      : {prec:.4f}")
print(f"  Recall         : {rec:.4f}")
print(f"  F1             : {best_f1:.4f}")
print(f"  TP={int(tp)}  FP={int(fp)}  FN={int(fn)}  TN={int(tn)}")

# Top features by weight magnitude
feat_imp = sorted(zip(FEATURE_COLS, np.abs(w)), key=lambda x: -x[1])
print("\n  Top 15 features by weight:")
for name, imp in feat_imp[:15]:
    bar = "█" * int(imp * 20)
    sign = "+" if w[FEATURE_COLS.index(name)] > 0 else "-"
    print(f"  {sign} {name:<25} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving model...")

model_bundle = {
    "weights"     : w.tolist(),
    "bias"        : float(b),
    "feature_mu"  : mu.tolist(),
    "feature_std" : std.tolist(),
    "feature_cols": FEATURE_COLS,
    "target"      : TARGET,
    "pos_weight"  : float(pos_w),
    "train_obs"   : int(len(train)),
    "test_obs"    : int(len(test)),
    "auc_roc"     : round(float(auc_roc), 6),
    "best_f1"     : round(float(best_f1), 6),
}

with open(MODEL_DIR / "distress_model.json", "w") as f:
    json.dump(model_bundle, f)

threshold_meta = {
    "threshold"  : float(best_thr),
    "auc_roc"    : round(float(auc_roc), 6),
    "best_f1"    : round(float(best_f1), 6),
    "target"     : TARGET,
    "model_type" : "logistic_regression_numpy",
}
with open(MODEL_DIR / "threshold.json", "w") as f:
    json.dump(threshold_meta, f, indent=2)

eval_lines = [
    "Financial Distress Model — Evaluation Report",
    "=" * 55,
    f"Model type     : Logistic Regression (NumPy)",
    f"Training data  : WRDS/Compustat 1989–2024",
    f"Target         : {TARGET}",
    f"Train period   : ≤2015  ({len(train):,} obs, {int(y_tr.sum())} positives)",
    f"Test period    : >2015  ({len(test):,} obs, {int(y_te.sum())} positives)",
    "",
    f"AUC-ROC        : {auc_roc:.4f}",
    f"Best threshold : {best_thr:.4f}",
    f"Precision      : {prec:.4f}",
    f"Recall         : {rec:.4f}",
    f"F1             : {best_f1:.4f}",
    f"TP={int(tp)}  FP={int(fp)}  FN={int(fn)}  TN={int(tn)}",
    "",
    "Top 15 Features by Weight Magnitude:",
    *[f"  {'+'if w[FEATURE_COLS.index(n)]>0 else '-'} {n:<25} {v:.4f}"
      for n, v in feat_imp[:15]],
]
with open(MODEL_DIR / "eval_report.txt", "w") as f:
    f.write("\n".join(eval_lines))

print(f"  ✓ model/distress_model.json")
print(f"  ✓ model/threshold.json")
print(f"  ✓ model/eval_report.txt")
print(f"""
{'='*65}
  Training complete!
  AUC-ROC = {auc_roc:.4f}  |  F1 = {best_f1:.4f}  |  Threshold = {best_thr:.4f}
  Next step: python 04_predict_ticker.py AAPL
{'='*65}
""")
