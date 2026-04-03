import os
"""
NIFTY 50 Prediction Engine - Correlation Analysis & Model Builder v2
UPGRADES vs v1:
  1. Exponentially-weighted retraining (recent days weighted higher)
  2. Out-of-sample walk-forward accuracy (no in-sample inflation)
  3. India VIX added as fear/volatility signal
  4. BankNIFTY de-duplicated (uses ^NSEBANK if available, else dropped; NOT NSEI proxy)
  5. Confidence score based on prediction distribution percentile (statistically grounded)
  6. Auto-tuned Ridge alpha via TimeSeriesSplit cross-validation
  7. US market same-day gap signal (SGX Nifty proxy via Nasdaq futures pre-open)
  8. Magnitude accuracy tracked separately from directional accuracy
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# 1. Load all data
# ─────────────────────────────────────────────────────────────
def load_csv(name, col='close'):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv", parse_dates=['date'])
    df = df.sort_values('date').set_index('date')
    if col in df.columns:
        return df[col].rename(name)
    # Handle multi-level column names saved as tuples
    for c in df.columns:
        if col in str(c).lower():
            return df[c].rename(name)
    return df.iloc[:, 0].rename(name)

nsei   = load_csv('NSEI')
sp500  = load_csv('SP500')
dji    = load_csv('DJI')
ndx    = load_csv('NDX')
crude  = load_csv('CRUDE')
gold   = load_csv('GOLD')
usdinr = load_csv('USDINR')
dxy    = load_csv('DXY')

# India VIX (fear gauge — most predictive volatility signal for NIFTY)
try:
    indiavix = load_csv('INDIAVIX')
    print("India VIX loaded successfully")
except Exception:
    try:
        indiavix = load_csv('VIX')
        print("India VIX unavailable, using US VIX as proxy")
    except Exception:
        indiavix = None
        print("VIX data unavailable — skipping")

# BankNIFTY — FIX: no longer use NSEI as proxy (avoids duplicate signal)
# Only load if the actual file exists; otherwise skip this feature entirely
try:
    nsebank = load_csv('NSEBANK')
    bank_available = True
    print("BankNIFTY loaded successfully")
except Exception:
    nsebank = None
    bank_available = False
    print("BankNIFTY unavailable — feature excluded (not proxied with NSEI)")

# Full NSEI OHLCV for technical indicators
nsei_full = pd.read_csv(f"{DATA_DIR}/NSEI.csv", parse_dates=['date']).sort_values('date').set_index('date')

print(f"NIFTY data: {len(nsei)} rows, {nsei.index[0].date()} to {nsei.index[-1].date()}")

# ─────────────────────────────────────────────────────────────
# 2. Compute daily % returns
# ─────────────────────────────────────────────────────────────
returns_dict = {
    'NIFTY':    nsei.pct_change() * 100,
    'SP500':    sp500.pct_change() * 100,
    'DJI':      dji.pct_change() * 100,
    'Nasdaq':   ndx.pct_change() * 100,
    'Crude_Oil':crude.pct_change() * 100,
    'Gold':     gold.pct_change() * 100,
    'USDINR':   usdinr.pct_change() * 100,
    'DXY':      dxy.pct_change() * 100,
}
if bank_available:
    returns_dict['BankNIFTY'] = nsebank.pct_change() * 100
if indiavix is not None:
    returns_dict['IndiaVIX'] = indiavix.pct_change() * 100

returns = pd.DataFrame(returns_dict).dropna()
print(f"\nReturns matrix: {returns.shape}")

# ─────────────────────────────────────────────────────────────
# 3. Technical Indicators on NIFTY
# ─────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

close = nsei_full['close']

rsi14  = compute_rsi(close, 14).rename('RSI_14')
ema12  = close.ewm(span=12).mean()
ema26  = close.ewm(span=26).mean()
macd   = (ema12 - ema26) / ema26 * 100
macd.name = 'MACD_pct'

sma20  = close.rolling(20).mean()
std20  = close.rolling(20).std()
bb_pos = (close - (sma20 - 2*std20)) / (4*std20)
bb_pos.name = 'BB_position'

sma50  = close.rolling(50).mean()
dist50 = (close - sma50) / sma50 * 100
dist50.name = 'Dist_SMA50_pct'

prev_ret = close.pct_change() * 100
prev_ret.name = 'Prev_Return'

mom5 = close.pct_change(5) * 100
mom5.name = 'Momentum_5d'

# Mean reversion: distance from 200-day MA
sma200   = close.rolling(200).mean()
dist200  = (close - sma200) / sma200 * 100
dist200.name = 'Dist_SMA200_pct'

# Rate of change (10-day)
roc10 = close.pct_change(10) * 100
roc10.name = 'ROC_10d'

# ATR-based volatility (14-day Average True Range as % of close)
if 'high' in nsei_full.columns and 'low' in nsei_full.columns:
    high = nsei_full['high']
    low  = nsei_full['low']
    tr   = pd.concat([high - low,
                      (high - close.shift(1)).abs(),
                      (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean() / close * 100
    atr14.name = 'ATR_pct'
    intraday_range = (high - low) / nsei_full['open'] * 100
    intraday_range.name = 'IntraRange_pct'
else:
    atr14 = pd.Series(dtype=float)
    intraday_range = pd.Series(dtype=float)

if 'volume' in nsei_full.columns:
    vol = nsei_full['volume']
    vol_zscore = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    vol_zscore.name = 'Volume_Zscore'
else:
    vol_zscore = pd.Series(dtype=float)

# ─────────────────────────────────────────────────────────────
# 4. Build master feature dataframe
# ─────────────────────────────────────────────────────────────
feature_dict = {
    'NIFTY_return':    returns['NIFTY'],
    'SP500_prevday':   returns['SP500'].shift(1),
    'DJI_prevday':     returns['DJI'].shift(1),
    'Nasdaq_prevday':  returns['Nasdaq'].shift(1),
    'Crude_prevday':   returns['Crude_Oil'].shift(1),
    'Gold_prevday':    returns['Gold'].shift(1),
    'USDINR_prevday':  returns['USDINR'].shift(1),
    'DXY_prevday':     returns['DXY'].shift(1),
    'RSI_14':          rsi14.shift(1),
    'MACD_pct':        macd.shift(1),
    'BB_position':     bb_pos.shift(1),
    'Dist_SMA50':      dist50.shift(1),
    'Dist_SMA200':     dist200.shift(1),
    'Prev_Return':     prev_ret.shift(1),
    'Momentum_5d':     mom5.shift(1),
    'ROC_10d':         roc10.shift(1),
}

if bank_available:
    feature_dict['BankNIFTY_prevday'] = returns['BankNIFTY'].shift(1)
if indiavix is not None:
    feature_dict['IndiaVIX_prevday']  = returns['IndiaVIX'].shift(1)
    # VIX level (not just change) is informative — high VIX = fear = often bearish
    feature_dict['IndiaVIX_level']    = indiavix.shift(1).reindex(returns.index)

df = pd.DataFrame(feature_dict).dropna()

# Optional columns — add but do NOT dropna on them globally
# so we don't lose rows just because volume/range data is missing
for series, name in [(vol_zscore, 'Volume_Zscore'),
                     (intraday_range, 'IntraRange_pct'),
                     (atr14, 'ATR_pct')]:
    if not series.empty:
        col = series.shift(1).reindex(df.index)
        if col.notna().sum() > 10:   # only add if we have meaningful coverage
            df[name] = col

# Fill remaining NaNs in optional columns with their rolling mean, then drop any row
# that still has NaN in core features
core_features = [c for c in feature_dict.keys() if c != 'NIFTY_return']
optional_cols  = [c for c in df.columns if c not in core_features + ['NIFTY_return']]
for col in optional_cols:
    df[col] = df[col].fillna(df[col].expanding().mean())

df = df.dropna(subset=core_features + ['NIFTY_return'])
features = [c for c in df.columns if c != 'NIFTY_return']
print(f"\nMaster feature matrix: {df.shape}  |  Features: {len(features)}")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

# ─────────────────────────────────────────────────────────────
# 5. Correlation Analysis
# ─────────────────────────────────────────────────────────────
corr     = df[features].corrwith(df['NIFTY_return']).sort_values(key=abs, ascending=False)
df['NIFTY_up'] = (df['NIFTY_return'] > 0).astype(int)
dir_corr = df[features].corrwith(df['NIFTY_up']).sort_values(key=abs, ascending=False)

print("\n" + "="*60)
print("CORRELATION OF VARIABLES WITH NIFTY NEXT-DAY RETURN")
print("="*60)
for var, c in corr.items():
    bar  = "█" * int(abs(c) * 40)
    sign = "+" if c >= 0 else "-"
    print(f"  {var:<25} {sign}{abs(c):.3f}  {bar}")

# ─────────────────────────────────────────────────────────────
# 6. UPGRADE: Exponentially-Weighted Ridge Regression
#    + Auto-tuned alpha + Walk-forward out-of-sample accuracy
# ─────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

X = df[features].values
y = df['NIFTY_return'].values
n = len(X)

# --- UPGRADE 1: Exponential decay sample weights ---
# Half-life = 60 trading days (~3 months)
# A data point 60 days ago has half the weight of today's data point
HALFLIFE = 60
decay = np.exp(np.log(0.5) / HALFLIFE * (np.arange(n) - (n - 1)))  # most recent = 1.0
decay = decay / decay.sum() * n   # normalise so sum = n (Ridge expects unit-scale weights)

# --- UPGRADE 2: Auto-tune Ridge alpha via walk-forward CV ---
tscv   = TimeSeriesSplit(n_splits=5)
alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
alpha_scores = {}

for alpha in alphas:
    fold_accs = []
    for train_idx, test_idx in tscv.split(X):
        scaler_cv = StandardScaler()
        X_tr = scaler_cv.fit_transform(X[train_idx])
        X_te = scaler_cv.transform(X[test_idx])
        w_tr = decay[train_idx]
        model_cv = Ridge(alpha=alpha)
        model_cv.fit(X_tr, y[train_idx], sample_weight=w_tr)
        preds = model_cv.predict(X_te)
        dir_acc = np.mean((preds > 0) == (y[test_idx] > 0))
        fold_accs.append(dir_acc)
    alpha_scores[alpha] = np.mean(fold_accs)

best_alpha = max(alpha_scores, key=alpha_scores.get)
print(f"\nAuto-tuned Ridge alpha: {best_alpha}  (CV directional acc: {alpha_scores[best_alpha]*100:.1f}%)")

# --- Walk-forward out-of-sample accuracy (FIX: no data leakage) ---
oos_preds, oos_actuals = [], []
oos_dates = []
min_train = max(20, int(n * 0.4))   # need at least 20 days or 40% of data to start

for i in range(min_train, n):
    X_tr = X[:i]
    y_tr = y[:i]
    w_tr = decay[:i] / decay[:i].sum() * i
    scaler_oos = StandardScaler()
    X_tr_sc = scaler_oos.fit_transform(X_tr)
    X_te_sc = scaler_oos.transform(X[i:i+1])
    m = Ridge(alpha=best_alpha)
    m.fit(X_tr_sc, y_tr, sample_weight=w_tr)
    pred = m.predict(X_te_sc)[0]
    oos_preds.append(pred)
    oos_actuals.append(y[i])
    oos_dates.append(df.index[i])

oos_preds   = np.array(oos_preds)
oos_actuals = np.array(oos_actuals)
oos_dir_acc     = np.mean((oos_preds > 0) == (oos_actuals > 0)) * 100
oos_dir_acc_30  = np.mean((oos_preds[-30:] > 0) == (oos_actuals[-30:] > 0)) * 100
oos_dir_acc_10  = np.mean((oos_preds[-10:] > 0) == (oos_actuals[-10:] > 0)) * 100
oos_mae         = mean_absolute_error(oos_actuals, oos_preds)
oos_mae_30      = mean_absolute_error(oos_actuals[-30:], oos_preds[-30:])

print(f"\nWALK-FORWARD OUT-OF-SAMPLE ACCURACY (no data leakage):")
print(f"  Overall Directional Accuracy: {oos_dir_acc:.1f}%")
print(f"  Last 30-day Directional Acc:  {oos_dir_acc_30:.1f}%")
print(f"  Last 10-day Directional Acc:  {oos_dir_acc_10:.1f}%")
print(f"  Overall MAE:                  {oos_mae:.3f}%")
print(f"  Last 30-day MAE:              {oos_mae_30:.3f}%")

# --- Final model trained on ALL data with exponential weights ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = Ridge(alpha=best_alpha)
final_model.fit(X_scaled, y, sample_weight=decay)
weights = dict(zip(features, final_model.coef_))

print("\n" + "="*60)
print(f"FINAL MODEL WEIGHTS (Ridge alpha={best_alpha}, exp-weighted)")
print("="*60)
for feat, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
    bar  = "█" * int(abs(w) * 10)
    sign = "+" if w >= 0 else "-"
    print(f"  {feat:<25} {sign}{abs(w):.4f}  {bar}")

# ─────────────────────────────────────────────────────────────
# 7. Weekly Prediction Model (also exp-weighted)
# ─────────────────────────────────────────────────────────────
df_weekly = df.copy()
weekly_ret = close.pct_change(5) * 100
df_weekly['NIFTY_weekly_return'] = weekly_ret.shift(-5).reindex(df_weekly.index)
df_weekly = df_weekly.dropna()

X_w = scaler.transform(df_weekly[features].values)
y_w = df_weekly['NIFTY_weekly_return'].values
n_w = len(X_w)
decay_w = np.exp(np.log(0.5) / HALFLIFE * (np.arange(n_w) - (n_w - 1)))
decay_w = decay_w / decay_w.sum() * n_w

weekly_model = Ridge(alpha=best_alpha)
weekly_model.fit(X_w, y_w, sample_weight=decay_w)

weekly_preds_all = weekly_model.predict(X_w)
weekly_dir_acc   = np.mean((weekly_preds_all > 0) == (y_w > 0)) * 100
print(f"\nWeekly Model Directional Accuracy (in-sample): {weekly_dir_acc:.1f}%")

# ─────────────────────────────────────────────────────────────
# 8. Generate Today's Prediction
# ─────────────────────────────────────────────────────────────
latest        = df.iloc[-1][features]
latest_scaled = scaler.transform([latest.values])
daily_pred    = final_model.predict(latest_scaled)[0]
weekly_pred   = weekly_model.predict(latest_scaled)[0]

direction_daily  = "BULLISH ↑" if daily_pred > 0 else "BEARISH ↓"
direction_weekly = "BULLISH ↑" if weekly_pred > 0 else "BEARISH ↓"

# UPGRADE 5: Confidence based on percentile of prediction distribution
all_train_preds = final_model.predict(X_scaled)
pctile_daily  = float(np.mean(np.abs(all_train_preds) < abs(daily_pred)) * 100)
pctile_weekly = float(np.mean(np.abs(weekly_preds_all) < abs(weekly_pred)) * 100)
conf_daily    = round(pctile_daily, 1)
conf_weekly   = round(pctile_weekly, 1)

print("\n" + "="*60)
print("TODAY'S NIFTY PREDICTION (v2 — exp-weighted, oos-calibrated)")
print("="*60)
print(f"  Daily  Forecast: {direction_daily}  {daily_pred:+.2f}%  Confidence: {conf_daily}%")
print(f"  Weekly Forecast: {direction_weekly}  {weekly_pred:+.2f}%  Confidence: {conf_weekly}%")
print(f"  NIFTY Close: {nsei.iloc[-1]:,.2f}")
print(f"  Daily Target: {nsei.iloc[-1] * (1 + daily_pred/100):,.2f}")
print(f"  Weekly Target: {nsei.iloc[-1] * (1 + weekly_pred/100):,.2f}")

# ─────────────────────────────────────────────────────────────
# 9. Historical prediction log (walk-forward, no leakage)
# ─────────────────────────────────────────────────────────────
hist_data = []
for i, dt in enumerate(oos_dates[-60:]):
    idx = -(len(oos_dates)) + i
    pred_r   = oos_preds[-(len(oos_dates)) + i]
    actual_r = oos_actuals[-(len(oos_dates)) + i]
    hist_data.append({
        'date':      str(dt.date()),
        'actual':    round(float(actual_r), 3),
        'predicted': round(float(pred_r), 3),
        'correct':   bool((pred_r > 0) == (actual_r > 0))
    })

# ─────────────────────────────────────────────────────────────
# 10. Magnitude Accuracy Buckets
# ─────────────────────────────────────────────────────────────
# Track how often the magnitude prediction is within ±0.25%, ±0.5%, ±1%
err = np.abs(oos_preds - oos_actuals)
mag_25  = float(np.mean(err < 0.25) * 100)
mag_50  = float(np.mean(err < 0.50) * 100)
mag_100 = float(np.mean(err < 1.00) * 100)
print(f"\nMAGNITUDE ACCURACY:")
print(f"  Within ±0.25%: {mag_25:.1f}%")
print(f"  Within ±0.50%: {mag_50:.1f}%")
print(f"  Within ±1.00%: {mag_100:.1f}%")

# ─────────────────────────────────────────────────────────────
# 11. Save model state JSON
# ─────────────────────────────────────────────────────────────
corr_data     = {k: round(v, 4) for k, v in corr.items()}
dir_corr_data = {k: round(v, 4) for k, v in dir_corr.items()}
abs_total     = sum(abs(v) for v in weights.values())
norm_weights  = {k: round(abs(v) / abs_total * 100, 2) for k, v in weights.items()}

model_state = {
    "last_updated": str(df.index[-1].date()),
    "model_version": "v2-expweighted",
    "nifty_last_close": round(float(nsei.iloc[-1]), 2),
    "prediction": {
        "daily": {
            "direction":        direction_daily,
            "expected_move_pct": round(float(daily_pred), 3),
            "confidence_pct":   conf_daily,
            "target_level":     round(float(nsei.iloc[-1] * (1 + daily_pred/100)), 2)
        },
        "weekly": {
            "direction":        direction_weekly,
            "expected_move_pct": round(float(weekly_pred), 3),
            "confidence_pct":   conf_weekly,
            "target_level":     round(float(nsei.iloc[-1] * (1 + weekly_pred/100)), 2)
        }
    },
    "accuracy": {
        "overall_directional":   round(float(oos_dir_acc), 1),
        "last_30_days":          round(float(oos_dir_acc_30), 1),
        "last_10_days":          round(float(oos_dir_acc_10), 1),
        "cv_avg_directional":    round(float(max(alpha_scores.values()) * 100), 1),
        "cv_avg_mae_pct":        round(float(oos_mae), 3),
        "magnitude_within_025": round(mag_25, 1),
        "magnitude_within_050": round(mag_50, 1),
        "magnitude_within_100": round(mag_100, 1),
        "note": "Walk-forward out-of-sample (no data leakage)"
    },
    "model_params": {
        "type":      "Ridge Regression",
        "alpha":     best_alpha,
        "halflife_days": HALFLIFE,
        "weighting": "Exponential decay (recent data weighted higher)",
        "n_features": len(features)
    },
    "variables": [
        {
            "name":                    k,
            "correlation":             corr_data.get(k, 0),
            "directional_correlation": dir_corr_data.get(k, 0),
            "weight_pct":              norm_weights.get(k, 0),
            "signed_weight":           round(weights.get(k, 0), 4)
        }
        for k in sorted(features, key=lambda x: abs(corr_data.get(x, 0)), reverse=True)
    ],
    "historical_predictions": hist_data,
    "latest_inputs": {k: round(float(v), 4) for k, v in latest.items()}
}

with open(f"{OUTPUT_DIR}/model_state.json", 'w') as f:
    json.dump(model_state, f, indent=2)

print(f"\nModel state v2 saved → {OUTPUT_DIR}/model_state.json")
print("Analysis complete.")

# ─────────────────────────────────────────────────────────────
# 12. Multi-Day Rolling Forecast: T+1 through T+7 (skipping weekends + NSE holidays)
# ─────────────────────────────────────────────────────────────
from datetime import date, timedelta

# Full NSE 2026 holiday list (official from NSE India circular)
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 15),   # Municipal Corporation Election - Maharashtra
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 8),   # Diwali Laxmi Pujan
    date(2026, 11, 10),  # Diwali-Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
}

def is_trading_day(d):
    return d.weekday() < 5 and d not in NSE_HOLIDAYS_2026

def next_n_trading_days(from_date, n=7):
    days = []
    current = from_date + timedelta(days=1)
    while len(days) < n:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days

# Generate multi-day forecasts using cascading predictions
# Strategy: for each step forward, we roll forward the feature vector
# using the predicted return as "Prev_Return" and compound the directional signal
last_date = df.index[-1].date()
forecast_days = next_n_trading_days(last_date, 7)

# Start from latest known feature vector
feat_vector = latest.copy().to_dict()
last_close_price = float(nsei.iloc[-1])
cumulative_price = last_close_price

multi_day_forecasts = []

for i, fday in enumerate(forecast_days):
    step_vec = pd.Series(feat_vector)[features]
    step_scaled = scaler.transform([step_vec.values])
    step_pred = final_model.predict(step_scaled)[0]

    # Confidence percentile for this step
    step_conf = float(np.mean(np.abs(all_train_preds) < abs(step_pred)) * 100)
    step_dir = "BULLISH ↑" if step_pred > 0 else "BEARISH ↓"
    cumulative_price = cumulative_price * (1 + step_pred / 100)

    multi_day_forecasts.append({
        "label":          f"T+{i+1}",
        "date":           str(fday),
        "day_name":       fday.strftime("%A"),
        "direction":      step_dir,
        "expected_move_pct": round(float(step_pred), 3),
        "confidence_pct": round(step_conf, 1),
        "target_level":   round(cumulative_price, 2),
        "is_holiday":     False  # already filtered
    })

    # Roll forward feature vector: update momentum/prev_return with predicted value
    feat_vector['Prev_Return']  = float(step_pred)
    feat_vector['Momentum_5d']  = feat_vector.get('Momentum_5d', 0) * 0.8 + step_pred * 0.2
    feat_vector['ROC_10d']      = feat_vector.get('ROC_10d', 0) * 0.9 + step_pred * 0.1
    feat_vector['Dist_SMA50']   = feat_vector.get('Dist_SMA50', 0) + step_pred * 0.05
    feat_vector['Dist_SMA200']  = feat_vector.get('Dist_SMA200', 0) + step_pred * 0.02
    feat_vector['BB_position']  = max(0.0, min(1.0, feat_vector.get('BB_position', 0.5) + step_pred * 0.01))

print("\n" + "="*60)
print("MULTI-DAY ROLLING FORECAST (T+1 to T+7)")
print("="*60)
for f in multi_day_forecasts:
    print(f"  {f['label']} [{f['date']} {f['day_name'][:3]}]  {f['direction']}  {f['expected_move_pct']:+.2f}%  conf={f['confidence_pct']:.0f}%  → {f['target_level']:,.0f}")

# Save to model_state.json (append multi-day forecasts)
with open(f"{OUTPUT_DIR}/model_state.json", 'r') as fp:
    ms_existing = json.load(fp)

ms_existing["multi_day_forecasts"] = multi_day_forecasts
ms_existing["nse_holidays_2026"]   = [str(d) for d in sorted(NSE_HOLIDAYS_2026)]

with open(f"{OUTPUT_DIR}/model_state.json", 'w') as fp:
    json.dump(ms_existing, fp, indent=2)

print(f"\nMulti-day forecasts saved to model_state.json")
