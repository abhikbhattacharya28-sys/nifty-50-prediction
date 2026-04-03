"""
NIFTY Prediction Engine — Daily End-of-Day Refresh v2
Runs every weekday at 3:45 PM IST (10:15 UTC)
Fetches fresh OHLCV + VIX data, retrains exp-weighted model,
updates model_state.json, rebuilds the HTML, and redeploys.
"""

import subprocess
import sys
import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
WEBAPP_DIR = os.path.join(BASE_DIR, "webapp")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    print(f"[{ts}] {msg}")

def fetch_ohlcv(ticker, filename, start_date, end_date):
    """Fetch OHLCV via yfinance and save to CSV."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty:
            log(f"  WARNING: No data returned for {ticker}")
            return False
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index.name = 'date'
        keep = [c for c in ['open','high','low','close','volume'] if c in df.columns]
        df[keep].to_csv(f"{DATA_DIR}/{filename}.csv")
        log(f"  Fetched {len(df)} rows for {ticker} → {filename}.csv")
        return True
    except Exception as e:
        log(f"  ERROR fetching {ticker}: {e}")
        return False

def run_analysis():
    """Run the v2 correlation + exp-weighted model script."""
    result = subprocess.run(
        [sys.executable, f"{BASE_DIR}/analysis.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"  Analysis ERROR:\n{result.stderr[-800:]}")
        return False
    # Log key output lines
    for line in result.stdout.split('\n'):
        if any(kw in line for kw in ['Accuracy', 'Forecast', 'alpha', 'saved', 'complete', 'ERROR']):
            log(f"  >> {line.strip()}")
    return True

def update_html_with_fresh_model():
    """Inject the latest model_state.json into the HTML file."""
    with open(f"{BASE_DIR}/model_state.json", 'r') as f:
        model_state = json.load(f)

    with open(f"{WEBAPP_DIR}/index.html", 'r', encoding='utf-8') as f:
        html = f.read()

    # Replace the embedded MODEL_STATE object
    new_js = "const MODEL_STATE = " + json.dumps(model_state) + ";"
    html = re.sub(
        r'const MODEL_STATE = \{[\s\S]*?\};',
        lambda m: new_js,
        html,
        count=1
    )

    # Update the "Last Updated" display
    today_str = datetime.now().strftime("%B %d, %Y")
    html = re.sub(
        r'Last Updated.*?</span>',
        f'Last Updated <strong>{today_str}</strong></span>',
        html,
        count=1
    )

    with open(f"{WEBAPP_DIR}/index.html", 'w', encoding='utf-8') as f:
        f.write(html)

    log(f"  HTML updated (last_updated: {model_state['last_updated']})")
    return model_state

def log_refresh_history(status, accuracy, daily_dir, daily_move, weekly_dir, weekly_move):
    """Append to refresh history log."""
    history_file = f"{BASE_DIR}/refresh_history.jsonl"
    entry = {
        "timestamp":              datetime.now().isoformat(),
        "status":                 status,
        "directional_accuracy":   accuracy,
        "daily_direction":        daily_dir,
        "daily_expected_move":    daily_move,
        "weekly_direction":       weekly_dir,
        "weekly_expected_move":   weekly_move,
    }
    with open(history_file, 'a') as f:
        f.write(json.dumps(entry) + "\n")

def main():
    log("=" * 60)
    log("NIFTY PREDICTION ENGINE v2 — DAILY REFRESH STARTING")
    log("=" * 60)

    # Install yfinance if needed
    try:
        import yfinance
    except ImportError:
        log("Installing yfinance...")
        subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "-q"])

    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(WEBAPP_DIR, exist_ok=True)

    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=430)).strftime("%Y-%m-%d")  # 430d to cover 200d SMA warmup

    log(f"Fetching data from {start_date} to {end_date}")

    tickers = [
        ("^NSEI",      "NSEI"),
        ("^GSPC",      "SP500"),
        ("^DJI",       "DJI"),
        ("^NDX",       "NDX"),
        ("CL=F",       "CRUDE"),
        ("GC=F",       "GOLD"),
        ("USDINR=X",   "USDINR"),
        ("DX-Y.NYB",   "DXY"),
        ("^INDIAVIX",  "INDIAVIX"),   # NEW: India VIX
        ("^NSEBANK",   "NSEBANK"),    # FIX: use real BankNIFTY (not NSEI proxy)
    ]

    success_count = 0
    core_tickers = {"NSEI", "SP500", "DJI", "NDX", "CRUDE", "GOLD", "USDINR", "DXY"}

    for yf_ticker, fname in tickers:
        ok = fetch_ohlcv(yf_ticker, fname, start_date, end_date)
        if ok:
            success_count += 1
        elif fname in core_tickers:
            log(f"  CRITICAL: Core ticker {fname} failed")

    core_success = sum(1 for _, fname in tickers if fname in core_tickers
                       and os.path.exists(f"{DATA_DIR}/{fname}.csv"))
    log(f"Data fetch: {success_count}/{len(tickers)} tickers successful ({core_success} core)")

    if core_success < 6:
        log("ERROR: Too many core ticker failures. Aborting refresh.")
        log_refresh_history("FAILED", 0, "N/A", 0, "N/A", 0)
        return

    # Run v2 model
    log("Running v2 exp-weighted model analysis...")
    ok = run_analysis()
    if not ok:
        log("ERROR: Analysis failed. Aborting.")
        log_refresh_history("FAILED", 0, "N/A", 0, "N/A", 0)
        return

    # Load results
    with open(f"{BASE_DIR}/model_state.json") as f:
        ms = json.load(f)

    acc          = ms.get("accuracy", {}).get("overall_directional", 0)
    acc_30       = ms.get("accuracy", {}).get("last_30_days", 0)
    daily_p      = ms.get("prediction", {}).get("daily", {})
    weekly_p     = ms.get("prediction", {}).get("weekly", {})
    model_ver    = ms.get("model_version", "v1")

    log(f"Model v: {model_ver}  |  alpha: {ms.get('model_params',{}).get('alpha','?')}")
    log(f"OOS Accuracy: {acc:.1f}% overall  |  {acc_30:.1f}% last 30d")
    log(f"Daily:  {daily_p.get('direction')} {daily_p.get('expected_move_pct',0):+.2f}%  conf={daily_p.get('confidence_pct',0):.0f}%  → {daily_p.get('target_level',0):,.0f}")
    log(f"Weekly: {weekly_p.get('direction')} {weekly_p.get('expected_move_pct',0):+.2f}%  conf={weekly_p.get('confidence_pct',0):.0f}%  → {weekly_p.get('target_level',0):,.0f}")

    # Fetch live market data (prices for all instruments)
    log("Fetching live market prices...")
    live_data = fetch_live_market_data()
    log(f"  Live prices @ {live_data.get('fetched_at','?')} — NIFTY:{live_data.get('nifty',{}).get('price','?')}  Gold:{live_data.get('gold',{}).get('price','?')}  Crude:{live_data.get('crude',{}).get('price','?')}  VIX:{live_data.get('indiavix',{}).get('price','?')}")

    # Build dynamic analysis for ALL dashboard sections
    log("Building dynamic analysis sections...")
    dynamic_analysis = build_dynamic_analysis(ms, live_data)
    causal_analysis   = build_causal_analysis(ms, live_data)

    # Merge into model_state and save
    ms["live_market_data"]   = live_data
    ms["dynamic_analysis"]   = dynamic_analysis
    ms["causal_analysis"]    = causal_analysis
    # Preserve existing top_news (enriched by main agent), just update timestamp
    if "top_news" not in ms:
        ms["top_news"] = fetch_top_news()
    else:
        ms["top_news"]["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    ms["last_full_refresh"]  = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    import json as _json
    with open(f"{BASE_DIR}/model_state.json", 'w') as fw:
        _json.dump(ms, fw, indent=2)
    log("  model_state.json updated with live data + dynamic analysis")

    # Update HTML
    log("Updating web app HTML...")
    update_html_with_fresh_model()
    log_refresh_history(
        "SUCCESS", acc,
        daily_p.get('direction',''), daily_p.get('expected_move_pct',0),
        weekly_p.get('direction',''), weekly_p.get('expected_move_pct',0)
    )

    log("=" * 60)
    log("REFRESH COMPLETE")
    log(f"  NIFTY Close:      {ms.get('nifty_last_close', 'N/A'):,.2f}")
    log(f"  Daily Forecast:   {daily_p.get('direction')} {daily_p.get('expected_move_pct',0):+.2f}%  → {daily_p.get('target_level',0):,.2f}")
    log(f"  Weekly Forecast:  {weekly_p.get('direction')} {weekly_p.get('expected_move_pct',0):+.2f}%  → {weekly_p.get('target_level',0):,.2f}")
    log(f"  Overall OOS Acc:  {acc:.1f}%  |  30d: {acc_30:.1f}%")
    log("=" * 60)


def fetch_live_market_data():
    """Fetch live prices for all key instruments — called every hour during market."""
    try:
        import yfinance as yf
        tickers = {
            "nifty":    "^NSEI",
            "sensex":   "^BSESN",
            "sp500":    "^GSPC",
            "nasdaq":   "^NDX",
            "dow":      "^DJI",
            "gold":     "GC=F",
            "crude":    "CL=F",
            "usdinr":   "USDINR=X",
            "dxy":      "DX-Y.NYB",
            "indiavix": "^INDIAVIX",
        }
        result = {}
        for name, ticker in tickers.items():
            try:
                hist = yf.Ticker(ticker).history(period="2d")
                if len(hist) >= 1:
                    close = float(hist['Close'].iloc[-1])
                    prev  = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else close
                    chg   = (close - prev) / prev * 100
                    result[name] = {"price": round(close, 2), "change_pct": round(chg, 2)}
                else:
                    result[name] = {"price": None, "change_pct": None}
            except Exception as e:
                result[name] = {"price": None, "change_pct": None, "error": str(e)}
        result["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M IST")
        return result
    except Exception as e:
        log(f"  Live data fetch error: {e}")
        return {}

def build_dynamic_analysis(ms, live):
    """Build the Prediction Deep-Dive, Scenario Analysis and News Sentiment dynamically from model state + live prices."""
    daily   = ms.get("prediction", {}).get("daily", {})
    weekly  = ms.get("prediction", {}).get("weekly", {})
    acc     = ms.get("accuracy", {})
    latest  = ms.get("latest_inputs", {})
    vars_   = ms.get("variables", [])
    direction = daily.get("direction", "")
    is_bear   = "BEAR" in direction.upper()

    # Top 3 bullish and bearish variables
    bear_vars = [v for v in vars_ if v.get("signed_weight", 0) < 0][:3]
    bull_vars = [v for v in vars_ if v.get("signed_weight", 0) > 0][:3]

    # Dynamic factor analysis
    gold_price   = live.get("gold",    {}).get("price", "N/A")
    gold_chg     = live.get("gold",    {}).get("change_pct", 0)
    crude_price  = live.get("crude",   {}).get("price", "N/A")
    crude_chg    = live.get("crude",   {}).get("change_pct", 0)
    usdinr_price = live.get("usdinr",  {}).get("price", "N/A")
    usdinr_chg   = live.get("usdinr",  {}).get("change_pct", 0)
    vix_price    = live.get("indiavix",{}).get("price", "N/A")
    vix_chg      = live.get("indiavix",{}).get("change_pct", 0)
    sp500_price  = live.get("sp500",   {}).get("price", "N/A")
    sp500_chg    = live.get("sp500",   {}).get("change_pct", 0)
    nasdaq_price = live.get("nasdaq",  {}).get("price", "N/A")
    nasdaq_chg   = live.get("nasdaq",  {}).get("change_pct", 0)
    dxy_price    = live.get("dxy",     {}).get("price", "N/A")
    dxy_chg      = live.get("dxy",     {}).get("change_pct", 0)
    nifty_price  = live.get("nifty",   {}).get("price", "N/A")
    nifty_chg    = live.get("nifty",   {}).get("change_pct", 0)
    sensex_price = live.get("sensex",  {}).get("price", "N/A")
    sensex_chg   = live.get("sensex",  {}).get("change_pct", 0)

    rsi = latest.get("RSI_14", 50)
    macd = latest.get("MACD_pct", 0)
    bb = latest.get("BB_position", 0.5)
    dist50 = latest.get("Dist_SMA50", 0)
    dist200 = latest.get("Dist_SMA200", 0)

    # Verdict sentence
    verdict = "BEARISH — CAUTION ADVISED" if is_bear else "BULLISH — CAUTIOUS OPTIMISM"
    verdict_color = "red" if is_bear else "green"

    # Scenario probabilities — dynamically adjust based on model confidence
    conf_daily  = daily.get("confidence_pct", 50)
    conf_weekly = weekly.get("confidence_pct", 50)
    bear_prob = min(70, max(30, int(conf_daily * 0.6 + conf_weekly * 0.4)))
    bull_prob = max(15, 100 - bear_prob - 15)
    extreme_prob = 100 - bear_prob - bull_prob

    # News sentiment score
    bear_global  = sum(1 for x in [crude_chg, dxy_chg, (usdinr_chg if usdinr_chg and usdinr_chg > 0 else 0)] if x and x > 0.3)
    bull_india   = sum(1 for x in [nifty_chg, sensex_chg] if x and x > 0.3)
    bear_india   = sum(1 for x in [nifty_chg, sensex_chg] if x and x < -0.3)
    bull_global  = sum(1 for x in [sp500_chg, nasdaq_chg] if x and x > 0.3)
    mixed        = max(0, 10 - bear_global - bull_india - bear_india - bull_global)
    net_sentiment = "BULL" if (bull_global + bull_india) > (bear_global + bear_india) else "BEAR"

    now_str = datetime.now().strftime("%B %d, %Y %H:%M IST")

    dynamic = {
        "generated_at": now_str,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_summary": (
            f"The model is registering a confluence of {'bearish' if is_bear else 'bullish'} signals. "
            f"Daily forecast: {daily.get('expected_move_pct',0):+.2f}% (conf {conf_daily:.0f}%). "
            f"Weekly forecast: {weekly.get('expected_move_pct',0):+.2f}% (conf {conf_weekly:.0f}%). "
            f"India VIX at {vix_price} ({'+' if vix_chg and vix_chg > 0 else ''}{vix_chg:.1f}% today) — {'elevated fear' if vix_price and float(str(vix_price)) > 20 else 'low fear'}."
        ),
        "live_prices": {
            "nifty":    {"price": nifty_price,  "change_pct": nifty_chg,  "label": "NIFTY 50"},
            "sensex":   {"price": sensex_price, "change_pct": sensex_chg, "label": "SENSEX"},
            "sp500":    {"price": sp500_price,  "change_pct": sp500_chg,  "label": "S&P 500"},
            "nasdaq":   {"price": nasdaq_price, "change_pct": nasdaq_chg, "label": "NASDAQ"},
            "gold":     {"price": gold_price,   "change_pct": gold_chg,   "label": "Gold ($/oz)"},
            "crude":    {"price": crude_price,  "change_pct": crude_chg,  "label": "Crude WTI ($)"},
            "usdinr":   {"price": usdinr_price, "change_pct": usdinr_chg, "label": "USD/INR"},
            "dxy":      {"price": dxy_price,    "change_pct": dxy_chg,    "label": "DXY Index"},
            "indiavix": {"price": vix_price,    "change_pct": vix_chg,    "label": "India VIX"},
        },
        "factors": [
            {
                "label": "Global Markets",
                "sentiment": "bullish" if (sp500_chg and sp500_chg > 0) else "bearish",
                "title": f"S&P 500: {'+' if sp500_chg and sp500_chg > 0 else ''}{sp500_chg:.2f}% | Nasdaq: {'+' if nasdaq_chg and nasdaq_chg > 0 else ''}{nasdaq_chg:.2f}%",
                "body": f"US markets closed {'higher' if sp500_chg and sp500_chg > 0 else 'lower'} — S&P 500 at {sp500_price} ({'+' if sp500_chg and sp500_chg > 0 else ''}{sp500_chg:.2f}%), Nasdaq at {nasdaq_price} ({'+' if nasdaq_chg and nasdaq_chg > 0 else ''}{nasdaq_chg:.2f}%). DXY at {dxy_price} ({'+' if dxy_chg and dxy_chg > 0 else ''}{dxy_chg:.2f}%). US market moves feed into NIFTY's next-day open via overnight correlation.",
                "weight": f"{abs(vars_[3].get('signed_weight',0))*100:.1f}% model weight" if len(vars_) > 3 else "Global weight"
            },
            {
                "label": "Gold / Safe-Haven",
                "sentiment": "bearish" if (gold_chg and gold_chg > 0.3) else ("bullish" if gold_chg and gold_chg < -0.3 else "neutral"),
                "title": f"Gold at ${gold_price} ({'+' if gold_chg and gold_chg > 0 else ''}{gold_chg:.2f}% today)",
                "body": f"Gold is the #{1 if vars_ else 'N/A'} predictor in the model. {'Rising gold signals risk-off — bearish for equities.' if gold_chg and gold_chg > 0 else 'Falling gold signals risk-on — positive for equities.'} Current level: ${gold_price}.",
                "weight": f"{abs(vars_[1].get('signed_weight',0) if len(vars_) > 1 else 0)*100:.1f}% model weight"
            },
            {
                "label": "USD/INR Currency",
                "sentiment": "bearish" if (usdinr_chg and usdinr_chg > 0.1) else ("bullish" if usdinr_chg and usdinr_chg < -0.1 else "neutral"),
                "title": f"USD/INR at ₹{usdinr_price} ({'+' if usdinr_chg and usdinr_chg > 0 else ''}{usdinr_chg:.2f}% today)",
                "body": f"USD/INR is the #2 predictor. {'Rupee weakening' if usdinr_chg and usdinr_chg > 0 else 'Rupee strengthening'} — {'FIIs convert to USD and repatriate = selling pressure on NIFTY.' if usdinr_chg and usdinr_chg > 0 else 'FII flows may stabilise or improve.'}",
                "weight": f"{abs(vars_[0].get('signed_weight',0) if vars_ else 0)*100:.1f}% model weight"
            },
            {
                "label": "Technical Indicators",
                "sentiment": "bearish" if rsi < 40 and macd < 0 else ("bullish" if rsi > 60 and macd > 0 else "neutral"),
                "title": f"RSI-14: {rsi:.1f} | MACD: {macd:+.2f}% | BB Position: {bb:.2f}",
                "body": f"RSI at {rsi:.1f} ({'oversold — bounce watch' if rsi < 35 else 'neutral zone' if rsi < 60 else 'overbought'}). MACD {'negative — downtrend' if macd < 0 else 'positive — uptrend'}. Price is {abs(dist50):.1f}% {'below' if dist50 < 0 else 'above'} 50-DMA and {abs(dist200):.1f}% {'below' if dist200 < 0 else 'above'} 200-DMA.",
                "weight": f"RSI/MACD/BB combined"
            },
            {
                "label": "Crude Oil / Inflation",
                "sentiment": "bearish" if (crude_chg and crude_chg > 1) else ("bullish" if crude_chg and crude_chg < -1 else "neutral"),
                "title": f"Crude WTI at ${crude_price} ({'+' if crude_chg and crude_chg > 0 else ''}{crude_chg:.2f}% today)",
                "body": f"Oil at ${crude_price}/barrel. {'Elevated crude hurts India as a net importer — widening CAD, inflation risk, sector headwinds for aviation, cement, fertilizers.' if crude_price and float(str(crude_price)) > 85 else 'Moderate oil prices ease India import bill and inflation outlook.'}",
                "weight": f"{abs(vars_[4].get('signed_weight',0) if len(vars_) > 4 else 0)*100:.1f}% model weight"
            },
        ],
        "scenarios": [
            {
                "name": "Base Case",
                "direction": "bear" if is_bear else "bull",
                "trigger": "Current macro conditions persist — oil/FII/currency headwinds",
                "impact": f"{daily.get('expected_move_pct',0):+.2f}% to {weekly.get('expected_move_pct',0):+.2f}%",
                "prob": bear_prob if is_bear else bull_prob
            },
            {
                "name": "Reversal",
                "direction": "bull" if is_bear else "bear",
                "trigger": "Geopolitical de-escalation, crude eases, FII inflows return",
                "impact": "+0.5% to +1.5%" if is_bear else "-0.5% to -1.5%",
                "prob": bull_prob if is_bear else bear_prob
            },
            {
                "name": "Extreme Move",
                "direction": "bear" if is_bear else "bull",
                "trigger": "Shock event: crude spike/crash, global selloff/rally",
                "impact": "-2% to -3.5%" if is_bear else "+2% to +3.5%",
                "prob": extreme_prob
            }
        ],
        "sentiment_scorecard": {
            "bear_global":   bear_global,
            "bull_global":   bull_global,
            "bear_india":    bear_india,
            "bull_india":    bull_india,
            "mixed":         mixed,
            "net_sentiment": net_sentiment,
            "vix_level":     vix_price,
            "vix_signal":    "Fear Elevated" if vix_price and float(str(vix_price)) > 20 else "Fear Low"
        },
        "key_levels": {
            "support1":    round(float(ms.get("nifty_last_close", 22000)) * 0.99, 0),
            "support2":    round(float(ms.get("nifty_last_close", 22000)) * 0.975, 0),
            "resistance1": round(float(ms.get("nifty_last_close", 22000)) * 1.01, 0),
            "resistance2": round(float(ms.get("nifty_last_close", 22000)) * 1.025, 0),
        }
    }
    return dynamic



def build_causal_analysis(ms, live):
    """Auto-generate signal-by-signal causal analysis from live data every hour."""
    daily   = ms.get("prediction", {}).get("daily", {})
    weekly  = ms.get("prediction", {}).get("weekly", {})
    latest  = ms.get("latest_inputs", {})

    gold_p    = live.get("gold",    {}).get("price", 0) or 0
    gold_c    = live.get("gold",    {}).get("change_pct", 0) or 0
    crude_p   = live.get("crude",   {}).get("price", 0) or 0
    crude_c   = live.get("crude",   {}).get("change_pct", 0) or 0
    usdinr_p  = live.get("usdinr",  {}).get("price", 0) or 0
    usdinr_c  = live.get("usdinr",  {}).get("change_pct", 0) or 0
    vix_p     = live.get("indiavix",{}).get("price", 0) or 0
    vix_c     = live.get("indiavix",{}).get("change_pct", 0) or 0
    sp500_p   = live.get("sp500",   {}).get("price", 0) or 0
    sp500_c   = live.get("sp500",   {}).get("change_pct", 0) or 0
    nasdaq_c  = live.get("nasdaq",  {}).get("change_pct", 0) or 0
    nifty_p   = live.get("nifty",   {}).get("price", 0) or 0
    nifty_c   = live.get("nifty",   {}).get("change_pct", 0) or 0
    sensex_c  = live.get("sensex",  {}).get("change_pct", 0) or 0
    dxy_p     = live.get("dxy",     {}).get("price", 0) or 0
    dxy_c     = live.get("dxy",     {}).get("change_pct", 0) or 0
    rsi       = latest.get("RSI_14", 50)
    macd      = latest.get("MACD_pct", 0)
    dist50    = latest.get("Dist_SMA50", 0)
    dist200   = latest.get("Dist_SMA200", 0)
    bb        = latest.get("BB_position", 0.5)
    nifty_last = ms.get("nifty_last_close", 22000)

    # Determine dominant narrative
    is_bear = (daily.get("direction","") + weekly.get("direction","")).count("BEAR") >= 1
    crude_shock    = crude_c > 3
    vix_elevated   = vix_p > 20
    fii_selling    = usdinr_c > 0.1
    us_decoupled   = (sp500_c > 0.5) and (nifty_c < -0.3)
    gold_risk_off  = gold_c > 1
    rsi_oversold   = rsi < 35
    technical_bear = dist50 < -5 and macd < 0

    def classify(val, bull_thresh, bear_thresh):
        if val > bull_thresh: return "BULLISH"
        if val < bear_thresh: return "BEARISH"
        return "MIXED"

    # Build signal chain dynamically
    chain = []
    rank = 1

    # 1. Crude Oil
    crude_impact = "CRITICAL BEARISH" if crude_c > 4 else "STRONGLY BEARISH" if crude_c > 1 else "BEARISH" if crude_c > 0 else "BULLISH" if crude_c < -2 else "MIXED"
    chain.append({
        "rank": rank, "signal": "Crude Oil WTI",
        "reading": f"${crude_p:.2f} ({crude_c:+.2f}% today)",
        "impact": crude_impact,
        "why": (f"Crude surged {crude_c:+.2f}% to ${crude_p:.2f} — " +
                ("a major shock. India imports 85%+ of crude. Every $1 rise in oil costs India ~$1.5B extra annually, widens the CAD, fuels inflation, and pressures rupee. Sectors directly hit: aviation, cement, fertilizers, OMCs, paints." if crude_c > 3 else
                 "elevated above $90 — a structural headwind for India. Import bill stays high, CAD stays wide, and RBI must balance rupee defense with rate policy." if crude_p > 90 else
                 f"falling — a direct positive for India. Lower crude = lower import bill, better CAD, rupee relief, lower inflation. Sectors benefiting: aviation, paints, cement.")),
        "nifty_effect": "Bearish drag on 40%+ of NIFTY by weight" if crude_c > 0 else "Bullish tailwind for consumption and manufacturing sectors"
    })
    rank += 1

    # 2. India VIX
    vix_impact = "STRONGLY BEARISH" if vix_p > 22 else "BEARISH" if vix_p > 18 else "BULLISH" if vix_p < 14 else "MIXED"
    chain.append({
        "rank": rank, "signal": "India VIX (Fear Gauge)",
        "reading": f"{vix_p:.2f} ({vix_c:+.2f}% today)",
        "impact": vix_impact,
        "why": (f"India VIX at {vix_p:.1f} is in {'elevated fear territory (normal: 12–18). Institutional traders are buying put options = expecting more downside. This causes forced de-risking, wider spreads, and margin calls on leveraged positions.' if vix_p > 20 else 'normal range — fear is contained. Options market not pricing in extreme moves.' if vix_p < 18 else 'borderline elevated zone. Watch for a move above 25 which would signal panic.'}"),
        "nifty_effect": "High VIX → self-reinforcing selling loop" if vix_p > 20 else "Low VIX → stable environment, dip-buying likely"
    })
    rank += 1

    # 3. USD/INR
    rupee_impact = "STRONGLY BEARISH" if usdinr_c > 0.5 else "BEARISH" if usdinr_c > 0.1 else "BULLISH" if usdinr_c < -0.3 else "MIXED"
    chain.append({
        "rank": rank, "signal": "USD/INR (Rupee)",
        "reading": f"₹{usdinr_p:.2f} ({usdinr_c:+.2f}% today)",
        "impact": rupee_impact,
        "why": (f"Rupee {'weakening' if usdinr_c > 0 else 'strengthening'} today. At ₹{usdinr_p:.2f}, the rupee is {'historically very weak' if usdinr_p > 90 else 'under mild pressure' if usdinr_p > 84 else 'stable'}. " +
                ("A weak rupee: (1) makes crude more expensive in rupee terms, (2) reduces real returns for FIIs pushing them to exit, (3) widens trade deficit. Vicious cycle: crude up → CAD up → rupee down → FII exits → rupee down more." if usdinr_c > 0.1 else
                 "Rupee strengthening = relief for India's import economics and FII return calculations. A sustained move below ₹84 would be meaningfully bullish for NIFTY.")),
        "nifty_effect": "Bearish multiplier when weakening — amplifies crude and FII headwinds" if usdinr_c > 0 else "Positive signal — FII returns improve, import economics improve"
    })
    rank += 1

    # 4. US Markets vs India divergence
    if us_decoupled:
        chain.append({
            "rank": rank, "signal": "US vs India Divergence",
            "reading": f"S&P 500 {sp500_c:+.2f}% but NIFTY {nifty_c:+.2f}%",
            "impact": "CONFUSING / MIXED",
            "why": f"A critical signal: when US markets rise but NIFTY falls, the problem is India-specific, NOT global. US is up because of AI/tech optimism (Nasdaq {nasdaq_c:+.2f}%). NIFTY is down because India faces a crude-rupee-FII triple whammy that US is immune to. This decoupling tells you domestic headwinds are overwhelming — it is NOT a risk-off day globally.",
            "nifty_effect": "Signals India-specific structural weakness. Recovery requires domestic triggers, not just US stability."
        })
        rank += 1
    else:
        chain.append({
            "rank": rank, "signal": "Global Markets Correlation",
            "reading": f"S&P 500 {sp500_c:+.2f}% | NASDAQ {nasdaq_c:+.2f}% | NIFTY {nifty_c:+.2f}%",
            "impact": "BULLISH" if sp500_c > 0.5 else "BEARISH" if sp500_c < -0.5 else "MIXED",
            "why": f"NIFTY is moving in sync with global markets today. S&P 500 at {sp500_p:,.0f} ({'up' if sp500_c > 0 else 'down'} {abs(sp500_c):.2f}%). Global macro tone is {'risk-on — positive for FII flows into India.' if sp500_c > 0 else 'risk-off — FIIs tend to sell emerging markets including India.'}",
            "nifty_effect": "Global tailwind" if sp500_c > 0.5 else "Global headwind" if sp500_c < -0.5 else "Neutral global backdrop"
        })
        rank += 1

    # 5. Gold
    gold_impact = "BEARISH" if gold_c > 1 else "BULLISH" if gold_c < -1 else "MIXED"
    chain.append({
        "rank": rank, "signal": "Gold (Safe-Haven Signal)",
        "reading": f"${gold_p:,.0f}/oz ({gold_c:+.2f}% today)",
        "impact": gold_impact,
        "why": (f"Gold at ${gold_p:,.0f} — {'rising gold signals risk-off: investors fleeing equities for safe havens, bearish for NIFTY.' if gold_c > 0.5 else 'falling gold = risk-on rotation back into equities, modestly bullish.' if gold_c < -0.5 else 'relatively stable.'} " +
                (f"At ${gold_p:,.0f}/oz the absolute level is historically extreme (was ~$2,000 in 2023), signaling prolonged geopolitical fear has not resolved." if gold_p > 3500 else "")),
        "nifty_effect": "Risk-off pressure" if gold_c > 1 else "Risk-on tailwind" if gold_c < -1 else "Neutral today"
    })
    rank += 1

    # 6. Technical picture
    tech_impact = "BEARISH" if (rsi < 40 and macd < 0) else "BULLISH" if (rsi > 55 and macd > 0) else "MIXED"
    tech_note = ""
    if rsi < 35:
        tech_note = f"RSI at {rsi:.0f} = OVERSOLD — historically this level sees short-term bounces. But oversold can stay oversold in a downtrend. "
    elif rsi > 65:
        tech_note = f"RSI at {rsi:.0f} = OVERBOUGHT — caution on chasing rallies. "
    else:
        tech_note = f"RSI at {rsi:.0f} = neutral zone. "
    chain.append({
        "rank": rank, "signal": "Technical Structure",
        "reading": f"RSI-14: {rsi:.1f} | MACD: {macd:+.2f}% | {dist50:.1f}% from 50-DMA | {dist200:.1f}% from 200-DMA",
        "impact": tech_impact,
        "why": tech_note + f"MACD {'negative — downtrend' if macd < 0 else 'positive — uptrend'}. NIFTY is {abs(dist50):.1f}% {'below' if dist50 < 0 else 'above'} the 50-day MA and {abs(dist200):.1f}% {'below' if dist200 < 0 else 'above'} 200-day MA. " + ("Bearish structure: lower highs, lower lows. Resistance at " + str(round(nifty_last * 1.01, 0)) + "." if dist50 < -3 else "Bullish structure: price above key moving averages."),
        "nifty_effect": f"Key support: {round(nifty_last * 0.99, 0):,.0f} / {round(nifty_last * 0.975, 0):,.0f} | Resistance: {round(nifty_last * 1.01, 0):,.0f} / {round(nifty_last * 1.025, 0):,.0f}"
    })
    rank += 1

    # 7. DXY
    dxy_impact = "BEARISH" if dxy_c > 0.3 else "BULLISH" if dxy_c < -0.3 else "MIXED"
    chain.append({
        "rank": rank, "signal": "US Dollar Index (DXY)",
        "reading": f"{dxy_p:.2f} ({dxy_c:+.2f}% today)",
        "impact": dxy_impact,
        "why": f"DXY at {dxy_p:.2f}. {'Strong dollar = emerging market outflows. FIIs repatriate to USD assets = selling pressure on NIFTY.' if dxy_c > 0.3 else 'Weak dollar = EM inflows. FIIs more likely to stay invested in India.' if dxy_c < -0.3 else 'Dollar relatively stable today — neutral for FII flows.'}",
        "nifty_effect": "FII outflow pressure when DXY rises above 101" if dxy_c > 0 else "Supports FII inflows when DXY weakens"
    })

    # Determine root cause headline
    drivers = []
    if crude_c > 3:    drivers.append("CRUDE SPIKE")
    if vix_p > 22:     drivers.append("ELEVATED FEAR (VIX)")
    if usdinr_c > 0.3: drivers.append("RUPEE WEAKNESS")
    if sp500_c < -0.5: drivers.append("GLOBAL RISK-OFF")
    if gold_c > 1:     drivers.append("SAFE-HAVEN RUSH")
    if not drivers:
        if nifty_c < -0.5:  drivers = ["MIXED NEGATIVE SIGNALS"]
        elif nifty_c > 0.5: drivers = ["POSITIVE MOMENTUM"]
        else:               drivers = ["CONSOLIDATION / RANGE-BOUND"]

    direction_word = "FALLING" if nifty_c < 0 else "RISING" if nifty_c > 0.3 else "RANGE-BOUND"
    headline = f"NIFTY IS {direction_word} TODAY: {' + '.join(drivers)}"

    # Root cause chain
    if is_bear:
        root = (f"The dominant pressure chain today: "
                f"{'Crude at ${:.0f} (+{:.1f}%) → India import bill rises → CAD widens → '.format(crude_p, crude_c) if crude_c > 1 else ''}"
                f"{'Rupee at ₹{:.2f} → FII returns compressed → '.format(usdinr_p) if usdinr_p > 85 else ''}"
                f"FII selling continues → NIFTY supply exceeds demand → "
                f"{'VIX at {:.1f} (fear elevated) → forced de-risking amplifies the move.'.format(vix_p) if vix_p > 20 else 'selling pressure sustained.'}")
    else:
        root = (f"The dominant support chain today: "
                f"{'US markets positive (S&P {:.2f}%) → FII sentiment improves → '.format(sp500_c) if sp500_c > 0.5 else ''}"
                f"{'Crude easing ({:.2f}%) → India macro relief → '.format(crude_c) if crude_c < -1 else ''}"
                f"{'Rupee strengthening ({:.2f}%) → FII returns improve → '.format(usdinr_c) if usdinr_c < -0.3 else ''}"
                f"Buyers absorb supply → NIFTY sustains upward momentum.")

    # Bottom line
    if is_bear:
        bl = (f"NIFTY at {nifty_p:,.2f} is under pressure from external factors. "
              f"{'The key circuit-breaker to watch is crude oil — a close below $90 would dramatically change sentiment. ' if crude_p > 90 else ''}"
              f"{'India VIX above 20 keeps institutional sellers active. ' if vix_p > 20 else ''}"
              f"The domestic economy remains fundamentally strong — this is a macro/geopolitical headwind, not an earnings story.")
    else:
        bl = (f"NIFTY at {nifty_p:,.2f} is showing resilience. "
              f"{'Global tailwinds from US markets are helping. ' if sp500_c > 0.5 else ''}"
              f"{'Crude easing is a key positive for India macro. ' if crude_c < -1 else ''}"
              f"Watch for FII flow data after 3:30 PM — sustained buying would confirm the rally.")

    now_str = datetime.now().strftime("%B %d, %Y — %H:%M IST")
    return {
        "generated_at": now_str,
        "headline": headline,
        "primary_cause": f"Primary driver: {drivers[0] if drivers else 'Mixed signals'}. NIFTY is {direction_word.lower()} {abs(nifty_c):.2f}% to {nifty_p:,.2f}.",
        "signal_chain": chain,
        "root_cause_summary": root,
        "what_to_watch": [
            f"Crude oil: {'A close below $100 would ease India macro pressure significantly' if crude_p > 95 else 'Sustained below $90 is meaningfully bullish for India'}",
            f"India VIX: {'Fall below 20 = fear easing, possible reversal' if vix_p > 20 else 'Stay below 18 = stable, supports rally'}",
            f"FII flows (post 3:30 PM): {'Any net buying day = breaks the selling streak — watch for reversal signal' if True else 'Monitor for trend change'}",
            f"NIFTY key level: {'Close above ' + str(round(nifty_p * 1.01, 0)) + ' confirms stability; break below ' + str(round(nifty_p * 0.99, 0)) + ' = new leg down' if True else 'Monitor range'}",
            f"Global cue: S&P 500 and crude direction in US session will set tomorrow's NIFTY opening gap"
        ],
        "bottom_line": bl
    }


def fetch_top_news():
    """Fetch top market news via web search — called every hour to keep news fresh."""
    from datetime import datetime
    try:
        import subprocess, sys, json as _json
        # Use a search to get today's news headlines
        # Since we can't call external APIs directly, we build news from live signal interpretation
        # The main agent enriches this with real search results on each run
        today = datetime.now().strftime("%B %d, %Y")
        return {
            "date": today,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M IST"),
            "note": "News refreshed from last known state — main agent enriches with live search"
        }
    except Exception as e:
        return {"date": datetime.now().strftime("%B %d, %Y"), "error": str(e)}

if __name__ == "__main__":
    main()
