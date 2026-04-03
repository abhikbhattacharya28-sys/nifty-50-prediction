"""
Market Analysis — Opportunity Landscape
Independent Flask API Server
Runs on Render.com — serves MODEL_STATE to the web app via HTTP API
Zero dependency on Perplexity.
"""

import os
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading

app = Flask(__name__, static_folder='webapp', static_url_path='')
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, 'model_state.json')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e), "last_updated": "N/A"}

def is_trading_time():
    """Check if current IST time is within market hours on a trading day."""
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST)
    holidays = {
        '2026-01-15','2026-01-26','2026-03-03','2026-03-26','2026-03-31',
        '2026-04-03','2026-04-14','2026-05-01','2026-05-28','2026-06-26',
        '2026-09-14','2026-10-02','2026-10-20','2026-11-08','2026-11-10',
        '2026-11-24','2026-12-25'
    }
    date_str = now.strftime('%Y-%m-%d')
    if now.weekday() >= 5 or date_str in holidays:
        return False, 'holiday'
    hhmm = now.hour * 60 + now.minute
    if hhmm < 9 * 60 + 15:   return False, 'pre-market'
    if hhmm > 15 * 60 + 30:  return False, 'post-market'
    return True, 'open'

def run_refresh():
    """Run the full data fetch + model retrain pipeline."""
    print(f"[{datetime.now().isoformat()}] Running daily_refresh.py...")
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, 'daily_refresh.py')],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("Refresh OK")
        else:
            print(f"Refresh ERROR:\n{result.stderr[-500:]}")
    except Exception as e:
        print(f"Refresh exception: {e}")

# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/api/state')
def api_state():
    """Serve the full MODEL_STATE as JSON — called by the web app."""
    state = load_state()
    resp = jsonify(state)
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/api/health')
def health():
    """Health check endpoint."""
    is_open, status = is_trading_time()
    state = load_state()
    return jsonify({
        'status':       'ok',
        'market':       status,
        'last_updated': state.get('last_updated', 'N/A'),
        'last_refresh': state.get('last_full_refresh', 'N/A'),
        'server_time':  datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime('%Y-%m-%d %H:%M IST')
    })

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Trigger a manual refresh (secured with env var token)."""
    token = os.environ.get('REFRESH_TOKEN', '')
    from flask import request
    if request.headers.get('X-Refresh-Token') != token:
        return jsonify({'error': 'unauthorized'}), 401
    thread = threading.Thread(target=run_refresh, daemon=True)
    thread.start()
    return jsonify({'status': 'refresh started'})

# ─────────────────────────────────────────────────────────────
# STATIC WEB APP — serve index.html
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('webapp', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('webapp', path)

# ─────────────────────────────────────────────────────────────
# BACKGROUND SCHEDULER — replaces Perplexity cron jobs
# Runs inside the server process, no external cron needed
# ─────────────────────────────────────────────────────────────

def scheduler_loop():
    """
    Background thread that replicates the Perplexity cron jobs:
    - Hourly during market hours (9:15 AM – 3:30 PM IST)
    - EOD full retrain at 3:45 PM IST
    Runs forever, independently of Perplexity.
    """
    import time
    last_run_hour = -1
    last_eod_date = None

    while True:
        try:
            IST = timezone(timedelta(hours=5, minutes=30))
            now = datetime.now(IST)
            is_open, status = is_trading_time()
            current_hour = now.hour
            current_date = now.date()
            hhmm = now.hour * 60 + now.minute

            # Hourly intraday refresh: trigger once per hour during market hours
            if is_open and current_hour != last_run_hour:
                print(f"[SCHEDULER] Intraday refresh at {now.strftime('%H:%M IST')}")
                run_refresh()
                last_run_hour = current_hour

            # EOD full retrain: 3:45–3:50 PM IST on trading days
            eod_window = (15 * 60 + 45 <= hhmm <= 15 * 60 + 50)
            if eod_window and status != 'holiday' and now.weekday() < 5 and last_eod_date != current_date:
                print(f"[SCHEDULER] EOD retrain at {now.strftime('%H:%M IST')}")
                run_refresh()
                last_eod_date = current_date

        except Exception as e:
            print(f"[SCHEDULER] Error: {e}")

        time.sleep(60)  # Check every minute

if __name__ == '__main__':
    # Start background scheduler thread
    sched = threading.Thread(target=scheduler_loop, daemon=True)
    sched.start()
    print("Background scheduler started.")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
