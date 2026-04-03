# Market Analysis — Opportunity Landscape
## Independent Deployment Guide (Zero Perplexity Dependency)

Once deployed on Render.com, this app runs **forever** — the hourly refresh,
EOD retrain, and dashboard all work independently of Perplexity.

---

## What You Get After Deployment

- A **permanent public URL** (e.g., `https://market-analysis-opportunity-landscape.onrender.com`)
- **Hourly auto-refresh** during market hours (9:15 AM – 3:30 PM IST) — built into the server
- **EOD full model retrain** at 3:45 PM IST every trading day — built into the server
- **Password-protected dashboard** (password: `nifty2026` — change below)
- **Zero dependency on Perplexity** — deleting the Perplexity task has no effect

---

## Step 1: Create a GitHub Account (if you don't have one)

1. Go to https://github.com
2. Click "Sign up" and create a free account
3. Verify your email

---

## Step 2: Create a New GitHub Repository

1. Go to https://github.com/new
2. Repository name: `market-analysis-opportunity-landscape`
3. Set to **Private** (recommended)
4. Do NOT initialize with README (leave unchecked)
5. Click "Create repository"

---

## Step 3: Upload Files to GitHub

**Option A — Upload via browser (easiest, no Git knowledge needed):**

1. Open your new repository on GitHub
2. Click "uploading an existing file" (link in the empty repo page)
3. Drag and drop ALL files from this package:
   - `server.py`
   - `analysis.py`
   - `daily_refresh.py`
   - `requirements.txt`
   - `render.yaml`
   - `.gitignore`
   - `model_state.json`
   - Folder: `data/` (all CSV files inside)
   - Folder: `webapp/` (index.html inside)
4. Scroll down, add commit message: "Initial deployment"
5. Click "Commit changes"

**Option B — Upload via Git (if you have Git installed):**
```bash
cd /path/to/this/package
git init
git add .
git commit -m "Initial deployment"
git remote add origin https://github.com/YOUR_USERNAME/market-analysis-opportunity-landscape.git
git push -u origin main
```

---

## Step 4: Deploy on Render.com

1. Go to https://render.com and sign up for a **free account**
   (Use "Sign in with GitHub" for easiest setup)

2. Click **"New +"** → **"Web Service"**

3. Connect your GitHub repository:
   - Select "market-analysis-opportunity-landscape"
   - Click "Connect"

4. Configure the service:
   - **Name:** `market-analysis-opportunity-landscape`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn server:app --workers 1 --threads 2 --timeout 120`
   - **Instance Type:** Free

5. Add Environment Variable:
   - Click "Advanced" → "Add Environment Variable"
   - Key: `REFRESH_TOKEN`
   - Value: any secret string (e.g., `my-secret-refresh-2026`)

6. Click **"Create Web Service"**

Render will build and deploy your app in 3-5 minutes.

---

## Step 5: Your Live URL

Once deployed, Render gives you a URL like:
```
https://market-analysis-opportunity-landscape.onrender.com
```

Open it in any browser — you will see the password prompt.
**Default password: `nifty2026`**

---

## Changing the Password

To change the password:

1. Open `webapp/index.html` in a text editor
2. Find this line near the top:
   ```
   var CORRECT_HASH = "8421433d399f65969c8268358b2e3e4f3dfe97235754c3208516fcd71c843b18";
   ```
3. Generate the SHA-256 hash of your new password at: https://emn178.github.io/online-tools/sha256.html
4. Replace the hash value
5. Push to GitHub — Render auto-deploys within 1-2 minutes

---

## How the Auto-Refresh Works

The `server.py` runs a background scheduler thread that:

| Time | Action |
|------|--------|
| Every 60 min, 9:15 AM – 3:30 PM IST (Mon-Fri, non-holidays) | Fetch live prices, retrain model, update dashboard |
| 3:45 PM IST (Mon-Fri, non-holidays) | Full EOD retrain with actual close price |
| Weekends & NSE holidays | Skip automatically |

**No cron jobs needed. No Perplexity needed. It just runs.**

---

## Keeping the Free Instance Alive

Render's free tier sleeps after 15 minutes of inactivity.
The built-in scheduler (which runs every 60 min during market hours) keeps it awake automatically during trading hours.

For off-hours, if you want it always awake, sign up at https://cron-job.org (free) and add a job to ping:
```
https://market-analysis-opportunity-landscape.onrender.com/api/health
```
Every 14 minutes — keeps it from sleeping.

---

## File Structure

```
market-analysis-opportunity-landscape/
├── server.py              # Flask API + background scheduler (replaces all Perplexity crons)
├── analysis.py            # Correlation engine + Ridge Regression model (v2, exp-weighted)
├── daily_refresh.py       # Data fetch + model retrain pipeline
├── requirements.txt       # Python dependencies
├── render.yaml            # Render.com deployment config
├── model_state.json       # Current model state (auto-updated by scheduler)
├── data/
│   ├── NSEI.csv           # NIFTY 50 historical OHLCV
│   ├── SP500.csv          # S&P 500
│   ├── DJI.csv            # Dow Jones
│   ├── NDX.csv            # Nasdaq 100
│   ├── CRUDE.csv          # WTI Crude Oil
│   ├── GOLD.csv           # Gold Futures
│   ├── USDINR.csv         # USD/INR
│   ├── DXY.csv            # US Dollar Index
│   ├── INDIAVIX.csv       # India VIX
│   └── NSEBANK.csv        # Bank NIFTY
└── webapp/
    └── index.html         # Full dashboard (password-protected, fetches /api/state live)
```

---

## Troubleshooting

**Dashboard shows old data:**
- Check `/api/health` — confirms last refresh time
- Render free tier may be sleeping — open the URL to wake it

**Build fails:**
- Check that all files are uploaded to GitHub
- Ensure `requirements.txt` is in the root folder (not inside a subfolder)

**Model not retraining:**
- Check Render logs (Dashboard → your service → Logs)
- yfinance occasionally fails for specific tickers — the script has fallbacks

---

## Support

This app was built by the Perplexity Computer AI assistant.
The code is fully self-contained — any Python developer can maintain it.
