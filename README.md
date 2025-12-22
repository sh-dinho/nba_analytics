# 🏀 NBA Analytics v3  
**End‑to‑End NBA Betting & Analytics Platform**

NBA Analytics v3 is a full-stack, production‑ready system for NBA game modeling and betting strategy evaluation. It ingests historical and live data, builds engineered features, trains predictive models, simulates betting strategies, and surfaces everything through a Streamlit **client portal** and **Telegram alerts**.

Designed as a consulting‑grade platform: transparent, auditable, and ready to demo.

---

## 🚀 Core Capabilities

### ✔ Data & Feature Pipeline
- Canonical **long-format** game representation (two rows per game: one per team)
- Historical ingestion snapshots (schedule, results, team stats)
- Rolling pre‑game features:
  - Win rate, points for/against
  - Home/away indicators
  - Opponent strength metrics
- Strict point‑in‑time correctness (no future leakage)

### ✔ Model Training & Registry
- Configurable classification model (e.g., Random Forest, XGBoost)
- Model registry with:
  - Versioning and timestamps  
  - Feature set metadata  
  - “Production” model selection
- Persisted models for reproducible predictions

### ✔ Game Predictions
- Loads latest production model from registry
- Builds features for a target date’s scheduled games
- Generates **win probabilities** per team
- Saves date‑stamped predictions:
  - `data/predictions/predictions_YYYY-MM-DD.parquet`

### ✔ Betting & Value Detection
- Ingests bookmaker odds snapshots:
  - `data/odds/odds_YYYY-MM-DD.parquet`
- Joins odds with model predictions
- Computes:
  - Implied probabilities (from American odds)
  - Model edge (model win prob − implied prob)
- Foundation for:
  - Value bet detection
  - Bankroll‑aware bet sizing
  - Automated or semi‑automated betting

---

## 📈 Backtesting & Strategy Evaluation

### ✔ Backtesting Engine
- Loads:
  - Historical predictions  
  - Historical odds  
  - Actual outcomes (from canonical long snapshot)
- Simulates bankroll evolution over time using:
  - Fractional Kelly staking  
  - Minimum edge threshold  
  - Max stake fraction per bet
- Outputs:
  - Per‑bet log (stake, result, profit, bankroll_after)
  - Summary metrics:
    - Final bankroll
    - Total profit
    - ROI
    - Hit rate
    - Max drawdown
    - Bets / wins / losses / pushes

### ✔ Accuracy Tracking
- Joins predictions with actual outcomes by date/game/team
- Computes:
  - Overall accuracy (classification)
  - Accuracy by season
- Useful for clients who want to see model performance beyond PnL.

### ✔ Strategy Comparison
- Compare multiple strategies over the same date range:
  - Different `min_edge`, `kelly_fraction`, `max_stake_fraction`
  - Includes a simple baseline (e.g., flat/no-edge)
- Outputs a comparison table:
  - ROI, drawdown, hit rate, bet count, and configuration parameters

---

## 📊 Client Portal (Streamlit)

A **role‑aware** dashboard that serves as your client‑facing UI.

### 🔐 Authentication & Roles
- Simple login:
  - `admin` role: full access
  - `client` role: restricted, presentation‑safe view
- Session‑based login with logout controls

### 🧭 Tabs (Admin)
- **Predictions**  
  - View today’s game probabilities  
  - Join predictions with odds and visualize edge  
- **Backtest / What-if**  
  - Choose date range and strategy parameters  
  - Run historical backtest on demand  
  - See bankroll curve, metrics, and per‑bet log  
  - Generate a client‑ready HTML report with one click  
- **Accuracy**  
  - Compute model accuracy over a given range  
  - See overall and per‑season accuracy  
  - Inspect a sample of predictions vs outcomes  
- **Strategy Comparison**  
  - Compare multiple strategies side‑by‑side

### 🧭 Tabs (Client)
- **Predictions**
- **Backtest / What-if**
- **Accuracy**

> Optional: a separate **Generate Report** tab for clients, if you choose.

### 🖼 Visualizations
- Bankroll over time
- Per‑bet logs
- Edge tables and prediction breakdowns

Run the portal:

```bash
streamlit run src/dashboard/app.py
```
📨 Alerts & Reporting
✔ Telegram Alerts
- Centralized alerts module under src/alerts/:
- Summary alerts from the orchestrator (success/fail per step)
- Backtest / season‑to‑date summaries
- Bankroll curves as images (matplotlib → Telegram photo)
- Environment‑based credentials:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
✔ HTML Reports (Client‑Ready)
- src/reports/backtest_report.py generates:
- Executive Summary with 3–5 auto‑generated insights
- Strategy configuration
- Backtest metrics (ROI, drawdown, win/loss, volume)
- Accuracy metrics (overall and by season)
- Generated to:
- data/reports/report_<start>_<end>_<timestamp>.html
- Can be opened in a browser or exported to PDF via “Print → Save as PDF”.
CLI usage:
```bash
python -m src.reports.run_report --start 2024-10-01 --end 2025-01-01
```
🔄 Orchestrator
The orchestrator coordinates the daily workflow
```bash
python -m src.pipeline.orchestrator
```
Current responsibilities:
- Validate / reuse canonical ingestion snapshots
- Run predictions for a target date
- Join predictions with odds in a betting pipeline
- Log step results with UTC timestamps
- Send a concise Telegram summary alert
The design keeps steps modular so you can plug in:
- Daily odds ingestion
- Automated bet execution
- Additional alerting rules

# Project structure
```bash
nba-analytics-v3/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── config/
│   │   └── paths.py
│   │
│   ├── pipeline/
│   │   └── orchestrator.py
│   │
│   ├── model/
│   │   └── predict.py
│   │
│   ├── features/
│   │   └── builder.py
│   │
│   ├── alerts/
│   │   └── telegram.py
│   │
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── accuracy.py
│   │   ├── compare.py
│   │   ├── run_backtest.py
│   │   └── run_season_to_date.py
│   │
│   ├── reports/
│   │   ├── backtest_report.py
│   │   └── run_report.py
│   │
│   └── dashboard/
│       ├── app.py
│       └── auth.py
│
└── data/
    ├── canonical/
    │   ├── schedule.parquet
    │   └── long.parquet
    ├── models/
    │   └── registry/
    ├── predictions/
    ├── odds/
    ├── logs/
    ├── orchestrator_logs/
    └── reports/
```

🧹 Housekeeping
- Data, models, logs, and reports live under data/ (ignored by Git)
- Orchestrator logs written to data/orchestrator_logs/
- Reports written to data/reports/
You can add your own maintenance scripts (archiving, cleanup) under scripts/ or similar.

📄 License
Internal / Private Project
Customize the license text based on your consulting / client needs.

🙌 Credits
Built with Python, pandas, scikit‑learn, Streamlit, matplotlib, and a lot of care for:
- Reproducibility
- Transparency
- Client‑ready storytelling
- Safe, auditable betting logic


If you want, next time we can:

- Tailor this README to a specific **client vertical** (e.g., “for sportsbooks”, “for hedge funds”, “for syndicates”)  
- Add concrete **example screenshots/flows** for your portal  
- Draft a 1–2 page “Capabilities Deck” you can send alongside this repo.


