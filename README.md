🏀 NBA Analytics Engine — Version 5
A modular, production‑ready NBA analytics system designed for:
- automated ingestion
- feature engineering
- model training + evaluation
- daily predictions
- monitoring + alerting
- backtesting + bankroll simulation
Version 5 represents the final iteration of the versioned architecture before the transition to the canonical, version‑agnostic pipeline.

🚀 Features (v5 Architecture)
1. Ingestion Pipeline (v5)
- Daily ingestion of:
- scoreboard data
- box scores
- betting odds
- team metadata
- Normalization into a long-format snapshot (LONG_SNAPSHOT_V5)
- Handles ESPN, NBA Stats, and betting feed inconsistencies
2. Feature Engineering (v5)
- Feature pipeline: feature_pipeline_v5
- Generates:
- rolling win rates
- ELO ratings
- point differentials
- rest days
- home/away indicators
- matchup features
- Outputs FEATURES_SNAPSHOT_V5
3. Model Training
- Scikit‑learn models (Logistic Regression, Random Forest, XGBoost optional)
- Versioned model registry:
data/models/model_vX.pkl
- Metadata stored in model_registry.json
4. Prediction Pipeline
- Predicts win probabilities for:
- today’s games
- upcoming schedule
- live in‑progress games
- Saves predictions to:
data/predictions/predictions_YYYYMMDD_vX.parquet


5. Monitoring
- Daily, weekly, and monthly monitoring jobs
- Tracks:
- prediction drift
- feature drift
- calibration
- accuracy
- data quality
- Outputs JSON reports in data/logs/
6. Backtesting Engine
- Historical simulation of:
- accuracy
- ROI
- value bets
- bankroll curves
- Configurable:
- Kelly fraction
- min edge
- max stake fraction
- Generates full HTML + JSON reports
7. Alerting System
- Daily alerts via:
- Telegram
- Slack (optional)
- Sends:
- data quality summary
- model monitoring summary
- betting recommendations
- bankroll charts

📁 Project Structure (v5)
src/
  ingestion/
    collector.py
    orchestrator.py
    normalizer/
  features/
    feature_pipeline_v5.py
    feature_schema.py
  model/
    training/
    prediction/
    registry/
  monitoring/
    model_monitor.py
  backtest/
    engine.py
  alerts/
    alert_manager.py
  scripts/
    predict_today.py
    predict_live.py
    predict_all_upcoming.py
    monitor_daily.py
    monitor_weekly.py
    monitor_monthly.py
    run_backtest_report.py
    send_daily_alerts.py
  utils/
    team_names.py
    validate_ingestion_team_names.py



flowchart TD

    %% ============================
    %% INGESTION LAYER
    %% ============================
    subgraph INGESTION["📥 Ingestion Pipeline (v5)"]
        A1[ESPN API] --> O1[Raw Scoreboard JSON]
        A2[NBA Stats API] --> O2[Raw Box Scores JSON]
        A3[Betting Feeds] --> O3[Raw Odds JSON]

        O1 --> N1[Normalizer]
        O2 --> N1
        O3 --> N1

        N1 --> L1[LONG_SNAPSHOT_V5.parquet]
    end

    %% ============================
    %% FEATURE ENGINEERING
    %% ============================
    subgraph FEATURES["🧮 Feature Engineering (v5)"]
        L1 --> F1[feature_pipeline_v5.py]
        F1 --> F2[FEATURES_SNAPSHOT_V5.parquet]
    end

    %% ============================
    %% MODEL TRAINING
    %% ============================
    subgraph MODEL["🤖 Model Training (v5)"]
        F2 --> M1[Train Model (scikit-learn)]
        M1 --> M2[model_vX.pkl]
        M1 --> M3[model_registry.json]
    end

    %% ============================
    %% PREDICTION PIPELINE
    %% ============================
    subgraph PREDICT["🔮 Prediction Pipeline (v5)"]
        F2 --> P1[predict_moneyline()]
        M2 --> P1
        P1 --> P2[predictions_YYYYMMDD_vX.parquet]
    end

    %% ============================
    %% MONITORING
    %% ============================
    subgraph MONITOR["📊 Monitoring (v5)"]
        P2 --> D1[Prediction Drift]
        F2 --> D2[Feature Drift]
        P2 --> D3[Calibration]
        P2 --> D4[Accuracy Tracking]

        D1 --> R1[monitor_report.json]
        D2 --> R1
        D3 --> R1
        D4 --> R1
    end

    %% ============================
    %% BACKTESTING
    %% ============================
    subgraph BACKTEST["📈 Backtesting Engine (v5)"]
        P2 --> B1[Backtest Engine]
        B1 --> B2[ROI / Accuracy / Value Bets]
        B1 --> B3[Bankroll Curve]
        B1 --> B4[backtest_report.html]
    end

    %% ============================
    %% ALERTING
    %% ============================
    subgraph ALERTS["📨 Alerting System (v5)"]
        R1 --> A4[Daily Alerts]
        P2 --> A4
        B3 --> A4
        A4 --> TG[Telegram]
        A4 --> SL[Slack (optional)]
    end
