# 🏀 NBA Analytics v3  
**Fully Automated ML Pipeline for NBA Game Predictions**

NBA Analytics v3 is a complete end‑to‑end machine learning system that ingests NBA data, builds engineered features, trains predictive models, monitors drift, generates predictions, and exposes results through a Streamlit dashboard — all automated and production‑ready.

---

## 🚀 Features

### **✔ Automated Ingestion**
- Fetches full NBA history (via nba_api)
- Daily incremental updates
- Normalized canonical schema
- Versioned ingestion snapshot

### **✔ Feature Engineering**
- Rolling pre‑game statistics (win rate, points for/against)
- Strict point‑in‑time correctness (no leakage)
- Versioned feature snapshots via FeatureStore

### **✔ Model Training**
- Random Forest classifier (configurable)
- Automatic model registry with versioning
- Metadata tracking (features used, params, version)

### **✔ Batch Predictions**
- Builds features for today’s scheduled games
- Generates win probabilities
- Saves versioned predictions + `predictions_latest.parquet`
- Integrated drift monitoring (KS-test)

### **✔ Monitoring**
- Prometheus metrics:
  - prediction runs
  - prediction failures
  - prediction duration
  - drifted features
- Grafana‑ready

### **✔ Streamlit Dashboard**
- Live predictions
- SHAP explainability
- Admin controls (run pipeline, refresh games)

### **✔ Full Automation**
A single orchestrator runs:
```
Ingestion → Feature Engineering → Training → Prediction → Drift Monitoring
```

---

## 📂 Project Structure

```
nba-analytics-v3/
│
├── app.py
├── config.py
│
├── src/
│   ├── ingestion/
│   ├── features/
│   ├── model/
│   ├── monitoring/
│   └── pipeline/
│
├── data/
│   ├── ingestion/
│   ├── features/
│   ├── predictions/
│   ├── models/
│   ├── raw/
│   └── parquet/
│
├── archive/
│   └── unused/
│
├── scripts/
│   └── cleanup_archive.sh
│
├── Makefile
└── README.md
```

---

## 🛠 Installation

### **1. Install dependencies**
```
pip install -r requirements.txt
```

### **2. Install nba_api**
```
pip install nba_api
```

### **3. Run Streamlit dashboard**
```
streamlit run app.py
```

---

## 🔄 Automated Pipeline

The orchestrator handles everything:

```
python -m src.pipeline.orchestrator
```

This will:

1. Start Prometheus metrics server  
2. Run ingestion (full or daily)  
3. Build training features  
4. Train a new model  
5. Predict today’s games  
6. Run drift monitoring  

---

## 📊 Monitoring

Prometheus metrics exposed at:

```
http://localhost:8000
```

Metrics include:

- `nba_predictions_total`
- `nba_prediction_failures_total`
- `nba_prediction_duration_seconds`
- `nba_drift_features_detected`

---

## 🖥 Streamlit Dashboard

Run:

```
streamlit run app.py
```

Tabs include:

- **Live Predictions** (reads `predictions_latest.parquet`)
- **Model Insights** (SHAP summary plot)
- **Admin Center** (run pipeline, refresh games)

---

## 🧹 Cleanup

To archive unused files:

```
bash scripts/cleanup_archive.sh
```

---

## 📄 License

Internal / Private Project (customize as needed)

---

## 🙌 Credits

Built with ❤️ using Python, nba_api, scikit‑learn, Streamlit, Prometheus, and Grafana.

