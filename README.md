# ParkSmart — Smart Parking Operations

AI-based smart parking availability prediction and anomaly detection with a live Streamlit dashboard.

- 100 bays across Zone A (40), Zone B (30), Zone C (30)
- Software-simulated IoT sensors (no physical hardware needed)
- Linear Regression forecast 30 minutes ahead (R² ≈ 0.97, MAE ≈ 1.6 bays)
- Z-Score anomaly detection (Critical / Warning / Informational)
- Live dashboard refreshing every 5 seconds

## Run locally

```bash
pip install -r requirements.txt
python simulation/simulator.py   # generate dataset (first time only)
python ml/train_model.py         # train model (first time only)
python anomaly/detector.py       # detect anomalies (first time only)
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

Main file: `app.py`. Data (`data/`) and model (`models/`) files are committed, so no build step is needed.
