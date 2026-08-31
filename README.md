# AI-Driven Anomaly Detection in Component Burn-In & Screening
**Smart India Hackathon (SIH) 2026**

## Project Overview
This project develops an AI-driven system for anomaly detection, early degradation identification, and quality screening in semiconductor/electronic component burn-in testing. 

Burn-in testing stresses components under elevated temperatures and operating voltages over time points (0h, 24h, 96h, 168h) to trigger infant mortality defects and screen out latent anomalies before field deployment.

---

## Project Structure
```text
sih-burnin-project/
│
├── data/
│   ├── raw/                  # Original synthetic burn-in measurement time-series (immutable)
│   ├── ground_truth/         # True component defect labels and baseline parameters (evaluation only)
│   └── ml_ready/             # Prepared feature matrix with multi-point drift metrics for ML training
│
├── eda/
│   └── outputs/              # Exploratory data analysis scripts, notebooks, plots, and figures
│
├── src/
│   ├── data/                 # Data loading, validation, and preprocessing modules
│   ├── features/             # Feature engineering, drift calculation, and transformation pipelines
│   └── models/               # Anomaly detection, drift classification, and screening model pipelines
│
├── models/                   # Serialized model artifacts, checkpoints, and scalers
│
├── reports/                  # Evaluation summaries, performance metrics, and SIH submission reports
│
├── tests/                    # Unit tests and data validation tests
│
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## Dataset Layout & Guidelines
* **`data/raw/`**: Contains `raw_burnin_data.csv` (40,000 rows across 10,000 components at 0h, 24h, 96h, 168h). This raw dataset is strictly read-only and immutable.
* **`data/ground_truth/`**: Contains `component_ground_truth.csv` (10,000 rows). Used strictly for evaluation/validation benchmarking, not as input features.
* **`data/ml_ready/`**: Contains `ml_features.csv` (10,000 rows × 27 columns). Feature-engineered dataset ready for modeling.  
Neel