# Clinical Acceptance Prediction for LLM Outputs

This repository contains three predictive modeling approaches designed to estimate whether clinicians will accept or reject large language model (LLM) responses in real time. The project operates entirely on **synthetic data** that mirrors the structure of the original deployment environment (PHI-protected).

---

## 📂 Repository Structure
├── sample_data.csv # Synthetic dataset used for training + evaluation \
├── scripts/ \
│ ├── logistic_regression.py # Baseline linear model \
│ ├── neural_rasch.py # Feature-based Neural Rasch / IRT-style model \
│ └── bayesian_irt.py # Bayesian IRT-inspired model \
├── requirements.txt # Python dependencies \
└── README.md

---

## 🔧 Installation

To set up the environment:

```bash
pip install -r requirements.txt
```
## ▶️ Running the Models

Each script loads the dataset, trains a model, and prints evaluation metrics.

Example usage:
```bash
python scripts/logistic_regression.py
python scripts/neural_rasch.py
python scripts/bayesian_irt.py
```
Outputs may include:

1. AUROC performance

2. Calibration / Expected Calibration Error (ECE)

3. Diagnostic plots or confusion summaries (if supported in the script)


## 🧠 Data

All data is stored in:
```bash
sample_data.csv
```

This is a synthetic dataset with no real patient data. It includes:

1. Questions for embedding

2. User/provider category metadata

3. Acceptance labels and timestamps

This file can be replaced with your own dataset that follows a similar schema when adapting the models to other settings.

## 🤝 Acknowledgments

This project was originally developed for use in clinical LLM evaluation workflows in secure environments. The released version has been adapted to support open experimentation.
