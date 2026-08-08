# Credit Risk Model Exploration

A machine learning project focused on predicting credit risk defaults using historical application and credit bureau data. This repository features data preprocessing pipelines, exploratory data analysis via Jupyter notebooks, and a deployment-ready Python application script.

## 📁 Repository Structure

```text
├── Credit_Risk_Model.ipynb   # Exploratory Data Analysis & Model Training Pipeline
├── app.py                    # Production inference / Application entry point
├── requirements.txt          # Verified project dependencies
└── .gitignore                # Safeguards ignoring bulky assets (.venv, raw data)
```

## 🛠️ Installation & Environment Setup

This project uses a dedicated Python virtual environment to manage dependencies securely without bloating the repository. Follow these steps to set up the runtime locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chhajerlokesh/codealpha_tasks.git
   cd codealpha_tasks
   ```

2. **Recreate the virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install exact dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Pipeline Overview

### 1. Data Cleaning & Feature Engineering
* Handled missing value imputations and categorical variable scaling.
* Structured internal evaluation metrics out of heavy application tracking logs.
* Localized raw source data arrays (`application_record.csv`, `credit_record.csv`) are strictly ignored locally by Git configuration policies to keep source trees clean.

### 2. Model Exploration (`Credit_Risk_Model.ipynb`)
* Deep dive performance scaling tracking across tabular datasets.
* Implements rigorous evaluation metrics (Precision, Recall, ROC-AUC) critical for sensitive credit banking constraints.

### 3. Production Serving (`app.py`)
* Dedicated deployment script structurally optimized to ingest new credit inquiries and produce risk flags on demand.

## 🛑 Data Notice
The underlying files `application_record.csv` and `credit_record.csv` exceed structural storage thresholds recommended for direct source code hosting. Please provision these datasets locally within your project root when reproducing the calculations.
