# Parkinson's Disease Severity Prediction
### Hybrid Ensemble ML with Explainable AI (SHAP)

**Author:** Archit Sanoria | 2210991334 | B.E. CSE, Chitkara University  
**Mentor:** Mr. Ajay Katiyar  
**Dataset:** Oxford Parkinson's Disease Telemonitoring (UCI) — 5,875 records, 42 subjects

---

## What This Project Does

Predicts Parkinson's disease severity (UPDRS score) from voice recordings using machine learning — without any data leakage. Unlike most similar studies, this project uses a **strict subject-wise split** so the model is tested on completely unseen patients.

---

## Key Results

| Model | Test R² | MAE |
|-------|---------|-----|
| **XGBoost** | **0.8767** | **2.34 pts** |
| SVR | 0.7850 | 3.23 pts |
| Random Forest | 0.7798 | 3.13 pts |
| Stacking Ensemble | 0.7346 | 3.44 pts |
| MLP | 0.6024 | 4.42 pts |

---

## Project Structure

```
PD_Final_Project/
├── data/
│   ├── parkinsons_updrs.csv        ← Raw dataset
│   └── clean_parkinsons.csv        ← Preprocessed (auto-generated)
├── notebooks/
│   ├── 01_Research_Objective.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Individual_Models.ipynb
│   ├── 05_Stacking_Ensemble.ipynb
│   ├── 06_SHAP_XAI.ipynb
│   └── 07_Literature_Comparison.ipynb
├── results/                        ← Auto-generated CSVs
├── figures/                        ← Auto-generated plots
└── requirements.txt
```

---

## How to Run

### Step 1 — Install Python 3.8+
Download from [python.org](https://www.python.org) if not already installed.

### Step 2 — Install dependencies
Open terminal in the `PD_Final_Project/` folder and run:
```bash
pip install -r requirements.txt
```

### Step 3 — Launch Jupyter
```bash
jupyter notebook
```

### Step 4 — Run notebooks IN ORDER

> ⚠️ Each notebook saves outputs used by the next one. Do not skip.

| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `01_Research_Objective.ipynb` | Read only — no execution needed |
| 02 | `02_Data_Preprocessing.ipynb` | Cleans data, saves `clean_parkinsons.csv` |
| 03 | `03_EDA.ipynb` | Generates correlation heatmap and feature plots |
| 04 | `04_Individual_Models.ipynb` | Trains all 4 models, runs GroupKFold CV |
| 05 | `05_Stacking_Ensemble.ipynb` | Builds the stacking ensemble |
| 06 | `06_SHAP_XAI.ipynb` | Generates SHAP explainability plots |
| 07 | `07_Literature_Comparison.ipynb` | Final comparison with prior studies |

In each notebook: **Kernel → Restart & Run All**

---

## Dependencies

```
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost, shap, scipy, jupyter
```

All installed automatically via `pip install -r requirements.txt`

---

## Output

After running all notebooks:
- **20 figures** saved in `figures/`
- **8 result CSVs** saved in `results/`
- Final model: **XGBoost R² = 0.8767** on 9 completely unseen patients

---


