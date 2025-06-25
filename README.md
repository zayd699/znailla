
# 📞 Znailla Bank Term Deposit Prediction

This project uses machine learning to optimize Znailla Bank's marketing campaign by predicting which customers are most likely to subscribe to a term deposit. The goal is to reduce unnecessary call costs and increase campaign profitability.

---

##  Project Overview

* Predict customer subscription likelihood using historical marketing data
* Reduce calls to uninterested customers
* Increase ROI by targeting only high-probability leads

---

## 📁 Structure

```
/               # Root folder
├── notebooks/   # Jupyter notebook pipeline
├── scripts/     # Inference and app scripts
├── models/      # Trained LightGBM model (.pkl)
├── data/        # Original dataset (not pushed)
├── requirements.txt
├── ML_System_Design.md
├── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/zayd699/znailla.git
cd znailla
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run scripts/main.py
```

Upload a CSV file and the app will display customers to contact based on model predictions.

### 4. Run Inference via CLI

```bash
python scripts/inference.py --input new_customers.csv --output contacts.csv
```

---

## � Model Details

* Final Model: LightGBM
* ROC AUC: \~0.9501
* Optimized threshold: 0.06 (for profit)

---

##  Business Impact

* Up to 75% fewer calls
* Retain most actual subscribers
* Significant increase in net profit and marketing efficiency

---

##  ML System Design

See `ML_System_Design.md` for a high-level architecture of the end-to-end deployment and monitoring strategy.

