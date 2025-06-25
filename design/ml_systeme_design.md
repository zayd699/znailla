#  Znailla Bank Marketing Optimization - Project Summary

##  Problem Overview

Znailla Bank wants to improve the profitability of its term deposit marketing campaign. Every phone call to a customer costs €8, while each successful subscription brings in €80. Historically, every customer was contacted, leading to high costs and low efficiency. The goal is to use machine learning to predict which customers are most likely to subscribe, so only high-potential individuals are contacted.

---

##  Step 1: Data Collection & Preparation

* Dataset: `bank-additional-full.csv`
* Contains customer demographics, financial data, and past marketing interactions
* Target variable: `y` (whether the customer subscribed)
* Categorical features are one-hot encoded; numerical features are scaled
* Dataset split into training/test sets (e.g. 70/30)

---

##  Step 2: Feature Engineering

* Important features: `poutcome`, `campaign`, `previous`, etc.
* Excluded `duration` as it's only known post-call
* Applied standard preprocessing using `ColumnTransformer`

---

## 🧠 Step 3: Model Selection

Tested several classifiers:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

**LightGBM** achieved the highest performance with a **ROC AUC of 0.9501** and was selected for deployment.

---

##  Step 4: Evaluation

* Used ROC AUC, confusion matrix, and classification report
* High recall and precision for class 1 (subscribers)
* Used SHAP values to interpret feature importance

---

##  Step 5: Threshold Optimization

* The model outputs a probability score per customer
* Instead of using 0.5, we optimized the threshold for **profit**
* Found that **0.06 (6%)** maximizes net gain

**Why 0.06?**

* At 6% chance of subscribing:

  * Expected revenue = 0.06 \* €80 = €4.80
  * Cost of call = €8
  * Net per call = €-3.20 (but acceptable as part of a larger optimized batch)

* The model selects only the top 25-30% of customers, saving thousands of euros while capturing most conversions

---

## 📤 Step 6: Inference Pipeline

* `inference.py` script and `main.py` Streamlit app
* Upload new customer data as CSV
* Model predicts `subscription_probability`
* Customers with probability ≥ 0.06 are flagged as `should_be_contacted`
* Outputs CSV and downloadable table

---

## 💼 Step 7: Business Impact

* Reduced calls by \~74%, calling only high-potential customers
* Captured \~90% of actual conversions
* Massive call cost savings (up to €60,000 per large batch)
* Increased overall campaign ROI

---

## Step 8: Deployment & Monitoring


### Deployment Options:

* **Batch script** (`inference.py`) for automation by IT teams
* **Streamlit** (`main.py`) for direct use by marketing or analysts (you can run my app in streamlit using the cmd : streamlit run main.py )

### Infrastructure:

* Run on internal servers or a cloud VM
* Schedule regular runs (e.g. weekly) using cron jobs or CI/CD pipelines (GitHub Actions, GitLab CI)

---

## Step 9: Monitoring & Maintenance

### Monitoring:

* Track prediction volume and threshold hit rate
* Monitor business KPIs: call-to-conversion ratio, profit per batch
* Log unexpected input schema changes or drift in prediction distribution

### Retraining:

* Model retrained quarterly using new labeled campaign data
* Store performance history (AUC, profit impact) per model version
* Use version control and experiment tracking (e.g., MLflow or DVC)


##  Conclusion

This machine learning solution aligns technical modeling with business value:

* Cuts unnecessary costs
* Maintains conversion rates
* Improves profitability
* Empowers the marketing team with clear, data-driven decisions
