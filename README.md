# Multiple Disease Prediction System

A robust machine learning screening system and interactive web application built with Python, Scikit-Learn, and Streamlit to evaluate the risk of multiple conditions (**Diabetes**, **Heart Disease**, and **Parkinson's Disease**) based on clinical and biomedical parameters.

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-red.svg)](https://streamlit.io/)
[![Test Suite](https://img.shields.io/badge/Tests-17%20Passed-brightgreen.svg)]()

---

## 1. Project Overview & Architecture

The **Multiple Disease Prediction System** is a portfolio machine learning application demonstrating end-to-end data science best practices:
- **Leakage-Free Preprocessing Pipelines**: Integrates missing value imputation, standard scaling, and classification inside Scikit-Learn `Pipeline` architectures to prevent data leakage during train/test splitting and cross-validation.
- **Subject-Aware Group Validation**: Implements `StratifiedGroupKFold` on the Oxford Parkinson's voice dataset to prevent acoustic leakage between phonations from the same individual.
- **Robust Input Validation & Error Handling**: Real-time parsing verifying numeric formatting, non-finite values (`NaN`, `+inf`, `-inf`), and clinical/physiological boundaries.
- **Interactive Multi-Page Streamlit UI**: Modern dark-mode layout with quick-fill documented benchmark cases and instant risk stratification.

---

## 2. Disease Modules & ML Methodologies

| Disease Module | Target Condition | Algorithm & Preprocessing Pipeline | Input Feature Dimension | Primary Evaluation Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Diabetes Prediction** | Diabetes onset ($0=$ Negative, $1=$ Diabetic) | `SimpleImputer(strategy='median')` → `StandardScaler()` → `SVC(kernel='linear')` | 8 diagnostic measures (Glucose, BP, Insulin, BMI, etc.) | Recall / Sensitivity (Screening priority) |
| **Heart Disease Prediction** | Cardiovascular defect ($0=$ Healthy, $1=$ Heart Disease) | `LogisticRegression(max_iter=1000)` on calibrated clinical features | 13 cardiovascular indicators (Chest pain, BP, Chol, ECG, etc.) | ROC-AUC & Well-Calibrated Posterior Probabilities |
| **Parkinson's Prediction** | Parkinson's vocal pathology ($0=$ Healthy, $1=$ Parkinson's) | `StandardScaler()` → `SVC(kernel='linear')` with Subject Group Splitting | 22 vocal acoustic features (Jitter, Shimmer, Fundamental Frequencies, PPE) | Group-Aware Stratified Cross-Validation ROC-AUC |

---

## 3. Verified Model Performance & Benchmarks

All models were evaluated on dedicated held-out test cohorts ($20\%$) and verified through **5-Fold Stratified Cross-Validation**.

| Metric | Diabetes Module (Linear SVC) | Heart Disease Module (Logistic Regression) | Parkinson's Module (Linear SVC) |
| :--- | :--- | :--- | :--- |
| **Training Set Accuracy** | 77.85% | 85.54% | 92.76% |
| **Held-Out Test Accuracy** | **77.27%** | **80.33%** | **79.07%** |
| **Test Precision** | 0.7568 | 0.8182 | 0.8235 |
| **Test Recall (Sensitivity)** | 0.5185 *(Tuned: 0.7222)* | 0.8182 | 0.9032 |
| **Test F1-Score** | 0.6154 *(Tuned: 0.6667)* | 0.8182 | 0.8615 |
| **Test ROC-AUC** | 0.8200 | 0.9037 | 0.5833 *(See Forensic Analysis below)* |
| **5-Fold CV Accuracy** | **76.43% (+/- 2.59%)** | **82.22% (+/- 6.01%)** | **77.97% (+/- 12.89%)** |
| **5-Fold CV ROC-AUC** | **0.8291 (+/- 0.0379)** | **0.8895 (+/- 0.0378)** | **0.7570 (+/- 0.1791)** |

### Advanced Methodological Findings & Statistical Limitations
1. **Diabetes Recall Optimization**: The baseline unweighted Linear SVC produced a test recall of 51.85% (missing 26 of 54 diabetics) due to a 2:1 class imbalance and symmetric margin placement. Using out-of-fold cross-validated decision threshold tuning ($t^* = -0.8074$, derived strictly on `X_train`) increases test recall to **72.22%** (reducing false negatives to 15) with test F1 = 0.6667 and specificity = 76.00%.
2. **Parkinson's Performance Discrepancy & Group Variability (Test ROC-AUC 0.5833 vs CV ROC-AUC 0.7570)**:
   - **Small-Cohort Sample Limitation**: With only 32 total subjects, the 80/20 subject split yields a test set of only 7 individuals (5 Parkinson's, 2 healthy controls). Evaluating rank-order ROC-AUC on a test partition with only $N=2$ negative subjects produces extreme metric volatility.
   - **High Cross-Validation Variance**: In 5-fold `StratifiedGroupKFold`, fold ROC-AUC scores vary widely ($0.7570 \pm 0.1791$, ranging from $0.4567$ to $0.9892$).
   - **Acoustic Hypothesis**: While feature inspection indicates that subject `S43` (a male control with a low fundamental pitch of 114 Hz) was assigned positive decision scores by the linear SVM, the low test ROC-AUC primarily demonstrates the instability of evaluating models on tiny sample sizes. These findings must not be interpreted as evidence of generalizable diagnostic performance.
3. **Heart Disease Probability Calibration & Sample Constraints**: On the held-out test cohort of 61 patients, Logistic Regression achieves a Brier Score Loss of **0.1363** compared to a naive base-rate prevalence baseline of **0.2483** (Brier Skill Score = **0.4511**). While this demonstrates statistical calibration relative to guessing the dataset base rate, it does **not** constitute clinical validation. With only 61 test patients from a single historical dataset (Cleveland, 1988), these probabilities must not be interpreted as clinically reliable risk assessments.

---

## 4. Tech Stack

- **Language**: Python 3.10+ (Tested on Python 3.13)
- **Machine Learning & Preprocessing**: `scikit-learn==1.9.0`, `joblib==1.5.3`
- **Data Manipulation & Computation**: `pandas==3.0.2`, `numpy==2.4.4`
- **Web Interface**: `streamlit==1.56.0`, `streamlit-option-menu==0.4.0`
- **Testing**: Python `unittest` test suite

---

## 5. Repository Structure

```
Health Issue prediction/
├── Dataset/
│   ├── diabetes.csv                    # PIMA Indians Diabetes dataset (768 rows)
│   ├── heart.csv                       # Cleveland Heart Disease dataset (303 rows)
│   └── parkinsons.csv                  # Oxford Parkinson's voice dataset (195 rows, 32 subjects)
├── Notebook/
│   ├── Multiple disease prediction system - diabetes.ipynb      # EDA, Pipeline, CV & threshold analysis
│   ├── Multiple disease prediction system - heart.ipynb         # EDA, Logistic Regression & calibration
│   └── Multiple disease prediction system - Parkinsons.ipynb    # Subject-level splitting & forensic audit
├── saved_models/
│   ├── diabetes_pipeline.joblib        # Preprocessing (Imputer + Scaler) + Linear SVC pipeline
│   ├── heart_disease_model.joblib      # Trained Logistic Regression model
│   ├── heart_disease_model.sav         # Backward-compatible serialized model
│   └── parkinsons_pipeline.joblib      # StandardScaler + Linear SVC pipeline
├── src/
│   ├── __init__.py                     # Package initializer
│   ├── model_loader.py                 # Safe model loading with artifact verification
│   ├── validation.py                   # Strict input validation & physiological bounds
│   └── inference.py                    # Zero-to-NaN conversion & inference dispatchers
├── tests/
│   ├── __init__.py                     # Test package initializer
│   └── test_system.py                  # 17 automated unit and integration tests
├── app.py                              # Streamlit web application
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation and technical report
```

---

## 6. Installation & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/rvrahul294/health-issue-prediction.git
cd "health-issue-prediction"
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Automated Test Suite
```bash
python -m unittest discover tests
```
*Expected output: `Ran 17 tests in ... OK`.*

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Medical Disclaimer & Limitations

> [!CAUTION]
> **Educational Portfolio Project**: This application and its predictive models are designed strictly for **educational, experimental, and portfolio demonstration purposes**.
> - The models have not undergone clinical trials, FDA/CE clearance, or validation on multi-center prospective patient populations.
> - Small dataset sample sizes (e.g., 32 subjects in Parkinson's, 303 in Heart Disease) introduce demographic and acoustic biases.
> - Diagnostic outputs must **never** be used as a substitute for professional clinical judgment, laboratory testing, or medical consultation.