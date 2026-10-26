# Multiple Disease Prediction System

A robust machine learning screening system and interactive web application built with Python, Scikit-Learn, and Streamlit to evaluate the risk of multiple conditions (**Diabetes**, **Heart Disease**, and **Parkinson's Disease**) based on clinical and biomedical parameters.

---

## Features & Disease Modules

| Disease Module | Machine Learning Model | Key Input Features | Output |
| :---- | :---- | :---- | :---- |
| **Diabetes Prediction** | Support Vector Classifier (`SVC`) with `SimpleImputer` & `StandardScaler` Pipeline | Glucose, Blood Pressure, Insulin, BMI, Age, Pregnancies, Skin Thickness, Diabetes Pedigree Function | Diabetic / Not Diabetic |
| **Heart Disease Prediction** | Logistic Regression | Age, Sex, Chest Pain Type (`cp`), Resting BP (`trestbps`), Cholesterol (`chol`), Fasting Blood Sugar (`fbs`), Resting ECG (`restecg`), Max Heart Rate (`thalach`), Exercise Angina (`exang`), ST Depression (`oldpeak`), Slope, Major Vessels (`ca`), Thalassemia (`thal`) | Heart Disease / No Heart Disease |
| **Parkinson's Disease Prediction** | Support Vector Classifier (`SVC`) | 22 vocal/acoustic signal metrics (Fundamental frequencies `MDVP:Fo/Fhi/Flo`, Jitter metrics, Shimmer metrics, `NHR`, `HNR`, `RPDE`, `DFA`, `spread1/2`, `D2`, `PPE`) | Parkinson's Detected / Healthy |

### Key System Capabilities
- **Pre-trained ML Models & Pipeline**: Integrated end-to-end preprocessing pipeline (`SimpleImputer` + `StandardScaler` + `SVC`) for Diabetes, and serialized classification models for Heart Disease and Parkinson's.
- **Robust Input Validation**: Real-time validation preventing runtime errors on missing or non-numeric inputs.
- **Quick-Fill Sample Data**: Integrated test buttons (`Load Healthy Sample` and `Load High-Risk Sample`) on every module for quick demonstration and testing without manual feature entry.
- **Interactive Multi-Page Interface**: Clean, responsive layout powered by Streamlit and Streamlit Option Menu.

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

> **IMPORTANT**: This application and its underlying machine learning models are designed for **educational, experimental, and portfolio demonstration purposes only**. They do **not** constitute medical advice, diagnosis, or clinical treatment plans. Users must consult licensed medical professionals for any clinical diagnoses and health evaluations.