import numpy as np
import pandas as pd


DIABETES_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
DIABETES_ZERO_AS_NAN_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

HEART_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

PARKINSONS_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
    'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]


# Cross-validated optimal decision threshold from training set (StratifiedKFold on X_train)
DIABETES_TUNED_THRESHOLD = -0.8074


def predict_diabetes(pipeline, cleaned_inputs, threshold=DIABETES_TUNED_THRESHOLD):
    """
    Executes inference for Diabetes using the trained pipeline.
    Ensures physiological zero values are converted to NaN for median imputation.
    Applies the cross-validated decision threshold (-0.8074) to prioritize sensitivity.
    """
    input_df = pd.DataFrame([cleaned_inputs], columns=DIABETES_FEATURES)
    # Convert impossible physiological zero values to NaN for imputer
    input_df[DIABETES_ZERO_AS_NAN_COLS] = input_df[DIABETES_ZERO_AS_NAN_COLS].replace(0, np.nan)

    decision_score = float(pipeline.decision_function(input_df)[0])
    is_positive = bool(decision_score >= threshold)
    pred_label = 1 if is_positive else 0

    return {
        'prediction': pred_label,
        'is_positive': is_positive,
        'decision_score': decision_score,
        'threshold': threshold,
        'label': 'Diabetic' if is_positive else 'Non-Diabetic'
    }


def predict_heart(model, cleaned_inputs):
    """
    Executes inference for Heart Disease using the trained Logistic Regression model.
    """
    input_df = pd.DataFrame([cleaned_inputs], columns=HEART_FEATURES)

    raw_pred = model.predict(input_df)[0]
    is_positive = bool(raw_pred == 1)

    proba = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(input_df)[0]
        proba = {
            'healthy': float(probs[0]),
            'disease': float(probs[1])
        }

    return {
        'prediction': int(raw_pred),
        'is_positive': is_positive,
        'probabilities': proba,
        'label': 'Heart Disease Detected' if is_positive else 'Healthy Cardiovascular Status'
    }


def predict_parkinsons(pipeline, cleaned_inputs):
    """
    Executes inference for Parkinson's Disease using the trained pipeline.
    """
    input_df = pd.DataFrame([cleaned_inputs], columns=PARKINSONS_FEATURES)

    raw_pred = pipeline.predict(input_df)[0]
    decision_score = float(pipeline.decision_function(input_df)[0])
    is_positive = bool(raw_pred == 1)

    return {
        'prediction': int(raw_pred),
        'is_positive': is_positive,
        'decision_score': decision_score,
        'label': "Parkinson's Disease Detected" if is_positive else 'Healthy Control'
    }
