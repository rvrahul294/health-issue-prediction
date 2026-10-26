import math

# Field definitions and realistic physiological/biomedical boundaries
DIABETES_FIELD_SPECS = [
    {
        'name': 'Pregnancies',
        'display_name': 'Number of Pregnancies',
        'min_val': 0,
        'max_val': 25,
        'is_integer': True
    },
    {
        'name': 'Glucose',
        'display_name': 'Glucose Level',
        'min_val': 0,      # 0 indicates missing/untested, imputed in pipeline
        'max_val': 500,
        'is_integer': False
    },
    {
        'name': 'BloodPressure',
        'display_name': 'Blood Pressure',
        'min_val': 0,      # 0 indicates missing/untested, imputed in pipeline
        'max_val': 260,
        'is_integer': False
    },
    {
        'name': 'SkinThickness',
        'display_name': 'Skin Thickness',
        'min_val': 0,      # 0 indicates missing/untested, imputed in pipeline
        'max_val': 110,
        'is_integer': False
    },
    {
        'name': 'Insulin',
        'display_name': 'Insulin Level',
        'min_val': 0,      # 0 indicates missing/untested, imputed in pipeline
        'max_val': 1000,
        'is_integer': False
    },
    {
        'name': 'BMI',
        'display_name': 'BMI',
        'min_val': 0,      # 0 indicates missing/untested, imputed in pipeline
        'max_val': 90,
        'is_integer': False
    },
    {
        'name': 'DiabetesPedigreeFunction',
        'display_name': 'Diabetes Pedigree Function',
        'min_val': 0.01,
        'max_val': 3.5,
        'is_integer': False
    },
    {
        'name': 'Age',
        'display_name': 'Age',
        'min_val': 1,
        'max_val': 125,
        'is_integer': True
    }
]

HEART_FIELD_SPECS = [
    {'name': 'age', 'display_name': 'Age', 'min_val': 1, 'max_val': 125, 'is_integer': True},
    {'name': 'sex', 'display_name': 'Sex', 'min_val': 0, 'max_val': 1, 'is_integer': True},
    {'name': 'cp', 'display_name': 'Chest Pain Type', 'min_val': 0, 'max_val': 3, 'is_integer': True},
    {'name': 'trestbps', 'display_name': 'Resting Blood Pressure', 'min_val': 50, 'max_val': 260, 'is_integer': False},
    {'name': 'chol', 'display_name': 'Serum Cholesterol', 'min_val': 80, 'max_val': 650, 'is_integer': False},
    {'name': 'fbs', 'display_name': 'Fasting Blood Sugar', 'min_val': 0, 'max_val': 1, 'is_integer': True},
    {'name': 'restecg', 'display_name': 'Resting ECG', 'min_val': 0, 'max_val': 2, 'is_integer': True},
    {'name': 'thalach', 'display_name': 'Maximum Heart Rate', 'min_val': 50, 'max_val': 250, 'is_integer': False},
    {'name': 'exang', 'display_name': 'Exercise Induced Angina', 'min_val': 0, 'max_val': 1, 'is_integer': True},
    {'name': 'oldpeak', 'display_name': 'ST Depression (oldpeak)', 'min_val': 0.0, 'max_val': 10.0, 'is_integer': False},
    {'name': 'slope', 'display_name': 'Slope of Peak Exercise ST', 'min_val': 0, 'max_val': 2, 'is_integer': True},
    {'name': 'ca', 'display_name': 'Major Vessels Colored by Fluoroscopy', 'min_val': 0, 'max_val': 3, 'is_integer': True},
    {'name': 'thal', 'display_name': 'Thallium Stress Test', 'min_val': 0, 'max_val': 3, 'is_integer': True}
]

PARKINSONS_FIELD_SPECS = [
    {'name': 'MDVP:Fo(Hz)', 'display_name': 'MDVP:Fo (Hz)', 'min_val': 40.0, 'max_val': 400.0},
    {'name': 'MDVP:Fhi(Hz)', 'display_name': 'MDVP:Fhi (Hz)', 'min_val': 50.0, 'max_val': 650.0},
    {'name': 'MDVP:Flo(Hz)', 'display_name': 'MDVP:Flo (Hz)', 'min_val': 30.0, 'max_val': 400.0},
    {'name': 'MDVP:Jitter(%)', 'display_name': 'MDVP:Jitter (%)', 'min_val': 0.0, 'max_val': 0.2},
    {'name': 'MDVP:Jitter(Abs)', 'display_name': 'MDVP:Jitter (Abs)', 'min_val': 0.0, 'max_val': 0.01},
    {'name': 'MDVP:RAP', 'display_name': 'MDVP:RAP', 'min_val': 0.0, 'max_val': 0.2},
    {'name': 'MDVP:PPQ', 'display_name': 'MDVP:PPQ', 'min_val': 0.0, 'max_val': 0.2},
    {'name': 'Jitter:DDP', 'display_name': 'Jitter:DDP', 'min_val': 0.0, 'max_val': 0.5},
    {'name': 'MDVP:Shimmer', 'display_name': 'MDVP:Shimmer', 'min_val': 0.0, 'max_val': 0.5},
    {'name': 'MDVP:Shimmer(dB)', 'display_name': 'MDVP:Shimmer (dB)', 'min_val': 0.0, 'max_val': 3.0},
    {'name': 'Shimmer:APQ3', 'display_name': 'Shimmer:APQ3', 'min_val': 0.0, 'max_val': 0.3},
    {'name': 'Shimmer:APQ5', 'display_name': 'Shimmer:APQ5', 'min_val': 0.0, 'max_val': 0.3},
    {'name': 'MDVP:APQ', 'display_name': 'MDVP:APQ', 'min_val': 0.0, 'max_val': 0.4},
    {'name': 'Shimmer:DDA', 'display_name': 'Shimmer:DDA', 'min_val': 0.0, 'max_val': 0.9},
    {'name': 'NHR', 'display_name': 'NHR', 'min_val': 0.0, 'max_val': 1.0},
    {'name': 'HNR', 'display_name': 'HNR', 'min_val': 0.0, 'max_val': 50.0},
    {'name': 'RPDE', 'display_name': 'RPDE', 'min_val': 0.0, 'max_val': 1.0},
    {'name': 'DFA', 'display_name': 'DFA', 'min_val': 0.0, 'max_val': 1.5},
    {'name': 'spread1', 'display_name': 'spread1', 'min_val': -15.0, 'max_val': 2.0},
    {'name': 'spread2', 'display_name': 'spread2', 'min_val': 0.0, 'max_val': 1.0},
    {'name': 'D2', 'display_name': 'D2', 'min_val': 0.5, 'max_val': 5.0},
    {'name': 'PPE', 'display_name': 'PPE', 'min_val': 0.0, 'max_val': 1.0}
]


def parse_and_validate_inputs(raw_inputs, field_specs):
    """
    Validates and converts a list of raw inputs into floats.
    Checks for:
      1. Empty fields
      2. Valid numeric types
      3. Non-finite values (NaN, +inf, -inf)
      4. Physiological and domain boundaries
    
    Returns:
        tuple: (cleaned_floats, error_message)
               If validation succeeds: (list_of_floats, None)
               If validation fails: (None, error_string)
    """
    if len(raw_inputs) != len(field_specs):
        return None, f"Input count mismatch: expected {len(field_specs)}, got {len(raw_inputs)}"

    # 1. Check for empty fields
    empty_fields = [
        spec['display_name'] for spec, val in zip(field_specs, raw_inputs)
        if val is None or str(val).strip() == ''
    ]
    if empty_fields:
        return None, f"Please fill in all fields. Missing: {', '.join(empty_fields)}"

    cleaned = []
    invalid_format_fields = []
    out_of_bounds_fields = []

    for spec, val in zip(field_specs, raw_inputs):
        display_name = spec['display_name']
        val_str = str(val).strip()

        # Reject literal NaN or Infinity strings
        if val_str.lower() in ('nan', 'inf', '-inf', '+inf', 'infinity', '-infinity'):
            invalid_format_fields.append(f"{display_name} (non-finite value)")
            continue

        try:
            num = float(val_str)
        except (ValueError, TypeError):
            invalid_format_fields.append(f"{display_name} ('{val_str}')")
            continue

        if not math.isfinite(num):
            invalid_format_fields.append(f"{display_name} (non-finite)")
            continue

        # Check bounds
        min_v = spec.get('min_val')
        max_v = spec.get('max_val')
        if min_v is not None and num < min_v:
            out_of_bounds_fields.append(f"{display_name} must be >= {min_v} (received {num})")
        elif max_v is not None and num > max_v:
            out_of_bounds_fields.append(f"{display_name} must be <= {max_v} (received {num})")
        else:
            cleaned.append(num)

    if invalid_format_fields:
        return None, f"Invalid numeric input in: {', '.join(invalid_format_fields)}"

    if out_of_bounds_fields:
        return None, f"Value out of valid range:\n• " + "\n• ".join(out_of_bounds_fields)

    return cleaned, None
