import streamlit as st
from streamlit_option_menu import option_menu

from src.model_loader import load_all_models, ModelLoadError
from src.validation import (
    parse_and_validate_inputs,
    DIABETES_FIELD_SPECS,
    HEART_FIELD_SPECS,
    PARKINSONS_FIELD_SPECS
)
from src.inference import (
    predict_diabetes,
    predict_heart,
    predict_parkinsons
)


st.set_page_config(
    page_title="Multiple Disease Prediction System",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Custom CSS for enhanced visual hierarchy, cards, and clean typography
st.markdown("""
<style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Sidebar Scrolling & Top Clean Start */
    [data-testid="stSidebar"] {
        top: 0;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
    }
    
    .module-desc {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }
    
    /* Quick-Fill Sample Box & Buttons */
    .sample-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 1.5rem;
    }
    .sample-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        white-space: nowrap !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 0.75rem !important;
        min-height: 2.4rem !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Form Section Headers */
    .form-section-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        letter-spacing: 0.015em;
    }
    
    /* Balanced Result Cards */
    .result-card {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .result-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(249, 115, 22, 0.08) 100%);
        border: 1.5px solid rgba(239, 68, 68, 0.45);
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(59, 130, 246, 0.08) 100%);
        border: 1.5px solid rgba(16, 185, 129, 0.45);
    }
    .result-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .result-icon {
        font-size: 1.35rem;
        line-height: 1;
    }
    .result-title {
        margin: 0;
        font-size: 0.98rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .result-positive .result-title {
        color: #f87171;
    }
    .result-negative .result-title {
        color: #34d399;
    }
    .result-subtitle {
        margin: 0.15rem 0 0 0;
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.35;
    }

    /* Sidebar Styling & Models & Pipelines */
    .sidebar-arch-card {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin-top: 0.35rem;
        width: 100%;
    }
    .arch-item {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 0.5rem 0.65rem;
        width: 100%;
        box-sizing: border-box;
        overflow: visible;
    }
    .arch-module {
        display: block;
        font-weight: 600;
        font-size: 0.82rem;
        color: #f1f5f9;
        margin-bottom: 0.2rem;
        white-space: normal;
        overflow: visible;
        word-break: break-word;
        line-height: 1.3;
    }
    .arch-pipeline {
        display: block;
        font-size: 0.72rem;
        color: #94a3b8;
        line-height: 1.35;
        white-space: normal;
        overflow: visible;
        word-break: break-word;
    }
    .sidebar-meta {
        font-size: 0.82rem;
        color: #94a3b8;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 1.5rem;
    }
    .sidebar-disclaimer {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.78rem;
        color: #fbbf24;
        line-height: 1.4;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Safe model loading with caching and error handling
@st.cache_resource
def get_cached_models():
    try:
        return load_all_models()
    except ModelLoadError as e:
        st.error(f"⚠️ Application Model Error: {e}")
        st.stop()


diabetes_model, heart_disease_model, parkinsons_model = get_cached_models()


# Sidebar Navigation & Information
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

    st.markdown("---")
    st.markdown("### 🤖 Models & Pipelines")
    st.markdown("""
    <div class="sidebar-arch-card">
        <div class="arch-item">
            <span class="arch-module">Diabetes</span>
            <span class="arch-pipeline">SimpleImputer → StandardScaler → SVC</span>
        </div>
        <div class="arch-item">
            <span class="arch-module">Heart Disease</span>
            <span class="arch-pipeline">Logistic Regression</span>
        </div>
        <div class="arch-item">
            <span class="arch-module">Parkinson's</span>
            <span class="arch-pipeline">StandardScaler → SVC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔗 Project Resources")
    st.markdown("""
    - [GitHub Repository](https://github.com/rvrahul294/health-issue-prediction)
    - [Project Documentation](https://github.com/rvrahul294/health-issue-prediction#readme)
    """)

    st.markdown("""
    <div class="sidebar-disclaimer">
        <strong>⚠️ Educational Disclaimer</strong><br>
        This application and its underlying machine learning models are created for portfolio demonstration and educational purposes only. They do not constitute clinical diagnoses or medical advice.
    </div>
    """, unsafe_allow_html=True)


# Helper callback to populate input fields with sample values
def load_sample_values(sample_dict):
    for key, value in sample_dict.items():
        st.session_state[key] = value


# Realistic Sample Data from Datasets
SAMPLES = {
    'diabetes_healthy': {
        'diab_pregnancies': '1',
        'diab_glucose': '85',
        'diab_bp': '66',
        'diab_skin': '29',
        'diab_insulin': '26',
        'diab_bmi': '26.6',
        'diab_dpf': '0.351',
        'diab_age': '31'
    },
    'diabetes_risk': {
        'diab_pregnancies': '6',
        'diab_glucose': '148',
        'diab_bp': '72',
        'diab_skin': '35',
        'diab_insulin': '168',
        'diab_bmi': '33.6',
        'diab_dpf': '0.627',
        'diab_age': '50'
    },
    'heart_healthy': {
        # Documented example from heart.ipynb (row 167 in Dataset/heart.csv, target=0)
        'heart_age': '62',
        'heart_sex': 0,
        'heart_cp': 0,
        'heart_trestbps': '140',
        'heart_chol': '268',
        'heart_fbs': 0,
        'heart_restecg': 0,
        'heart_thalach': '160',
        'heart_exang': 0,
        'heart_oldpeak': '3.6',
        'heart_slope': 0,
        'heart_ca': 2,
        'heart_thal': 2
    },
    'heart_risk': {
        # Documented example from Dataset/heart.csv row 0 (target=1)
        'heart_age': '63',
        'heart_sex': 1,
        'heart_cp': 3,
        'heart_trestbps': '145',
        'heart_chol': '233',
        'heart_fbs': 1,
        'heart_restecg': 0,
        'heart_thalach': '150',
        'heart_exang': 0,
        'heart_oldpeak': '2.3',
        'heart_slope': 0,
        'heart_ca': 0,
        'heart_thal': 1
    },
    'parkinsons_healthy': {
        'park_fo': '197.076',
        'park_fhi': '206.896',
        'park_flo': '192.055',
        'park_jitter_pct': '0.00289',
        'park_jitter_abs': '0.00001',
        'park_rap': '0.00166',
        'park_ppq': '0.00168',
        'park_ddp': '0.00498',
        'park_shimmer': '0.01098',
        'park_shimmer_db': '0.097',
        'park_apq3': '0.00563',
        'park_apq5': '0.0068',
        'park_apq': '0.00802',
        'park_dda': '0.01689',
        'park_nhr': '0.00339',
        'park_hnr': '26.775',
        'park_rpde': '0.422229',
        'park_dfa': '0.741367',
        'park_spread1': '-7.3483',
        'park_spread2': '0.177551',
        'park_d2': '1.743867',
        'park_ppe': '0.085569'
    },
    'parkinsons_risk': {
        'park_fo': '119.992',
        'park_fhi': '157.302',
        'park_flo': '74.997',
        'park_jitter_pct': '0.00784',
        'park_jitter_abs': '0.00007',
        'park_rap': '0.0037',
        'park_ppq': '0.00554',
        'park_ddp': '0.01109',
        'park_shimmer': '0.04374',
        'park_shimmer_db': '0.426',
        'park_apq3': '0.02182',
        'park_apq5': '0.0313',
        'park_apq': '0.02971',
        'park_dda': '0.06545',
        'park_nhr': '0.02211',
        'park_hnr': '21.033',
        'park_rpde': '0.414783',
        'park_dfa': '0.815285',
        'park_spread1': '-4.813031',
        'park_spread2': '0.266482',
        'park_d2': '2.301442',
        'park_ppe': '0.284654'
    }
}


# ==========================================
# 1. Diabetes Prediction Page
# ==========================================
if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction')
    st.markdown(
        '<p class="module-desc">Predicts diabetes risk using physiological diagnostic parameters trained on the PIMA Indians Diabetes dataset with an integrated Support Vector Machine pipeline.</p>',
        unsafe_allow_html=True
    )

    col_btn1, col_btn2, _ = st.columns([1.8, 1.8, 3.4])
    with col_btn1:
        st.button('📋 Load Healthy Sample', on_click=load_sample_values, args=(SAMPLES['diabetes_healthy'],), key='btn_diab_healthy', use_container_width=True)
    with col_btn2:
        st.button('⚠️ Load High-Risk Sample', on_click=load_sample_values, args=(SAMPLES['diabetes_risk'],), key='btn_diab_risk', use_container_width=True)

    with st.form(key='diabetes_form', border=True):
        st.markdown('<div class="form-section-header">Patient Diagnostic Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies', key='diab_pregnancies', help="Total number of pregnancies")
            SkinThickness = st.text_input('Skin Thickness (mm)', key='diab_skin', help="Triceps skin fold thickness in mm (enter 0 if unmeasured)")
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', key='diab_dpf', help="Genetic diabetes pedigree function score")

        with col2:
            Glucose = st.text_input('Glucose Level (mg/dL)', key='diab_glucose', help="Plasma glucose concentration (2 hours in oral glucose test)")
            Insulin = st.text_input('Insulin Level (μU/mL)', key='diab_insulin', help="2-Hour serum insulin (enter 0 if unmeasured)")
            Age = st.text_input('Age (years)', key='diab_age', help="Patient age in years")

        with col3:
            BloodPressure = st.text_input('Blood Pressure (mm Hg)', key='diab_bp', help="Diastolic blood pressure in mm Hg (enter 0 if unmeasured)")
            BMI = st.text_input('BMI (kg/m²)', key='diab_bmi', help="Body mass index = weight in kg / (height in m)^2 (enter 0 if unmeasured)")

        submitted = st.form_submit_button('Run Diabetes Assessment', use_container_width=True)

    if submitted:
        raw_inputs = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]

        cleaned_inputs, error_msg = parse_and_validate_inputs(raw_inputs, DIABETES_FIELD_SPECS)

        if error_msg:
            st.error(error_msg)
        else:
            result = predict_diabetes(diabetes_model, cleaned_inputs)

            if result['is_positive']:
                st.markdown("""
                <div class="result-card result-positive">
                    <div class="result-header">
                        <span class="result-icon">⚠️</span>
                        <div>
                            <h3 class="result-title">Elevated Risk: Diabetic Result</h3>
                            <p class="result-subtitle">The model predicts that the diagnostic measurements are consistent with diabetes.</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card result-negative">
                    <div class="result-header">
                        <span class="result-icon">✅</span>
                        <div>
                            <h3 class="result-title">Low Risk: Non-Diabetic Result</h3>
                            <p class="result-subtitle">The model predicts that the diagnostic measurements are consistent with a non-diabetic profile.</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==========================================
# 2. Heart Disease Prediction Page
# ==========================================
if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction')
    st.markdown(
        '<p class="module-desc">Evaluates cardiovascular disease risk using 13 clinical features trained on the Cleveland Heart Disease dataset with Logistic Regression.</p>',
        unsafe_allow_html=True
    )

    col_btn1, col_btn2, _ = st.columns([1.8, 1.8, 3.4])
    with col_btn1:
        st.button('📋 Load Healthy Sample', on_click=load_sample_values, args=(SAMPLES['heart_healthy'],), key='btn_heart_healthy', use_container_width=True)
    with col_btn2:
        st.button('⚠️ Load High-Risk Sample', on_click=load_sample_values, args=(SAMPLES['heart_risk'],), key='btn_heart_risk', use_container_width=True)

    with st.form(key='heart_form', border=True):
        st.markdown('<div class="form-section-header">1. Patient Profile & Symptoms</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age (years)', key='heart_age', help="Patient age in years")

        with col2:
            sex = st.selectbox(
                'Sex',
                options=[1, 0],
                format_func=lambda x: "Male (1)" if x == 1 else "Female (0)",
                key='heart_sex',
                help="Biological sex"
            )

        with col3:
            cp = st.selectbox(
                'Chest Pain Type',
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0: Typical Angina",
                    1: "1: Atypical Angina",
                    2: "2: Non-anginal Pain",
                    3: "3: Asymptomatic"
                }.get(x, str(x)),
                key='heart_cp',
                help="Chest pain category (0 to 3)"
            )

        st.markdown('<div class="form-section-header">2. Resting Measurements & Blood Tests</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)

        with col4:
            trestbps = st.text_input('Resting Blood Pressure (mm Hg)', key='heart_trestbps', help="Resting BP in mm Hg upon hospital admission")

        with col5:
            chol = st.text_input('Serum Cholesterol (mg/dL)', key='heart_chol', help="Serum cholesterol in mg/dL")

        with col6:
            fbs = st.selectbox(
                'Fasting Blood Sugar > 120 mg/dL',
                options=[0, 1],
                format_func=lambda x: "0: False (≤ 120 mg/dL)" if x == 0 else "1: True (> 120 mg/dL)",
                key='heart_fbs',
                help="Fasting blood sugar > 120 mg/dL"
            )

        st.markdown('<div class="form-section-header">3. Electrocardiogram & Exercise Stress Test</div>', unsafe_allow_html=True)
        col7, col8, col9 = st.columns(3)

        with col7:
            restecg = st.selectbox(
                'Resting ECG Results',
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0: Normal",
                    1: "1: ST-T Wave Abnormality",
                    2: "2: Left Ventricular Hypertrophy"
                }.get(x, str(x)),
                key='heart_restecg',
                help="Resting electrocardiographic results"
            )

        with col8:
            thalach = st.text_input('Maximum Heart Rate Achieved', key='heart_thalach', help="Max heart rate reached during exercise test")

        with col9:
            exang = st.selectbox(
                'Exercise Induced Angina',
                options=[0, 1],
                format_func=lambda x: "0: No" if x == 0 else "1: Yes",
                key='heart_exang',
                help="Exercise-induced angina status"
            )

        st.markdown('<div class="form-section-header">4. Fluoroscopy & Thallium Diagnostics</div>', unsafe_allow_html=True)
        col10, col11, col12 = st.columns(3)

        with col10:
            oldpeak = st.text_input('ST Depression (oldpeak)', key='heart_oldpeak', help="ST depression induced by exercise relative to rest")

        with col11:
            slope = st.selectbox(
                'Slope of Peak Exercise ST',
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0: Upsloping",
                    1: "1: Flat",
                    2: "2: Downsloping"
                }.get(x, str(x)),
                key='heart_slope',
                help="Slope of peak exercise ST segment"
            )

        with col12:
            ca = st.selectbox(
                'Major Vessels Colored by Fluoroscopy',
                options=[0, 1, 2, 3],
                format_func=lambda x: f"{x} Major Vessel(s)" if x > 0 else "0: None",
                key='heart_ca',
                help="Number of major vessels (0-3) colored by fluoroscopy"
            )

        col13, _, _ = st.columns(3)
        with col13:
            thal = st.selectbox(
                'Thallium Stress Test (Thal)',
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0: Unspecified / Missing",
                    1: "1: Fixed Defect",
                    2: "2: Normal Blood Flow",
                    3: "3: Reversible Defect"
                }.get(x, str(x)),
                key='heart_thal',
                help="Thallium scintigraphy stress test defect classification"
            )

        submitted = st.form_submit_button('Run Heart Disease Assessment', use_container_width=True)

    if submitted:
        raw_inputs = [
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]

        cleaned_inputs, error_msg = parse_and_validate_inputs(raw_inputs, HEART_FIELD_SPECS)

        if error_msg:
            st.error(error_msg)
        else:
            result = predict_heart(heart_disease_model, cleaned_inputs)
            proba_info = ""
            if result.get('probabilities'):
                p_dis = result['probabilities']['disease'] * 100
                p_hea = result['probabilities']['healthy'] * 100
                proba_info = f"<p class='result-subtitle' style='margin-top: 4px; font-weight: 500;'>Model Confidence: Disease Risk {p_dis:.1f}% | Healthy Likelihood {p_hea:.1f}%</p>"

            if result['is_positive']:
                st.markdown(f"""
                <div class="result-card result-positive">
                    <div class="result-header">
                        <span class="result-icon">⚠️</span>
                        <div>
                            <h3 class="result-title">Elevated Risk: Heart Disease Detected</h3>
                            <p class="result-subtitle">The model predicts that the diagnostic profile is consistent with the presence of heart disease.</p>
                            {proba_info}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-negative">
                    <div class="result-header">
                        <span class="result-icon">✅</span>
                        <div>
                            <h3 class="result-title">Low Risk: Healthy Cardiovascular Result</h3>
                            <p class="result-subtitle">The model predicts that the diagnostic profile is consistent with a healthy cardiovascular status.</p>
                            {proba_info}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==========================================
# 3. Parkinson's Prediction Page
# ==========================================
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction")
    st.markdown(
        '<p class="module-desc">Analyzes 22 biomedical voice acoustic measurements using a Support Vector Machine pipeline trained on the Oxford Parkinson\'s disease voice dataset.</p>',
        unsafe_allow_html=True
    )

    col_btn1, col_btn2, _ = st.columns([1.8, 1.8, 3.4])
    with col_btn1:
        st.button('📋 Load Healthy Sample', on_click=load_sample_values, args=(SAMPLES['parkinsons_healthy'],), key='btn_park_healthy', use_container_width=True)
    with col_btn2:
        st.button('⚠️ Load High-Risk Sample', on_click=load_sample_values, args=(SAMPLES['parkinsons_risk'],), key='btn_park_risk', use_container_width=True)

    with st.form(key='parkinsons_form', border=True):
        st.markdown('<div class="form-section-header">1. Fundamental Frequency Measures (Vocal Pitch)</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = st.text_input('MDVP:Fo (Hz)', key='park_fo', help="Average vocal fundamental frequency")
        with col2:
            fhi = st.text_input('MDVP:Fhi (Hz)', key='park_fhi', help="Maximum vocal fundamental frequency")
        with col3:
            flo = st.text_input('MDVP:Flo (Hz)', key='park_flo', help="Minimum vocal fundamental frequency")

        st.markdown('<div class="form-section-header">2. Jitter Metrics (Frequency Variation)</div>', unsafe_allow_html=True)
        col4, col5, col6, col7, col8 = st.columns(5)
        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter (%)', key='park_jitter_pct', help="Percentage variation in fundamental frequency")
        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter (Abs)', key='park_jitter_abs', help="Absolute jitter in microseconds")
        with col6:
            RAP = st.text_input('MDVP:RAP', key='park_rap', help="Relative amplitude perturbation")
        with col7:
            PPQ = st.text_input('MDVP:PPQ', key='park_ppq', help="Five-point period perturbation quotient")
        with col8:
            DDP = st.text_input('Jitter:DDP', key='park_ddp', help="Average absolute difference of differences between jitter cycles")

        st.markdown('<div class="form-section-header">3. Shimmer Metrics (Amplitude Variation)</div>', unsafe_allow_html=True)
        col9, col10, col11 = st.columns(3)
        with col9:
            Shimmer = st.text_input('MDVP:Shimmer', key='park_shimmer', help="Local shimmer variation in amplitude")
            Shimmer_dB = st.text_input('MDVP:Shimmer (dB)', key='park_shimmer_db', help="Local shimmer variation in decibels")
        with col10:
            APQ3 = st.text_input('Shimmer:APQ3', key='park_apq3', help="Three-point amplitude perturbation quotient")
            APQ5 = st.text_input('Shimmer:APQ5', key='park_apq5', help="Five-point amplitude perturbation quotient")
        with col11:
            APQ = st.text_input('MDVP:APQ', key='park_apq', help="11-point amplitude perturbation quotient")
            DDA = st.text_input('Shimmer:DDA', key='park_dda', help="Average absolute difference between consecutive amplitude differences")

        st.markdown('<div class="form-section-header">4. Harmonicity & Noise Measures</div>', unsafe_allow_html=True)
        col12, col13 = st.columns(2)
        with col12:
            NHR = st.text_input('NHR (Noise-to-Harmonics)', key='park_nhr', help="Ratio of noise to tonal components in the voice")
        with col13:
            HNR = st.text_input('HNR (Harmonics-to-Noise)', key='park_hnr', help="Ratio of tonal to noise components in the voice")

        st.markdown('<div class="form-section-header">5. Nonlinear Dynamical & Complexity Features</div>', unsafe_allow_html=True)
        col14, col15, col16 = st.columns(3)
        with col14:
            RPDE = st.text_input('RPDE', key='park_rpde', help="Recurrence period density entropy")
            DFA = st.text_input('DFA', key='park_dfa', help="Detrended fluctuation analysis")
        with col15:
            spread1 = st.text_input('spread1', key='park_spread1', help="Nonlinear fundamental frequency variation parameter 1")
            spread2 = st.text_input('spread2', key='park_spread2', help="Nonlinear fundamental frequency variation parameter 2")
        with col16:
            D2 = st.text_input('D2', key='park_d2', help="Correlation dimension")
            PPE = st.text_input('PPE', key='park_ppe', help="Pitch period entropy")

        submitted = st.form_submit_button("Run Parkinson's Assessment", use_container_width=True)

    if submitted:
        raw_inputs = [
            fo, fhi, flo, Jitter_percent, Jitter_Abs,
            RAP, PPQ, DDP, Shimmer, Shimmer_dB,
            APQ3, APQ5, APQ, DDA, NHR,
            HNR, RPDE, DFA, spread1, spread2, D2, PPE
        ]

        cleaned_inputs, error_msg = parse_and_validate_inputs(raw_inputs, PARKINSONS_FIELD_SPECS)

        if error_msg:
            st.error(error_msg)
        else:
            result = predict_parkinsons(parkinsons_model, cleaned_inputs)

            if result['is_positive']:
                st.markdown("""
                <div class="result-card result-positive">
                    <div class="result-header">
                        <span class="result-icon">⚠️</span>
                        <div>
                            <h3 class="result-title">Elevated Risk: Parkinson's Disease Detected</h3>
                            <p class="result-subtitle">The model predicts that the acoustic vocal measurements are consistent with Parkinson's disease.</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card result-negative">
                    <div class="result-header">
                        <span class="result-icon">✅</span>
                        <div>
                            <h3 class="result-title">Low Risk: Healthy Control Result</h3>
                            <p class="result-subtitle">The model predicts that the acoustic vocal measurements are consistent with a healthy control profile.</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)