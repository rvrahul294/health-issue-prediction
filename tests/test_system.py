import os
import math
import unittest
import numpy as np
import pandas as pd

from src.model_loader import load_all_models, ModelLoadError, get_default_models_dir
from src.validation import (
    parse_and_validate_inputs,
    DIABETES_FIELD_SPECS,
    HEART_FIELD_SPECS,
    PARKINSONS_FIELD_SPECS
)
from src.inference import (
    predict_diabetes,
    predict_heart,
    predict_parkinsons,
    DIABETES_FEATURES,
    HEART_FEATURES,
    PARKINSONS_FEATURES,
    DIABETES_ZERO_AS_NAN_COLS
)


class TestModelArtifactLoading(unittest.TestCase):
    """Verifies that all serialized models and pipelines load correctly with expected steps."""

    @classmethod
    def setUpClass(cls):
        cls.models_dir = get_default_models_dir()
        cls.diabetes_model, cls.heart_model, cls.parkinsons_model = load_all_models(cls.models_dir)

    def test_artifacts_exist(self):
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, 'diabetes_pipeline.joblib')))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, 'parkinsons_pipeline.joblib')))
        heart_joblib = os.path.exists(os.path.join(self.models_dir, 'heart_disease_model.joblib'))
        heart_sav = os.path.exists(os.path.join(self.models_dir, 'heart_disease_model.sav'))
        self.assertTrue(heart_joblib or heart_sav)

    def test_diabetes_pipeline_structure(self):
        named_steps = dict(self.diabetes_model.steps)
        self.assertIn('imputer', named_steps)
        self.assertIn('scaler', named_steps)
        self.assertIn('classifier', named_steps)
        # Verify imputer was fitted with 8 features
        self.assertEqual(len(named_steps['imputer'].statistics_), 8)

    def test_heart_model_capabilities(self):
        self.assertTrue(hasattr(self.heart_model, 'predict'))
        self.assertTrue(hasattr(self.heart_model, 'predict_proba'))
        # 13 coefficients for 13 features
        self.assertEqual(self.heart_model.coef_.shape[1], 13)

    def test_parkinsons_pipeline_structure(self):
        named_steps = dict(self.parkinsons_model.steps)
        self.assertIn('scaler', named_steps)
        self.assertIn('classifier', named_steps)
        # Scaler fitted with 22 features
        self.assertEqual(len(named_steps['scaler'].mean_), 22)

    def test_missing_model_directory_raises_error(self):
        fake_dir = os.path.join(self.models_dir, 'non_existent_folder')
        with self.assertRaises(ModelLoadError):
            load_all_models(fake_dir)


class TestFeatureOrdering(unittest.TestCase):
    """Verifies that inference feature column lists match training datasets exactly."""

    def test_diabetes_feature_order(self):
        df = pd.read_csv('Dataset/diabetes.csv')
        expected_cols = [c for c in df.columns if c != 'Outcome']
        self.assertEqual(DIABETES_FEATURES, expected_cols)

    def test_heart_feature_order(self):
        df = pd.read_csv('Dataset/heart.csv')
        expected_cols = [c for c in df.columns if c != 'target']
        self.assertEqual(HEART_FEATURES, expected_cols)

    def test_parkinsons_feature_order(self):
        df = pd.read_csv('Dataset/parkinsons.csv')
        expected_cols = [c for c in df.columns if c not in ('name', 'status')]
        self.assertEqual(PARKINSONS_FEATURES, expected_cols)


class TestInputValidation(unittest.TestCase):
    """Tests rigorous validation: missing fields, invalid formats, non-finite values, and bounds."""

    def test_valid_diabetes_input_passes(self):
        valid_input = ['1', '85', '66', '29', '26', '26.6', '0.351', '31']
        cleaned, err = parse_and_validate_inputs(valid_input, DIABETES_FIELD_SPECS)
        self.assertIsNone(err)
        self.assertEqual(len(cleaned), 8)
        self.assertAlmostEqual(cleaned[1], 85.0)

    def test_empty_input_rejected(self):
        empty_input = ['1', '', '66', '29', '26', '26.6', '0.351', '31']
        cleaned, err = parse_and_validate_inputs(empty_input, DIABETES_FIELD_SPECS)
        self.assertIsNone(cleaned)
        self.assertIn("Please fill in all fields", err)
        self.assertIn("Glucose Level", err)

    def test_non_numeric_input_rejected(self):
        invalid_input = ['1', 'glucose_abc', '66', '29', '26', '26.6', '0.351', '31']
        cleaned, err = parse_and_validate_inputs(invalid_input, DIABETES_FIELD_SPECS)
        self.assertIsNone(cleaned)
        self.assertIn("Invalid numeric input", err)
        self.assertIn("glucose_abc", err)

    def test_non_finite_inputs_rejected(self):
        for bad_val in ['nan', 'inf', '-inf', 'NaN', 'Infinity']:
            raw_input = ['1', bad_val, '66', '29', '26', '26.6', '0.351', '31']
            cleaned, err = parse_and_validate_inputs(raw_input, DIABETES_FIELD_SPECS)
            self.assertIsNone(cleaned)
            self.assertIn("non-finite", err.lower())

    def test_out_of_bounds_rejected(self):
        # Negative age
        bad_age = ['1', '85', '66', '29', '26', '26.6', '0.351', '-5']
        cleaned, err = parse_and_validate_inputs(bad_age, DIABETES_FIELD_SPECS)
        self.assertIsNone(cleaned)
        self.assertIn("Age must be >= 1", err)

        # Heart resting BP > 260
        bad_bp = ['62', '0', '0', '350', '268', '0', '0', '160', '0', '3.6', '0', '2', '2']
        cleaned, err = parse_and_validate_inputs(bad_bp, HEART_FIELD_SPECS)
        self.assertIsNone(cleaned)
        self.assertIn("Resting Blood Pressure must be <= 260", err)


class TestSuccessfulPredictions(unittest.TestCase):
    """Verifies inference outputs against documented dataset benchmark samples."""

    @classmethod
    def setUpClass(cls):
        cls.models_dir = get_default_models_dir()
        cls.diabetes_model, cls.heart_model, cls.parkinsons_model = load_all_models(cls.models_dir)

    def test_diabetes_predictions(self):
        # Healthy control sample: Non-diabetic
        healthy_sample = [1.0, 85.0, 66.0, 29.0, 26.0, 26.6, 0.351, 31.0]
        res_h = predict_diabetes(self.diabetes_model, healthy_sample)
        self.assertFalse(res_h['is_positive'])
        self.assertEqual(res_h['prediction'], 0)
        self.assertEqual(res_h['label'], 'Non-Diabetic')

        # High-risk sample: Diabetic
        risk_sample = [6.0, 148.0, 72.0, 35.0, 168.0, 33.6, 0.627, 50.0]
        res_r = predict_diabetes(self.diabetes_model, risk_sample)
        self.assertTrue(res_r['is_positive'])
        self.assertEqual(res_r['prediction'], 1)
        self.assertEqual(res_r['label'], 'Diabetic')

    def test_heart_predictions(self):
        # Documented sample from heart.ipynb Section 9 (Row 167 in Dataset/heart.csv, target=0)
        healthy_sample = [62.0, 0.0, 0.0, 140.0, 268.0, 0.0, 0.0, 160.0, 0.0, 3.6, 0.0, 2.0, 2.0]
        res_h = predict_heart(self.heart_model, healthy_sample)
        self.assertFalse(res_h['is_positive'])
        self.assertEqual(res_h['prediction'], 0)
        self.assertIn('probabilities', res_h)
        self.assertGreater(res_h['probabilities']['healthy'], 0.5)

        # High-risk sample from Dataset/heart.csv Row 0 (target=1)
        risk_sample = [63.0, 1.0, 3.0, 145.0, 233.0, 1.0, 0.0, 150.0, 0.0, 2.3, 0.0, 0.0, 1.0]
        res_r = predict_heart(self.heart_model, risk_sample)
        self.assertTrue(res_r['is_positive'])
        self.assertEqual(res_r['prediction'], 1)
        self.assertGreater(res_r['probabilities']['disease'], 0.5)

    def test_parkinsons_predictions(self):
        # Documented sample from parkinsons.ipynb Section 7 (Row 30, Subject S07 healthy control)
        healthy_sample = [
            197.076, 206.896, 192.055, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498,
            0.01098, 0.097, 0.00563, 0.0068, 0.00802, 0.01689, 0.00339, 26.775,
            0.422229, 0.741367, -7.3483, 0.177551, 1.743867, 0.085569
        ]
        res_h = predict_parkinsons(self.parkinsons_model, healthy_sample)
        self.assertFalse(res_h['is_positive'])
        self.assertEqual(res_h['prediction'], 0)
        self.assertLess(res_h['decision_score'], 0.0)

        # Documented sample from Dataset/parkinsons.csv Row 0 (Subject S01 Parkinson's patient)
        risk_sample = [
            119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
            0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
            0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654
        ]
        res_r = predict_parkinsons(self.parkinsons_model, risk_sample)
        self.assertTrue(res_r['is_positive'])
        self.assertEqual(res_r['prediction'], 1)
        self.assertGreater(res_r['decision_score'], 0.0)


class TestPreprocessingConsistency(unittest.TestCase):
    """Verifies that zero-value conversion to NaN matches training imputation logic."""

    @classmethod
    def setUpClass(cls):
        cls.models_dir = get_default_models_dir()
        cls.diabetes_model, _, _ = load_all_models(cls.models_dir)

    def test_zero_insulin_is_imputed_identically(self):
        # A sample with Insulin=0 (which is physiologically unmeasured in PIMA)
        sample_with_zero = [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]
        
        # Test directly via inference module
        res = predict_diabetes(self.diabetes_model, sample_with_zero)
        
        # Now manually test with NaN replacement as done in notebook
        manual_df = pd.DataFrame([sample_with_zero], columns=DIABETES_FEATURES)
        manual_df[DIABETES_ZERO_AS_NAN_COLS] = manual_df[DIABETES_ZERO_AS_NAN_COLS].replace(0, np.nan)
        manual_decision = float(self.diabetes_model.decision_function(manual_df)[0])
        
        self.assertAlmostEqual(res['decision_score'], manual_decision, places=5)


if __name__ == '__main__':
    unittest.main()
