import os
import joblib


class ModelLoadError(Exception):
    """Raised when a model artifact cannot be located or loaded."""
    pass


def get_default_models_dir():
    """Returns the absolute path to the saved_models directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, 'saved_models')


def load_all_models(models_dir=None):
    """
    Loads all trained models and pipelines with error handling.
    
    Returns:
        tuple: (diabetes_pipeline, heart_disease_model, parkinsons_pipeline)
    Raises:
        ModelLoadError: If any artifact is missing or corrupted.
    """
    if models_dir is None:
        models_dir = get_default_models_dir()

    diabetes_path = os.path.join(models_dir, 'diabetes_pipeline.joblib')
    # Support both .joblib and legacy .sav for heart disease
    heart_path = os.path.join(models_dir, 'heart_disease_model.joblib')
    if not os.path.exists(heart_path):
        heart_path = os.path.join(models_dir, 'heart_disease_model.sav')
    parkinsons_path = os.path.join(models_dir, 'parkinsons_pipeline.joblib')

    missing = []
    for label, path in [
        ("Diabetes Pipeline", diabetes_path),
        ("Heart Disease Model", heart_path),
        ("Parkinson's Pipeline", parkinsons_path)
    ]:
        if not os.path.exists(path):
            missing.append(f"{label} ({path})")

    if missing:
        raise ModelLoadError(f"Missing required model artifact(s): {', '.join(missing)}")

    try:
        diabetes_pipeline = joblib.load(diabetes_path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load Diabetes pipeline from {diabetes_path}: {e}")

    try:
        heart_model = joblib.load(heart_path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load Heart model from {heart_path}: {e}")

    try:
        parkinsons_pipeline = joblib.load(parkinsons_path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load Parkinson's pipeline from {parkinsons_path}: {e}")

    return diabetes_pipeline, heart_model, parkinsons_pipeline
