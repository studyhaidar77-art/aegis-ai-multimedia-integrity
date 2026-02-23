# utils_deepfake_model.py
import numpy as np
from PIL import Image

# Cloud-safe mode:
# Streamlit Cloud is currently using Python 3.13 in your build logs,
# and torch/transformers wheels don't match -> deployment fails.
# So we disable the HF model in the public demo.

MODEL_NAME = "dima806/deepfake_vs_real_image_detection"


def load_detector():
    """
    Cloud-safe detector loader.
    Returns None intentionally to avoid torch/transformers dependency on Streamlit Cloud.
    (Your app will fall back to heuristic signals.)
    """
    return None


def predict_faces(detector, face_crops_rgb):
    """
    Cloud-safe prediction:
    - If detector is None, return a clear message (no crash).
    - Keeps function signature same so app.py works without changes.
    """

    if detector is None:
        return [], 0.0, "Deepfake ML model disabled on Streamlit Cloud demo"

    if not face_crops_rgb:
        return [], 0.0, "No faces found"

    # If in future you enable a detector, you can put real prediction here.
    # For now, return safe fallback.
    return [], 0.0, "Deepfake ML model not active"