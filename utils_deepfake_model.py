import numpy as np
from PIL import Image
import torch
from transformers import pipeline

# A solid real-vs-fake image model (we'll run it on face crops)
MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

def load_detector():
    """
    Loads once. First run will download the model from Hugging Face.
    """
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("image-classification", model=MODEL_NAME, device=device)
    return clf

def predict_faces(detector, face_crops_rgb):
    """
    face_crops_rgb: list of numpy arrays (RGB, 224x224)
    returns:
      - per_face: list of dicts: {"fake": float, "real": float}
      - avg_fake: float (0-100)
      - verdict: str
    """
    if not face_crops_rgb:
        return [], 0.0, "No faces found"

    per_face = []
    fake_probs = []

    for face in face_crops_rgb:
        img = Image.fromarray(face.astype("uint8"), mode="RGB")
        out = detector(img)

        # out example: [{'label': 'Real', 'score': 0.98}, {'label': 'Fake', 'score': 0.02}]
        scores = {"fake": 0.0, "real": 0.0}
        for item in out:
            label = item["label"].strip().lower()
            if "real" in label:
                scores["real"] = float(item["score"])
            elif "fake" in label:
                scores["fake"] = float(item["score"])

        # If model labels are reversed/odd, make sure they sum
        if scores["fake"] == 0.0 and scores["real"] == 0.0 and len(out) >= 1:
            # fallback: assume first item is the predicted label
            top = out[0]["label"].strip().lower()
            if "fake" in top:
                scores["fake"] = float(out[0]["score"])
                scores["real"] = 1.0 - scores["fake"]
            else:
                scores["real"] = float(out[0]["score"])
                scores["fake"] = 1.0 - scores["real"]

        per_face.append(scores)
        fake_probs.append(scores["fake"])

    avg_fake = float(np.mean(fake_probs)) * 100.0

    if avg_fake >= 60:
        verdict = "Likely FAKE (deepfake/manipulated)"
    elif avg_fake <= 40:
        verdict = "Likely REAL"
    else:
        verdict = "Uncertain (mixed signals)"

    return per_face, avg_fake, verdict
