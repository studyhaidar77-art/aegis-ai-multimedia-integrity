import numpy as np
from PIL import Image

# DO NOT import torch or transformers here
# Lazy loading prevents Streamlit crashes

MODEL_NAME = "dima806/deepfake_vs_real_image_detection"


def load_detector():
    """
    Loads the deepfake model safely.
    If loading fails (common on Streamlit Cloud),
    we return None instead of crashing the app.
    """
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1

        clf = pipeline(
            "image-classification",
            model=MODEL_NAME,
            device=device
        )

        return clf

    except Exception as e:
        print("Deepfake model failed to load:", e)
        return None


def predict_faces(detector, face_crops_rgb):
    """
    Runs prediction safely.
    """

    # ✅ If model failed → don't crash app
    if detector is None:
        return [], 0.0, "Deepfake model unavailable"

    if not face_crops_rgb:
        return [], 0.0, "No faces found"

    per_face = []
    fake_probs = []

    for face in face_crops_rgb:
        try:
            img = Image.fromarray(face.astype("uint8"), mode="RGB")
            out = detector(img)

            scores = {"fake": 0.0, "real": 0.0}

            for item in out:
                label = item["label"].strip().lower()

                if "real" in label:
                    scores["real"] = float(item["score"])

                elif "fake" in label:
                    scores["fake"] = float(item["score"])

            # fallback protection
            if scores["fake"] == 0.0 and scores["real"] == 0.0:
                top = out[0]["label"].strip().lower()

                if "fake" in top:
                    scores["fake"] = float(out[0]["score"])
                    scores["real"] = 1.0 - scores["fake"]
                else:
                    scores["real"] = float(out[0]["score"])
                    scores["fake"] = 1.0 - scores["real"]

            per_face.append(scores)
            fake_probs.append(scores["fake"])

        except Exception as e:
            print("Prediction failed:", e)

    if not fake_probs:
        return [], 0.0, "Prediction failed"

    avg_fake = float(np.mean(fake_probs)) * 100.0

    if avg_fake >= 60:
        verdict = "Likely FAKE (deepfake/manipulated)"
    elif avg_fake <= 40:
        verdict = "Likely REAL"
    else:
        verdict = "Uncertain (mixed signals)"

    return per_face, avg_fake, verdict
