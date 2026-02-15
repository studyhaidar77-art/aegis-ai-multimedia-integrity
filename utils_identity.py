import os
import numpy as np
import cv2

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


def load_det_rec_models():
    # Use local cache folder (good for cloud too)
    os.environ.setdefault("INSIGHTFACE_HOME", "./.insightface")

    # Try NEW insightface API (providers supported)
    try:
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
    except TypeError:
        # Fallback OLD insightface API (no providers arg)
        app = FaceAnalysis(name="buffalo_l")

    # ctx_id: -1 = CPU
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # Recognition model (ArcFace)
    rec_model = get_model("buffalo_l")
    try:
        rec_model.prepare(ctx_id=-1)
    except Exception:
        pass

    return app, rec_model


def get_face_embedding(bgr_img, det_model, rec_model):
    # det_model here is FaceAnalysis app
    faces = det_model.get(bgr_img)
    if not faces:
        raise ValueError("No face detected")

    # take biggest face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding  # already normalized in many versions
    emb = np.asarray(emb).reshape(-1).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb
