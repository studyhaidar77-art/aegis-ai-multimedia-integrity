import numpy as np
import cv2
from insightface.app import FaceAnalysis


# -------------------------
# Load Models (AUTO)
# -------------------------
def load_det_rec_models():
    """
    Automatically downloads + loads InsightFace models.
    No manual ONNX paths needed.
    """

    app = FaceAnalysis(
        name="buffalo_l",  # best balance of speed + accuracy
        providers=["CPUExecutionProvider"]
    )

    # ctx_id = -1 → CPU
    # (use 0 if you ever get GPU)
    app.prepare(ctx_id=-1, det_size=(640, 640))

    return app, None


# -------------------------
# Face Embedding
# -------------------------
def get_face_embedding(img_bgr, det_model, rec_model=None):

    faces = det_model.get(img_bgr)

    if not faces:
        raise ValueError("No face detected.")

    # pick largest face
    def area(face):
        x1, y1, x2, y2 = face.bbox.astype(int)
        return (x2 - x1) * (y2 - y1)

    face = max(faces, key=area)

    emb = face.normed_embedding

    if emb is None:
        emb = face.embedding
        emb = emb / (np.linalg.norm(emb) + 1e-9)

    return emb.astype(np.float32).reshape(-1)


# -------------------------
# Cosine Distance
# -------------------------
def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))
