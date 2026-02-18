# utils_identity.py
import os
import numpy as np
from insightface.app import FaceAnalysis

# ✅ Detect Streamlit Cloud
ON_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SERVER_RUNNING") == "true"


def _pick_ctx_id(prefer_gpu: bool = True) -> int:
    """-1 = CPU, 0 = GPU (if CUDA provider is available)"""
    if not prefer_gpu:
        return -1
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return 0
    except Exception:
        pass
    return -1


def load_det_rec_models(
    det_size=(640, 640),
    prefer_gpu: bool = False,   # ✅ default CPU for Streamlit Cloud
    insightface_home: str = "./.insightface",
):
    # ✅ HARD BLOCK InsightFace on Streamlit Cloud demo
    if ON_STREAMLIT_CLOUD:
        raise RuntimeError("InsightFace disabled on Streamlit Cloud demo (model download too heavy).")

    os.environ.setdefault("INSIGHTFACE_HOME", insightface_home)

    ctx_id = _pick_ctx_id(prefer_gpu)

    # ✅ Debug prints for local logs
    print("📥 Loading InsightFace model (buffalo_s)...")
    det_model = FaceAnalysis(name="buffalo_s")
    print("✅ FaceAnalysis created. Preparing...")
    det_model.prepare(ctx_id=ctx_id, det_size=det_size)
    print("✅ InsightFace ready.")

    # compatibility with app.py
    rec_model = None
    return det_model, rec_model


def get_face_embedding(bgr_img, det_model, rec_model=None):
    if bgr_img is None or getattr(bgr_img, "size", 0) == 0:
        raise ValueError("Empty image provided")

    faces = det_model.get(bgr_img)
    if not faces:
        raise ValueError("No face detected")

    # biggest face
    face = max(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
    )

    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        raise ValueError("Face embedding not available")

    emb = np.asarray(emb).reshape(-1).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb
