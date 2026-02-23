# utils_identity.py (CLOUD-SAFE)
import os
import numpy as np

# Detect Streamlit Cloud (Streamlit sets this env var)
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
    prefer_gpu: bool = False,
    insightface_home: str = "./.insightface",
):
    """
    Returns (det_model, rec_model).
    On Streamlit Cloud: returns (None, None) safely (no insightface dependency).
    """
    if ON_STREAMLIT_CLOUD:
        # Cloud demo: identity module disabled
        return None, None

    # ✅ Lazy import so Cloud doesn't crash
    from insightface.app import FaceAnalysis

    os.environ.setdefault("INSIGHTFACE_HOME", insightface_home)

    ctx_id = _pick_ctx_id(prefer_gpu)

    det_model = FaceAnalysis(name="buffalo_s")
    det_model.prepare(ctx_id=ctx_id, det_size=det_size)

    rec_model = None  # kept for compatibility with app.py
    return det_model, rec_model


def get_face_embedding(bgr_img, det_model, rec_model=None):
    """
    Returns normalized embedding (np.ndarray).
    """
    if det_model is None:
        raise RuntimeError("Identity module disabled (det_model is None)")

    if bgr_img is None or getattr(bgr_img, "size", 0) == 0:
        raise ValueError("Empty image provided")

    faces = det_model.get(bgr_img)
    if not faces:
        raise ValueError("No face detected")

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