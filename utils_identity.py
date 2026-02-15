from pathlib import Path
import numpy as np
import cv2
import insightface
from insightface.utils.face_align import norm_crop


def load_det_rec_models():
    """
    Loads InsightFace detection + recognition ONNX models directly.
    This bypasses FaceAnalysis and avoids the 'detection' assertion error.
    """
    base = Path(__file__).resolve().parent
    model_dir = base / ".insightface" / "models" / "buffalo_l"

    det_path = str(model_dir / "det_10g.onnx")
    rec_path = str(model_dir / "w600k_r50.onnx")

    det_model = insightface.model_zoo.get_model(det_path, providers=["CPUExecutionProvider"])
    rec_model = insightface.model_zoo.get_model(rec_path, providers=["CPUExecutionProvider"])

    # detector prepare (some versions accept det_thresh, some don't)
    try:
        det_model.prepare(ctx_id=-1, det_thresh=0.3, input_size=(1280, 1280))
    except TypeError:
        det_model.prepare(ctx_id=-1, input_size=(1280, 1280))

    rec_model.prepare(ctx_id=-1)

    return det_model, rec_model


def _pick_largest(bboxes: np.ndarray) -> int:
    """Pick the largest detected face box index."""
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    return int(np.argmax(areas))


def get_face_embedding(img_bgr: np.ndarray, det_model, rec_model) -> np.ndarray:
    """
    Detect face -> align (if keypoints exist) -> compute embedding.
    Fallback: bbox crop if keypoints missing.
    Returns normalized embedding.
    """
    bboxes, kpss = det_model.detect(img_bgr, max_num=0)

    if bboxes is None or len(bboxes) == 0:
        raise ValueError("No face detected in this image/frame.")

    i = _pick_largest(bboxes)

    # --- Try keypoint alignment first ---
    if kpss is not None and len(kpss) > i and kpss[i] is not None:
        try:
            kps = kpss[i]
            face = norm_crop(img_bgr, kps)  # aligned face crop (112x112)
            emb = rec_model.get_feat(face).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            return emb
        except Exception:
            pass

    # --- Fallback: bbox crop ---
    x1, y1, x2, y2 = bboxes[i][:4].astype(int)

    h, w = img_bgr.shape[:2]
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Face crop failed.")

    crop = cv2.resize(crop, (112, 112))
    emb = rec_model.get_feat(crop).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))
